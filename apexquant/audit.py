"""ApexQuant fan-in audit.

Walks a PyTorch model and decides, layer by layer, whether ApexQuant's
Beta-matched codebook is a good fit. The decision is driven entirely by the
fan-in d of each quantizable parameter, because d controls how sharply the
post-rotation Beta(d/2, d/2) density concentrates and therefore how much edge
the matched codebook has over a uniform grid (QuaRot).

The thresholds come from the empirical figure in the paper:
    d >= 100           -> ApexQuant wins decisively at low bit-widths
    32  <= d < 100     -> ApexQuant still helps, margins shrink
    d < 32             -> Beta is too flat, codebook mismatched, prefer QuaRot

Usage::

    from apexquant import audit
    audit(model)                          # prints a report
    report = audit(model, verbose=False)  # returns a ModelReport you can act on

Or from the CLI::

    python -m apexquant.audit --model resnet50
    python -m apexquant.audit --checkpoint path/to/model.pt

Note: this module inspects ``module.in_features``, ``module.kernel_size``,
``module.groups`` and similar structural attributes — it never reads weight
tensors. There is intentionally no ``device`` argument; moving weights to a
device contributes nothing to a structural audit.

TODO: LLaMA / GPT-style architectures from the ``transformers`` library do
not use ``nn.MultiheadAttention``; they implement attention with a custom
module that exposes individual ``nn.Linear`` projections (q_proj, k_proj,
v_proj, o_proj). Those linears are picked up by the standard ``nn.Linear``
branch below, so wrapping the HF model and calling :func:`audit` will work
out of the box. The reason it's a TODO rather than a guarantee is that
some transformers variants pack q+k+v into a fused ``Wqkv`` linear with
non-standard fan-in semantics, which we have not yet validated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn


# Architectural-boundary thresholds, tied to the figure in the paper.
# Anything below SMALL_D is the depthwise-kernel failure regime.
# Anything at or above LARGE_D is the regime where Beta(d/2,d/2) is sharp.
SMALL_D = 32
LARGE_D = 100


# Verdict labels for individual layers and for the whole model.
GOOD = "good"           # d >= LARGE_D  -> ApexQuant strongly preferred
MARGINAL = "marginal"   # SMALL_D <= d < LARGE_D  -> ApexQuant still helps
BAD = "bad"             # d < SMALL_D  -> use QuaRot for this layer
SKIP = "skip"           # not a quantizable parameter at all
EMPTY = "empty"         # model has no quantizable layers (e.g. only LayerNorm)


@dataclass
class LayerReport:
    """One row of the audit table."""
    name: str
    module_type: str
    shape: Tuple[int, ...]
    fan_in: int
    n_params: int
    verdict: str
    note: str = ""


@dataclass
class ModelReport:
    """Aggregated audit across all quantizable layers."""
    layers: list = field(default_factory=list)
    total_params: int = 0
    quantizable_params: int = 0

    # Parameter-weighted breakdown.
    params_good: int = 0
    params_marginal: int = 0
    params_bad: int = 0

    # Layer-count breakdown. This matters separately from param count,
    # because a small-d depthwise layer with few parameters still sits in
    # the forward path of every forward pass — quantization error there
    # propagates to every subsequent layer. MobileNet-V2's depthwise
    # kernels are only 2% of params but kill end-to-end accuracy.
    n_layers_good: int = 0
    n_layers_marginal: int = 0
    n_layers_bad: int = 0
    n_layers_depthwise: int = 0  # subset of the BAD count, called out

    @property
    def good_fraction(self) -> float:
        return self.params_good / self.quantizable_params if self.quantizable_params else 0.0

    @property
    def marginal_fraction(self) -> float:
        return self.params_marginal / self.quantizable_params if self.quantizable_params else 0.0

    @property
    def bad_fraction(self) -> float:
        return self.params_bad / self.quantizable_params if self.quantizable_params else 0.0

    @property
    def n_quantizable_layers(self) -> int:
        return self.n_layers_good + self.n_layers_marginal + self.n_layers_bad

    @property
    def bad_layer_fraction(self) -> float:
        return self.n_layers_bad / self.n_quantizable_layers if self.n_quantizable_layers else 0.0

    @property
    def overall_verdict(self) -> str:
        # An empty model has nothing to audit. The fraction-based rules below
        # would return MARGINAL by accident, which is misleading.
        if self.quantizable_params == 0:
            return EMPTY

        # Decision rule combines two signals:
        #   1. Parameter-weighted bad fraction > 30% -> BAD (a lot of mass
        #      sits in mismatched layers).
        #   2. Layer-count bad fraction > 20% -> BAD (even if those layers
        #      are small in params, they're spread through the network and
        #      each one is a bottleneck the signal must pass through).
        # Either signal triggers BAD. This catches MobileNet-V2 (~30% of
        # layers are d=9 depthwise, but only 2% of params) which signal
        # 1 alone would miss.
        if self.bad_fraction > 0.30 or self.bad_layer_fraction > 0.20:
            return BAD
        if self.good_fraction >= 0.70 and self.bad_layer_fraction < 0.10:
            return GOOD
        return MARGINAL


def _classify_d(d: int) -> str:
    """Map a fan-in to a verdict using the architectural-boundary thresholds."""
    if d >= LARGE_D:
        return GOOD
    if d >= SMALL_D:
        return MARGINAL
    return BAD


def _compute_fan_in(module: nn.Module, name: str) -> Optional[Tuple[int, Tuple[int, ...], int, str]]:
    """Decide whether ``module`` is quantizable, and if so compute its fan-in d.

    Returns ``(d, shape, n_params, note)`` or ``None`` if the module isn't
    quantizable. Conventions match the paper:

      - ``nn.Linear``                       d = in_features
      - standard ``nn.Conv2d``              d = C_in * k_H * k_W
      - depthwise ``nn.Conv2d`` (groups=Cin) d = k_H * k_W (the failure mode)
      - grouped ``nn.Conv2d``               d = (C_in / groups) * k_H * k_W
      - ``nn.MultiheadAttention.in_proj_weight`` d = embed_dim

    Other module types are skipped — LayerNorm, BatchNorm, embeddings, biases.
    The paper does not quantize them.
    """
    if isinstance(module, nn.Linear):
        d = module.in_features
        shape = tuple(module.weight.shape)
        n = module.weight.numel()
        return d, shape, n, ""

    if isinstance(module, nn.Conv2d):
        c_in = module.in_channels
        c_out = module.out_channels
        k_h, k_w = module.kernel_size
        groups = module.groups

        # Depthwise convolution: each output channel sees only k_h*k_w
        # input values, not C_in*k_h*k_w. This is the small-d failure case.
        if groups == c_in and groups == c_out and groups > 1:
            d = k_h * k_w
            shape = tuple(module.weight.shape)
            n = module.weight.numel()
            return d, shape, n, "depthwise"

        # Grouped (but not depthwise) convolution.
        if groups > 1:
            d = (c_in // groups) * k_h * k_w
            shape = tuple(module.weight.shape)
            n = module.weight.numel()
            return d, shape, n, f"grouped (g={groups})"

        # Standard convolution.
        d = c_in * k_h * k_w
        shape = tuple(module.weight.shape)
        n = module.weight.numel()
        return d, shape, n, ""

    if isinstance(module, nn.MultiheadAttention):
        # The combined Q/K/V projection is exposed as `in_proj_weight` of
        # shape (3*embed_dim, embed_dim). The paper treats it as a linear
        # weight with fan-in equal to embed_dim.
        if module.in_proj_weight is not None:
            d = module.embed_dim
            shape = tuple(module.in_proj_weight.shape)
            n = module.in_proj_weight.numel()
            return d, shape, n, "MHA in_proj"

    return None


def audit(model: nn.Module, verbose: bool = True) -> ModelReport:
    """Walk every submodule, classify each quantizable layer by its fan-in d,
    and return a :class:`ModelReport`. If ``verbose``, print a human-readable
    table.
    """
    report = ModelReport()
    report.total_params = sum(p.numel() for p in model.parameters())

    seen: set = set()  # avoid double-counting parameters shared across modules

    for name, module in model.named_modules():
        info = _compute_fan_in(module, name)
        if info is None:
            continue

        # Identify the parameter tensor we'd actually be quantizing, so we
        # can de-dupe in case of weight sharing.
        if isinstance(module, nn.MultiheadAttention):
            param = module.in_proj_weight
        elif hasattr(module, "weight"):
            param = module.weight
        else:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))

        d, shape, n_params, note = info
        verdict = _classify_d(d)

        report.layers.append(LayerReport(
            name=name or "(root)",
            module_type=type(module).__name__,
            shape=shape,
            fan_in=d,
            n_params=n_params,
            verdict=verdict,
            note=note,
        ))

        report.quantizable_params += n_params
        if verdict == GOOD:
            report.params_good += n_params
            report.n_layers_good += 1
        elif verdict == MARGINAL:
            report.params_marginal += n_params
            report.n_layers_marginal += 1
        elif verdict == BAD:
            report.params_bad += n_params
            report.n_layers_bad += 1
            if note == "depthwise":
                report.n_layers_depthwise += 1

    if verbose:
        _print_report(report)
    return report


def _print_report(report: ModelReport) -> None:
    """Pretty-print the audit. No bells, just columns that line up."""
    if report.overall_verdict == EMPTY:
        print()
        print("VERDICT: Nothing to quantize.")
        print("  No nn.Linear, nn.Conv2d, or nn.MultiheadAttention layers were")
        print("  found in this model. Nothing for ApexQuant or QuaRot to act on.")
        print()
        return

    name_w = max(len(l.name) for l in report.layers)
    name_w = max(name_w, 4)
    type_w = max(len(l.module_type) for l in report.layers)
    type_w = max(type_w, 6)

    print()
    print(f"{'Layer'.ljust(name_w)}  {'Type'.ljust(type_w)}  {'d':>7}  {'#params':>12}  Verdict")
    print(f"{'-' * name_w}  {'-' * type_w}  {'-' * 7}  {'-' * 12}  -------")

    for l in report.layers:
        flag = {GOOD: "GOOD", MARGINAL: "MARG", BAD: "BAD "}[l.verdict]
        suffix = f"  ({l.note})" if l.note else ""
        print(f"{l.name.ljust(name_w)}  {l.module_type.ljust(type_w)}  "
              f"{l.fan_in:>7}  {l.n_params:>12,}  {flag}{suffix}")

    print()
    print("Parameter-weighted breakdown:")
    print(f"  total params              {report.total_params:>14,}")
    print(f"  quantizable params        {report.quantizable_params:>14,}  "
          f"({100 * report.quantizable_params / report.total_params:.1f}% of total)")
    print(f"  in GOOD layers (d >= {LARGE_D})  {report.params_good:>14,}  "
          f"({100 * report.good_fraction:.1f}%)")
    print(f"  in MARG layers (d >= {SMALL_D})  {report.params_marginal:>14,}  "
          f"({100 * report.marginal_fraction:.1f}%)")
    print(f"  in BAD  layers (d < {SMALL_D})   {report.params_bad:>14,}  "
          f"({100 * report.bad_fraction:.1f}%)")
    print()
    print("Layer-count breakdown:")
    print(f"  GOOD layers               {report.n_layers_good:>14,}")
    print(f"  MARG layers               {report.n_layers_marginal:>14,}")
    print(f"  BAD  layers               {report.n_layers_bad:>14,}  "
          f"({100 * report.bad_layer_fraction:.1f}%)")
    if report.n_layers_depthwise:
        print(f"    of which depthwise      {report.n_layers_depthwise:>14,}")
    print()

    verdict = report.overall_verdict
    if verdict == GOOD:
        print(f"VERDICT: ApexQuant recommended.")
        print(f"  At least 70% of weight mass sits in d >= {LARGE_D} layers, where the")
        print(f"  Beta(d/2, d/2) approximation is sharp and the matched codebook")
        print(f"  outperforms QuaRot decisively at 2-4 bits.")
    elif verdict == BAD:
        print(f"VERDICT: Use QuaRot instead.")
        print(f"  Either >30% of weight mass or >20% of layers sit at d < {SMALL_D}")
        print(f"  (typically depthwise convolutions). The Beta density at small d")
        print(f"  is nearly flat, so ApexQuant's matched codebook offers no edge")
        print(f"  and may underperform QuaRot's uniform grid (see MobileNet-V2 in")
        print(f"  the paper). Even small-param-count layers matter here because")
        print(f"  every forward pass routes through them.")
    else:
        print(f"VERDICT: Mixed. ApexQuant for large-d layers, QuaRot for small-d.")
        print(f"  Consider a per-layer policy: apply ApexQuant where d >= {LARGE_D}")
        print(f"  and QuaRot where d < {SMALL_D}. Layers in [{SMALL_D}, {LARGE_D}) can go")
        print(f"  either way; pick based on bit budget (ApexQuant matters more")
        print(f"  at lower bit-widths).")
    print()


def _build_demo_model(name: str) -> nn.Module:
    """Load a torchvision model by name for the CLI."""
    import torchvision.models as tvm
    builders = {
        "resnet18": tvm.resnet18,
        "resnet50": tvm.resnet50,
        "vit_b_16": tvm.vit_b_16,
        "convnext_tiny": tvm.convnext_tiny,
        "mobilenet_v2": tvm.mobilenet_v2,
        "efficientnet_b0": tvm.efficientnet_b0,
    }
    if name not in builders:
        raise ValueError(f"Unknown model {name!r}. Choose from: {sorted(builders)}")
    return builders[name](weights=None)


def _load_checkpoint(path: str) -> nn.Module:
    """Load a checkpoint file. Accepts a saved nn.Module; rejects state_dicts."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        return obj
    raise SystemExit(
        "--checkpoint must contain a full nn.Module, not a state_dict. "
        "Reconstruct the model in code and call audit() directly."
    )


def _load_hf_model(model_id: str) -> nn.Module:
    """Load a HuggingFace causal LM by repo ID. Lazy-imports transformers
    so the audit module does not have a hard dependency on it.
    """
    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise SystemExit(
            "--hf-model requires the transformers package. "
            "Install with: pip install transformers"
        )
    return AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--model", default=None,
                     help="torchvision model name (resnet18, resnet50, vit_b_16, "
                          "convnext_tiny, mobilenet_v2, efficientnet_b0). "
                          "Default: resnet50 if neither flag is given.")
    src.add_argument("--checkpoint", default=None,
                     help="path to a torch.save'd nn.Module (not a state_dict). "
                          "Loaded with weights_only=False — only point at "
                          "trusted files.")
    src.add_argument("--hf-model", default=None, dest="hf_model",
                     help="HuggingFace causal LM repo ID, e.g. "
                          "'TinyLlama/TinyLlama-1.1B-Chat-v1.0' or "
                          "'microsoft/phi-1_5'. Requires the transformers package.")
    args = parser.parse_args()

    if args.checkpoint is not None:
        print(f"Auditing checkpoint {args.checkpoint}...")
        model = _load_checkpoint(args.checkpoint)
    elif args.hf_model is not None:
        print(f"Auditing HuggingFace model {args.hf_model}...")
        model = _load_hf_model(args.hf_model)
    else:
        name = args.model or "resnet50"
        print(f"Auditing {name}...")
        model = _build_demo_model(name)

    audit(model)
