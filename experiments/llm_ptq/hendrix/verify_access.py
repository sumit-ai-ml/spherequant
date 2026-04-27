"""Sanity check before submitting the SLURM jobs.

Verifies:
  1. HuggingFace login works and has access to Llama 3.
  2. Required Python packages import cleanly.
  3. WikiText-2 dataset is reachable.
  4. lm-eval-harness is installed and at a compatible version.

Run this on a Hendrix login node:

    conda activate h2quant
    python verify_access.py

It does NOT download the model weights, only the tokenizer config, so it is
fast and uses no GPU.
"""

from __future__ import annotations

import importlib
import sys


def check(name: str, fn):
    try:
        result = fn()
        print(f"  [OK]  {name}: {result}")
        return True
    except Exception as e:
        print(f"  [!!]  {name}: {type(e).__name__}: {e}")
        return False


def main():
    ok = True

    print("Python packages:")
    for pkg in ["torch", "transformers", "accelerate", "datasets", "scipy",
                "numpy", "lm_eval"]:
        try:
            m = importlib.import_module(pkg)
            ver = getattr(m, "__version__", "?")
            print(f"  [OK]  {pkg} {ver}")
        except ImportError as e:
            print(f"  [!!]  {pkg} not installed: {e}")
            ok = False

    print("\nGPU:")
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  [OK]  cuda:{i} = {torch.cuda.get_device_name(i)}")
    else:
        print("  [!!]  CUDA not available. You are probably on a login node; "
              "that is fine for this script, but the real jobs need GPU.")

    print("\nHuggingFace login:")
    try:
        from huggingface_hub import whoami
        info = whoami()
        print(f"  [OK]  logged in as {info['name']}")
    except Exception as e:
        print(f"  [!!]  not logged in: {e}")
        print("         run: huggingface-cli login")
        ok = False

    print("\nTokenizer access for Llama 3 (no weights downloaded):")
    from transformers import AutoTokenizer
    for model_id in ["meta-llama/Meta-Llama-3-8B",
                     "meta-llama/Meta-Llama-3-70B"]:
        ok &= check(model_id,
                    lambda mid=model_id: type(
                        AutoTokenizer.from_pretrained(mid)).__name__)

    print("\nWikiText-2 dataset:")
    def load_wt2():
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        return f"{len(ds)} rows, first row = {ds[0]['text'][:40]!r}"
    ok &= check("wikitext-2-raw-v1", load_wt2)

    print("\nlm-eval version compatibility:")
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM  # noqa: F401
        print(f"  [OK]  lm_eval {lm_eval.__version__} HFLM imports cleanly")
    except Exception as e:
        print(f"  [!!]  lm-eval setup: {e}")
        ok = False

    print("\n" + ("Ready to submit SLURM jobs." if ok
                  else "Fix the [!!] items above before submitting."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
