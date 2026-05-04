"""End-to-end audit + quantize + benchmark for HuggingFace models.

Two task paths, dispatched on model architecture:

  - Image classification (``AutoModelForImageClassification``): top-1 / top-5
    accuracy on a HuggingFace image classification dataset. Default dataset:
    ``benjamin-paine/imagenet-1k-256x256`` validation split.
  - Causal LM (``AutoModelForCausalLM``): WikiText-2-style perplexity on a
    HuggingFace text dataset. Default dataset: ``wikitext`` /
    ``wikitext-2-raw-v1`` test split.

CLI::

    python -m apexquant.bench --hf-model <id> [--bits 2 4 6 8]
                                [--methods apexquant quarot rtn_absmax]
                                [--hf-dataset <id>] [--subset-size N]
                                [--out results.jsonl]

The pipeline is: load -> audit (verdict printed) -> eval reference ->
for each (method, bits): deepcopy -> quantize -> eval -> log.
"""

from apexquant.bench._eval import BenchResult, format_summary, write_jsonl
from apexquant.bench.llm import benchmark_causal_lm
from apexquant.bench.vision import benchmark_image_classifier

__all__ = [
    "BenchResult",
    "benchmark_causal_lm",
    "benchmark_image_classifier",
    "format_summary",
    "write_jsonl",
]
