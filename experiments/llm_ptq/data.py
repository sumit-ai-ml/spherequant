"""WikiText-2 loader and perplexity evaluator.

Standard protocol for LLM perplexity:
- Concatenate the entire test corpus into one long token sequence.
- Split into non-overlapping chunks of length `seq_len`.
- For each chunk, compute cross-entropy loss on the target tokens.
- Perplexity = exp(mean token loss).

Matches the setup used in GPTQ, AWQ, QuaRot papers.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from datasets import load_dataset


def get_wikitext2_text(split: str = "test") -> str:
    """Return the WikiText-2 corpus as a single string."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(ds["text"])


@torch.no_grad()
def perplexity(model, tokenizer, device: str = "cuda",
               seq_len: int = 2048, max_chunks: Optional[int] = None,
               text: Optional[str] = None) -> float:
    """Compute perplexity on WikiText-2 test set.

    max_chunks: if not None, only use the first N chunks (for smoke testing).
    """
    if text is None:
        text = get_wikitext2_text("test")
    model.eval()
    model = model.to(device)

    # Tokenize the full corpus once
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    total_tokens = input_ids.shape[1]

    n_chunks = total_tokens // seq_len
    if max_chunks is not None:
        n_chunks = min(n_chunks, max_chunks)

    total_loss = 0.0
    total_count = 0
    for i in range(n_chunks):
        start = i * seq_len
        end = start + seq_len
        chunk = input_ids[:, start:end]
        # forward pass with labels for cross-entropy
        out = model(chunk, labels=chunk)
        # HF returns mean loss over the sequence (labels shifted internally)
        # Weight by seq_len to get total
        total_loss += out.loss.float().item() * (seq_len - 1)
        total_count += (seq_len - 1)

    mean_loss = total_loss / total_count
    return float(math.exp(mean_loss))
