# Implementation Notes

## Goal

Reimplement core components of the Transformer (Vaswani et al., 2017) with:

- correctness first (masks, shapes, training loop)
- a clean training interface via **PyTorch Lightning**
- reproducible experiment tracking via **MLflow**

## What’s implemented

- Encoder–decoder Transformer with:
  - Multi-Head Attention
  - sinusoidal positional encoding
  - post-norm residual connections (LayerNorm(x + sublayer(x)))
  - tied input/output embeddings
- Training features:
  - Adam(β1=0.9, β2=0.98, ε=1e-9)
  - Noam learning rate schedule with warmup
  - label smoothing loss (ε=0.1)
- Data pipeline:
  - reads whitespace-tokenized BPE text files
  - builds a shared vocabulary from train source+target
  - pads and produces padding + causal masks for attention

## What differs from the paper (current)

This repo currently focuses on a **CPU-friendly** configuration.

Missing paper-critical pieces:

- Beam search decoding + length penalty
- SacreBLEU evaluation on `newstest2014`
- checkpoint averaging
- batching by tokens / length bucketing

## How to reproduce a run

1) Install dependencies (see README)

2) (Optional) start MLflow UI

3) Run training:

- `python -m src.train`

Artifacts:

- MLflow runs under `./mlruns`
- checkpoints under `./checkpoints`

## Reference

Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. arXiv:1706.03762
