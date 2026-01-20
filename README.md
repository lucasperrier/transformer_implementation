# Attention Is All You Need — Transformer (PyTorch Lightning)

Reimplementation project inspired by **Vaswani et al., 2017** (“Attention Is All You Need”, arXiv:1706.03762).

This repo contains:

- A Transformer encoder–decoder implementation
- A WMT14 En→De data pipeline (BPE) wired for local training
- Training loop in **PyTorch Lightning**
- Experiment tracking with **MLflow**

> Status: trains end-to-end; currently focused on correctness + reproducibility (CPU-friendly defaults).

## Quickstart (CPU sanity run)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

### Data

This project expects WMT14 En-De BPE files under:

- `data/raw/wmt14_en_de/train.en.bpe32000`
- `data/raw/wmt14_en_de/train.de.bpe32000`

If these files are missing or empty, you can regenerate them via:

```bash
python3 data/load.py
```

### (Optional) MLflow UI

In a separate terminal:

```bash
mlflow ui --backend-store-uri "file:./mlruns" --host 127.0.0.1 --port 5000
```

Open: http://127.0.0.1:5000

### Train

```bash
python3 -m src.train
```

Artifacts:

- MLflow runs in `./mlruns`
- checkpoints in `./checkpoints`

## What’s implemented

- Encoder–decoder Transformer
- sinusoidal positional encoding
- attention masks (padding + causal)
- Noam learning rate schedule + label smoothing
- Lightning training loop + MLflow logging

## What’s missing for paper-quality BLEU (next work)

- Beam search decoding + length penalty
- SacreBLEU evaluation on `newstest2014`
- checkpoint averaging
- token-based batching / length bucketing

## Docs

See `docs/implementation_notes.md` for a short write-up.

## Reference

Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. arXiv:1706.03762
# transformer_implementation
