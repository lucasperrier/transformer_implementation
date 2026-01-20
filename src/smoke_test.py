from __future__ import annotations

import torch

from src.data_module import WMT14EnDeDataModule
from src.lightning_module import TransformerMTModule


def main():
    dm = WMT14EnDeDataModule(
        data_dir="raw/wmt14_en_de",
        bpe_merges=32000,
        max_len=32,
        vocab_size=8000,
        batch_size=4,
    )
    dm.setup()

    batch = next(iter(dm.train_dataloader()))

    model = TransformerMTModule(
        vocab_size=len(dm.vocab.itos),
        pad_id=dm.vocab.pad_id,
        d_model=64,
        n_heads=4,
        num_layers=2,
        d_ff=128,
        dropout=0.1,
        warmup_steps=10,
        lr_factor=1.0,
        label_smoothing=0.1,
    )

    logits = model.model(
        batch["src"],
        batch["tgt_in"],
        src_mask=batch["src_pad_mask"],
        tgt_mask=batch["tgt_pad_mask"] & batch["tgt_causal_mask"],
    )

    assert logits.shape[0] == batch["src"].shape[0]
    assert logits.shape[1] == batch["tgt_in"].shape[1]
    assert logits.shape[2] == len(dm.vocab.itos)

    # loss runs
    loss = model.loss_fn(logits, batch["tgt_out"])
    print("OK logits:", tuple(logits.shape), "loss:", float(loss))


if __name__ == "__main__":
    main()
