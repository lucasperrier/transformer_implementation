from __future__ import annotations

from pathlib import Path
import os

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data_module import WMT14EnDeDataModule
from src.lightning_module import TransformerMTModule


def main():
    # CPU-friendly config (good for iteration & CV artifacts). You can scale up later.
    dm = WMT14EnDeDataModule(
        # Canonical path in this repo
        data_dir="data/raw/wmt14_en_de",
        bpe_merges=32000,
        max_len=64,
        vocab_size=20000,
        batch_size=16,
        num_workers=0,
    )
    dm.setup()

    model = TransformerMTModule(
        vocab_size=len(dm.vocab.itos),
        pad_id=dm.vocab.pad_id,
        d_model=256,
        n_heads=4,
        num_layers=3,
        d_ff=512,
        dropout=0.1,
        warmup_steps=4000,
        lr_factor=1.0,
        label_smoothing=0.1,
    )

    mlf_logger = MLFlowLogger(
        experiment_name="attention_is_all_you_need_cpu",
        tracking_uri="file:./mlruns",
        run_name="tiny_transformer_sanity",
    )

    ckpt = ModelCheckpoint(
        dirpath=str(Path("checkpoints")),
        filename="{epoch}-{step}-{val/loss:.3f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
    )

    accelerator = "gpu" if os.environ.get("USE_GPU", "0") == "1" else "cpu"
    devices = 1

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=10,
        limit_train_batches=50,
        limit_val_batches=10,
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        logger=mlf_logger,
        callbacks=[ckpt, LearningRateMonitor(logging_interval="step")],
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
