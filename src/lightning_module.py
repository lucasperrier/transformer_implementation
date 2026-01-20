from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
import lightning as L

from .transformer import Transformer


def noam_lr(step: int, d_model: int, warmup_steps: int) -> float:
    # step is 1-based in the paper; avoid div by zero
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size: int, padding_idx: int, epsilon: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B,T,V), target: (B,T)
        B, T, V = logits.shape
        logits = logits.view(B * T, V)
        target = target.view(B * T)

        # ignore padding
        non_pad = target != self.padding_idx
        logits = logits[non_pad]
        target = target[non_pad]

        if logits.numel() == 0:
            return logits.sum()  # 0

        log_probs = F.log_softmax(logits, dim=-1)

        # smoothed NLL
        nll = F.nll_loss(log_probs, target, reduction="mean")
        smooth = -log_probs.mean(dim=-1).mean()
        return (1.0 - self.epsilon) * nll + self.epsilon * smooth


class TransformerMTModule(L.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 3,
        d_ff: int = 1024,
        dropout: float = 0.1,
        warmup_steps: int = 4000,
        lr_factor: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Transformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_len=512,
            tie_embeddings=True,
        )
        self.loss_fn = LabelSmoothingLoss(vocab_size=vocab_size, padding_idx=pad_id, epsilon=label_smoothing)

    def training_step(self, batch, batch_idx):
        logits = self._forward_batch(batch)
        loss = self.loss_fn(logits, batch["tgt_out"])
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self._forward_batch(batch)
        loss = self.loss_fn(logits, batch["tgt_out"])
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def _forward_batch(self, batch):
        src = batch["src"]
        tgt_in = batch["tgt_in"]

        src_mask = batch["src_pad_mask"]
        # decoder mask: padding AND causal
        tgt_mask = batch["tgt_pad_mask"] & batch["tgt_causal_mask"].to(batch["tgt_pad_mask"].device)

        return self.model(src, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9)

        def lr_lambda(step):
            # Lightning passes step starting at 0
            return self.hparams.lr_factor * noam_lr(step + 1, self.hparams.d_model, self.hparams.warmup_steps)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "step",
                "frequency": 1,
            },
        }
