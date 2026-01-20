from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L


SPECIAL_TOKENS = {
    "pad": "<pad>",
    "unk": "<unk>",
    "bos": "<bos>",
    "eos": "<eos>",
}


@dataclass(frozen=True)
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[SPECIAL_TOKENS["pad"]]

    @property
    def unk_id(self) -> int:
        return self.stoi[SPECIAL_TOKENS["unk"]]

    @property
    def bos_id(self) -> int:
        return self.stoi[SPECIAL_TOKENS["bos"]]

    @property
    def eos_id(self) -> int:
        return self.stoi[SPECIAL_TOKENS["eos"]]

    def encode(self, line: str) -> List[int]:
        # Input files are BPE'd but still whitespace tokenized.
        toks = line.strip().split()
        return [self.stoi.get(t, self.unk_id) for t in toks]

    def decode(self, ids: List[int]) -> str:
        toks = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                toks.append(SPECIAL_TOKENS["unk"])
            else:
                toks.append(self.itos[i])
        return " ".join(toks)


def build_vocab_from_files(paths: List[Path], max_size: int = 37000, min_freq: int = 1) -> Vocab:
    from collections import Counter

    counter: Counter[str] = Counter()
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                counter.update(line.strip().split())

    # Reserve special tokens first.
    itos: List[str] = [
        SPECIAL_TOKENS["pad"],
        SPECIAL_TOKENS["unk"],
        SPECIAL_TOKENS["bos"],
        SPECIAL_TOKENS["eos"],
    ]

    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok in itos:
            continue
        itos.append(tok)
        if len(itos) >= max_size:
            break

    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


class ParallelTextDataset(Dataset):
    def __init__(self, src_path: Path, tgt_path: Path, vocab: Vocab, max_len: int = 256):
        src_lines = src_path.read_text(encoding="utf-8").splitlines() if src_path.exists() else []
        tgt_lines = tgt_path.read_text(encoding="utf-8").splitlines() if tgt_path.exists() else []

        # Be forgiving: some repos ship partial data. Keep only aligned, non-empty pairs.
        n = min(len(src_lines), len(tgt_lines))
        pairs = [(src_lines[i].strip(), tgt_lines[i].strip()) for i in range(n)]
        pairs = [(s, t) for (s, t) in pairs if s and t]

        self.src_lines = [s for (s, _) in pairs]
        self.tgt_lines = [t for (_, t) in pairs]

        if len(self.src_lines) == 0:
            raise ValueError(
                "No aligned (non-empty) sentence pairs found. "
                f"Checked src={src_path} ({len(src_lines)} lines) and tgt={tgt_path} ({len(tgt_lines)} lines). "
                "You likely need to (re)generate the dataset under data/raw/wmt14_en_de/."
            )
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.src_lines)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        src = self.vocab.encode(self.src_lines[idx])[: self.max_len]
        tgt = self.vocab.encode(self.tgt_lines[idx])[: self.max_len]
        return src, tgt


def collate_parallel(batch: List[Tuple[List[int], List[int]]], vocab: Vocab):
    # Returns tensors:
    # src: (B, S)
    # tgt_in: (B, T)  with <bos> ...
    # tgt_out: (B, T) with ... <eos>
    # src_pad_mask: (B, 1, 1, S)
    # tgt_pad_mask: (B, 1, 1, T)
    # tgt_causal_mask: (1, 1, T, T)

    src_seqs, tgt_seqs = zip(*batch)

    def add_bos_eos(seq: List[int]) -> Tuple[List[int], List[int]]:
        # teacher forcing: input shifted right
        tgt_in = [vocab.bos_id] + seq
        tgt_out = seq + [vocab.eos_id]
        return tgt_in, tgt_out

    tgt_in_seqs, tgt_out_seqs = zip(*(add_bos_eos(s) for s in tgt_seqs))

    src_lens = [len(s) for s in src_seqs]
    tgt_lens = [len(s) for s in tgt_in_seqs]
    max_s = max(src_lens)
    max_t = max(tgt_lens)

    pad = vocab.pad_id

    def pad_to(seqs: List[List[int]], max_len: int) -> torch.Tensor:
        out = torch.full((len(seqs), max_len), pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    src = pad_to(list(src_seqs), max_s)
    tgt_in = pad_to(list(tgt_in_seqs), max_t)
    tgt_out = pad_to(list(tgt_out_seqs), max_t)

    src_pad_mask = (src != pad).unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
    tgt_pad_mask = (tgt_in != pad).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)

    # causal mask: allow attending to self and previous positions
    causal = torch.tril(torch.ones((max_t, max_t), dtype=torch.bool)).unsqueeze(0).unsqueeze(0)  # (1,1,T,T)

    return {
        "src": src,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "src_pad_mask": src_pad_mask,
        "tgt_pad_mask": tgt_pad_mask,
        "tgt_causal_mask": causal,
    }


class WMT14EnDeDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/raw/wmt14_en_de",
        bpe_merges: int = 32000,
        max_len: int = 128,
        vocab_size: int = 37000,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.bpe_merges = bpe_merges
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.vocab: Vocab | None = None

    def setup(self, stage: str | None = None):
        # Use the BPE'd files if they exist; otherwise fall back to non-BPE.
        train_src = self.data_dir / f"train.en.bpe{self.bpe_merges}"
        train_tgt = self.data_dir / f"train.de.bpe{self.bpe_merges}"
        if not train_src.exists():
            train_src = self.data_dir / "train.en"
        if not train_tgt.exists():
            train_tgt = self.data_dir / "train.de"

        valid_src = self.data_dir / f"valid.en.bpe{self.bpe_merges}"
        valid_tgt = self.data_dir / f"valid.de.bpe{self.bpe_merges}"
        if not valid_src.exists():
            valid_src = self.data_dir / "valid.en"
        if not valid_tgt.exists():
            valid_tgt = self.data_dir / "valid.de"

        self.vocab = build_vocab_from_files([train_src, train_tgt], max_size=self.vocab_size)

        self.train_ds = ParallelTextDataset(train_src, train_tgt, vocab=self.vocab, max_len=self.max_len)
        self.valid_ds = ParallelTextDataset(valid_src, valid_tgt, vocab=self.vocab, max_len=self.max_len)

    def train_dataloader(self):
        assert self.vocab is not None
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda b: collate_parallel(b, self.vocab),
            pin_memory=False,
        )

    def val_dataloader(self):
        assert self.vocab is not None
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda b: collate_parallel(b, self.vocab),
            pin_memory=False,
        )
