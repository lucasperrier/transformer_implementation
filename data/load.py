import os
import io
from pathlib import Path
from datasets import load_dataset
from sacremoses import MosesPunctNormalizer, MosesTokenizer
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE



# Config
OUTDIR = Path("raw/wmt14_en_de")  # output dir (relative to repo root)
BPE_MERGES = 32000
LANG_SRC = "en"
LANG_TGT = "de"
USE_TRUECASE = False  # set True to add truecasing (requires extra code)
BATCH_WRITE = 10000  # flush file every N lines to avoid large memory use

OUTDIR.mkdir(parents=True, exist_ok=True)


normalizer_en = MosesPunctNormalizer(lang=LANG_SRC)
tokenizer_en = MosesTokenizer(lang=LANG_SRC)
normalizer_de = MosesPunctNormalizer(lang=LANG_TGT)
tokenizer_de = MosesTokenizer(lang=LANG_TGT)

def get_splits():
    dataset = load_dataset("wmt14", "de-en")
    train = dataset["train"]

    valid = dataset.get("validation")
    test = dataset.get("test")

    if valid is None:
        valid = train.select((range(3000)))
    if test is None:
        test = train.select((range(3000,6000)))
    
    return train, valid, test

def normalize_and_tokenize_line(line, lang):
    if lang == LANG_SRC:
        n = normalizer_en.normalize(line)
        return tokenizer_en.tokenize(n, return_str=True)
    else:
        n = normalizer_de.normalize(line)
        return tokenizer_de.tokenize(n, return_str=True)
    
def dump_split(dataset, name, out_dir):
    src_path = out_dir / f"{name}.{LANG_SRC}"
    tgt_path = out_dir / f"{name}.{LANG_TGT}"

    with src_path.open("w", encoding="utf-8") as fs, tgt_path.open("w", encoding="utf-8") as ft:
        for i, ex in enumerate(dataset):
            src = ex["translation"][LANG_SRC].strip()
            tgt = ex["translation"][LANG_TGT].strip()
            if not src or not tgt:
                continue
            src_tok = normalize_and_tokenize_line(src, LANG_SRC)
            tgt_tok = normalize_and_tokenize_line(tgt, LANG_TGT)
            fs.write(src_tok + "\n")
            ft.write(tgt_tok + "\n")
            if (i + 1) % BATCH_WRITE == 0:
                fs.flush()
                ft.flush()
    return src_path, tgt_path
        
def train_bpe(codes_path, files, num_merges=BPE_MERGES):
    with io.open(codes_path, "w", encoding="utf-8") as codes_out:
        inp = io.StringIO()
        for p in files:
            with io.open(p, "r", encoding="utf-8") as fh:
                for line in fh:
                    inp.write(line)
        inp.seek(0)
        learn_bpe(inp, codes_out, num_merges)

def apply_bpe_to_file(codes_path, in_path, out_path):
    with io.open(codes_path, "r", encoding="utf-8") as codes_f:
        bpe = BPE(codes_f)
        with io.open(in_path, "r", encoding="utf-8") as fin, io.open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                fout.write(bpe.process_line(line))

def main():
    train_ds, valid_ds, test_ds = get_splits()
    out_dir = OUTDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Dumping tokenized splits")
    train_src, train_tgt = dump_split(train_ds, "train", out_dir)
    valid_src, valid_tgt = dump_split(valid_ds, "valid", out_dir)
    test_src, test_tgt = dump_split(test_ds, "test", out_dir)

    codes_path = out_dir / f"bpe.{BPE_MERGES}.codes"
    print(f"Learning BPE ({BPE_MERGES}) -> {codes_path} ...")
    train_bpe(str(codes_path), [str(train_src), str(train_tgt)], num_merges=BPE_MERGES)

    # Apply BPE to all splits
    print("Applying BPE to splits...")
    for (inp, lang) in [(train_src, LANG_SRC), (train_tgt, LANG_TGT),
                        (valid_src, LANG_SRC), (valid_tgt, LANG_TGT),
                        (test_src, LANG_SRC), (test_tgt, LANG_TGT)]:
        outp = out_dir / (inp.name + f".bpe{BPE_MERGES}")
        apply_bpe_to_file(str(codes_path), inp, outp)

    print("Done. Files written to:", out_dir.resolve())
    print("Train (bpe):", out_dir / f"train.{LANG_SRC}.bpe{BPE_MERGES}", out_dir / f"train.{LANG_TGT}.bpe{BPE_MERGES}")
    print("Valid (bpe):", out_dir / f"valid.{LANG_SRC}.bpe{BPE_MERGES}", out_dir / f"valid.{LANG_TGT}.bpe{BPE_MERGES}")
    print("Test  (bpe):", out_dir / f"test.{LANG_SRC}.bpe{BPE_MERGES}", out_dir / f"test.{LANG_TGT}.bpe{BPE_MERGES}")
    print("BPE codes:", codes_path)

if __name__ == "__main__":
    main()
