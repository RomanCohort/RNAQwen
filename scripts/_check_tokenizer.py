"""核对 pubmedbert 的 vocab.txt 与生成的 tokenizer.json 是否自洽。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _dtensor_shim  # noqa: E402

from transformers import AutoTokenizer, BertTokenizer  # noqa: E402

BERT = Path(r"C:\qwen-api\RNAQwen\models\bert-base-uncased")
PMB = Path(r"C:\qwen-api\RNAQwen\models\pubmedbert")
PROBE = [
    "the RNA regulates gene expression.",
    "BRCA1 mutations in breast cancer",
    "p53",
    "TP53[Gene]",
    "The quick brown fox 123",
    "café",
]

for name, d in (("bert-base-uncased", BERT), ("pubmedbert", PMB)):
    lines = (d / "vocab.txt").read_text(encoding="utf-8").splitlines()
    print(f"===== {name}: vocab.txt {len(lines)} 行, {len(lines)==30522}")
    for special in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"):
        idx = [i for i, t in enumerate(lines) if t == special]
        print(f"    {special:7s} vocab.txt 索引 = {idx}")
    print(f"    lines[0:5] = {lines[0:5]}")

    fast = AutoTokenizer.from_pretrained(str(d))
    slow = BertTokenizer.from_pretrained(str(d), use_fast=False)
    print(f"    fast={type(fast).__name__} cls_id={fast.cls_token_id} sep_id={fast.sep_token_id} "
          f"pad_id={fast.pad_token_id} unk_id={fast.unk_token_id} mask_id={fast.mask_token_id} "
          f"vocab_size={fast.vocab_size}")
    print(f"    slow={type(slow).__name__} cls_id={slow.cls_token_id} sep_id={slow.sep_token_id} "
          f"pad_id={slow.pad_token_id} unk_id={slow.unk_token_id} mask_id={slow.mask_token_id} "
          f"vocab_size={slow.vocab_size}")
    for text in PROBE:
        fi = fast(text)["input_ids"]
        si = slow(text)["input_ids"]
        print(f"      {'OK ' if fi == si else 'DIFF'} {text!r}\n           fast={fi}\n           slow={si}")
print("done")
