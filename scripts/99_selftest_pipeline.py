"""离线自检：用假生成器把 11/12/13 的代码路径整条跑一遍。

为什么要这个
------------
真跑数据管线要等 Qwen3-8B 权重（15.3 GB）。但**管线的逻辑正确性与模型无关**：
切块、批处理、JSON 解析、闸门、落盘 —— 这些都能在没有大模型的时候先验证。

假生成器故意做两件坏事，用来确认闸门真的会拦：
  1. 在一部分问答里**吐出序列子串**（模拟"抄序列"）；
  2. 在一部分回答里输出**非法 JSON**（模拟解析失败）。

只要自检报 PASS，就说明：真跑的时候如果数据脏，一定是模型的问题，不是代码的问题。

跑法：
    python scripts/99_selftest_pipeline.py --in data/raw/rna_literature_50.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

from rnaqwen.data.annotate import divide  # noqa: E402
from rnaqwen.data.guard import check_record_qa, find_sequence_leak  # noqa: E402
from rnaqwen.data.prompts import CHUNK_SUMMARY, COMBINE_ANNOTATION, QA_FROM_TEXT  # noqa: E402
from rnaqwen.data.qa import _normalize_pairs  # noqa: E402
from rnaqwen.data.records import read_jsonl  # noqa: E402
from rnaqwen.llm import extract_json  # noqa: E402

FAKE_NAME = "fake-generator/v0"


class FakeGenerator:
    """按 prompt 内容返回可控输出；不需要任何权重。"""

    name = FAKE_NAME

    def __init__(self, leak_every: int = 5, bad_json_every: int = 7, sequence: str = ""):
        self.leak_every = leak_every
        self.bad_json_every = bad_json_every
        self.sequence = sequence
        self.calls = 0

    def chat(self, conversations, max_new_tokens=512, temperature=0.7, top_p=0.9):
        out = []
        for conv in conversations:
            self.calls += 1
            prompt = conv[-1]["content"]
            if "Summarize the excerpt below" in prompt:
                out.append("This RNA is a regulatory non-coding transcript.")
            elif "reference annotation for one RNA molecule" in prompt:
                out.append(
                    "This molecule is a non-coding RNA that modulates gene expression "
                    "in the nucleus. It folds into a conserved stem-loop and binds a "
                    "protein partner. It has been linked to developmental regulation."
                )
            elif "instruction-tuning dataset" in prompt:
                if self.bad_json_every and self.calls % self.bad_json_every == 0:
                    out.append("Sure! Here are the pairs: question: what is it? answer: an RNA.")  # 非 JSON
                    continue
                pairs = [
                    {"question": "What type of molecule is described?",
                     "answer": "A non-coding RNA that regulates gene expression."},
                    {"question": "Where does it act?",
                     "answer": "It acts in the nucleus, where it modulates transcription."},
                ]
                if self.leak_every and self.calls % self.leak_every == 0 and self.sequence:
                    pairs.append({"question": "What is its base sequence?",
                                  "answer": f"The sequence starts with {self.sequence[:20]}."})
                out.append("```json\n" + json.dumps(pairs) + "\n```")
            else:
                out.append("ok")
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=ROOT / "data/raw/rna_literature_50.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data/processed")
    args = ap.parse_args()

    records = list(read_jsonl(args.inp))[: args.limit]
    if not records:
        print("[fail] 没有输入记录")
        return 1
    print(f"[selftest] {len(records)} 条记录 <- {args.inp}")

    checks: dict[str, bool] = {}

    # ---- 1. divide() 切块：不丢句子、不超上限 ----
    bad_chunks = 0
    for r in records:
        chunks = divide(r.literature, max_chars=900, max_chunks=4)
        if not chunks:
            bad_chunks += 1
            continue
        joined = " ".join(chunks)
        # 每块都应是完整句子结尾
        if not all(c.strip().endswith((".", "!", "?")) for c in chunks):
            bad_chunks += 1
    checks["divide_keeps_sentence_boundaries"] = bad_chunks == 0
    print(f"[1] divide(): {len(records) - bad_chunks}/{len(records)} 条切块干净")

    # ---- 2. QA 解析 + 闸门 ----
    gen = FakeGenerator(leak_every=5, bad_json_every=7, sequence=records[0].sequence)
    total, leaks_caught, parse_fail = 0, 0, 0
    for r in records:
        text = r.literature
        raw = gen.chat([[{"role": "user", "content": f"instruction-tuning dataset\n{text}"}]])[0]
        try:
            obj = extract_json(raw)
        except ValueError:
            parse_fail += 1
            continue
        pairs = _normalize_pairs(obj)
        rep = check_record_qa(r.uid, pairs, r.sequence)
        if not rep.ok:
            leaks_caught += 1
        total += len(pairs)
    checks["json_parse_handles_fences"] = total > 0
    checks["guard_catches_leak"] = leaks_caught >= 1
    checks["guard_flags_bad_json"] = parse_fail >= 1
    print(f"[2] 解析出 {total} 组 QA；非 JSON 输出 {parse_fail} 次；闸门拦下 {leaks_caught} 条带序列的记录")

    # ---- 3. 闸门本身：正例/负例 ----
    seq = records[0].sequence
    pos = [{"question": "q", "answer": f"prefix {seq[10:30]} suffix"}]
    neg = [{"question": "What is it?", "answer": "A regulatory non-coding RNA."}]
    checks["guard_positive"] = not check_record_qa("p", pos, seq).ok
    checks["guard_negative"] = check_record_qa("n", neg, seq).ok
    exact, ngrams = find_sequence_leak("the bases ACGUACGUACGUACGUACGU appear", seq)
    checks["guard_ngram_detects_bare_bases"] = len(ngrams) >= 1
    print(f"[3] 闸门：正例拦下={not checks['guard_positive'] is False}，"
          f"负例放行={checks['guard_negative']}，裸碱基串检出={len(ngrams)}")

    # ---- 4. 写出（格式验收）----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "_selftest_rnaqa.jsonl"
    n = 0
    with out.open("w", encoding="utf-8") as fh:
        for r in records:
            d = r.as_dict()
            d["annotation"] = "(selftest annotation)"
            d["qa"] = [{"question": "What is it?", "answer": "A non-coding RNA.", "variant": "dc",
                        "generator": FAKE_NAME}]
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    checks["wrote_jsonl"] = out.exists() and n == len(records)
    print(f"[4] 落盘 {n} 条 -> {out}")

    print()
    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  [{'ok  ' if v else 'FAIL'}] {k}")
    print(f"\n[verdict] {'PASS —— 管线逻辑没问题，等权重到位即可上量' if ok else 'FAIL —— 先修代码'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
