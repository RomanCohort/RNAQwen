"""Is the zeroed-code arm's constant answer the *question prior*?

The claim under test: with the code channel cut, the model emits one answer per
template regardless of the sequence, and that answer is the template's most frequent
training answer.  That is checkable without the LLM: count the training answers.

Reads the same corpus the training run used, `split == "train"` only, and prints the
top-3 answers per template with their share.  Compare against the modal answer slot
that the zeroed arm actually produced (see _reanalyze_acc.py).
"""
import json
from collections import Counter
from pathlib import Path

DATA = Path(r"C:\qwen-api\RNAQwen\data\processed\rnaqa_seqtask_full.jsonl")
TEMPLATES = [
    "What are the first 4 nucleotides of this RNA?",
    "What are the first 6 nucleotides of this RNA?",
    "What are the first 8 nucleotides of this RNA?",
    "What are the last 4 nucleotides of this RNA?",
    "What are the last 6 nucleotides of this RNA?",
    "What are the last 8 nucleotides of this RNA?",
    "What are the nucleotides at positions 1 through 6 of this RNA?",
    "What is the dinucleotide at positions 1 and 2 of this RNA?",
    "What is the reverse complement of the first 4 nucleotides?",
    "What is the reverse complement of the first 6 nucleotides?",
]


def main() -> int:
    ans: dict[str, Counter] = {t: Counter() for t in TEMPLATES}
    total = 0
    with DATA.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != "train":
                continue
            t = r.get("question")
            if t in ans:
                ans[t][r["answer"]] += 1
                total += 1

    print(f"训练集里这 {len(TEMPLATES)} 个模板共 {total:,} 条\n")
    print(f"{'template':<50} {'n':>9}  top-3 答案（占比）")
    print("-" * 96)
    for t in TEMPLATES:
        c = ans[t]
        n = sum(c.values())
        top = "  ".join(f"{a} {v / n:.2%}" for a, v in c.most_common(3))
        print(f"{t[:48]:<50} {n:>9,}  {top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
