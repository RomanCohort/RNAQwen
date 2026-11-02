"""机制诊断（一次性）：semantic F1 地板到底是什么造成的。

三个对照实验：
  1. **词序置换** —— 打乱每个句子内部的词序。词汇完全一样、语义全毁。
     如果 F1 还是很高，说明这条指标主要在量"词汇重叠"，不是"语义"。
  2. **纯乱码** —— 随机字母串、字母表片段。
  3. **池子大小** —— 同样的无关对，参考池从 2 句加到 40 句。
     max-cosine 的 max 是会对"候选越多"单调上升的，所以地板应当随池子变大而升高。
     这决定"地板"到底该报成单值还是"给定池子大小时的函数"。
"""

from __future__ import annotations

import random
import statistics
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: E402,F401

from rnaqwen.eval.semantic import build_embedder, split_sentences  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from importlib import import_module  # noqa: E402

floor = import_module("03b_semantic_floor")
CORPUS = floor.CORPUS
score_pair = floor.score_pair
encode_all = floor.encode_all
stats = floor.stats

SEED = 20261008
GIBBERISH = [
    "qzx vpr mnt lkjh gfds wqpo azxc vbnm",
    "zzzz qqqq xxxx jjjj vvvv kkkk wwww",
    "abc def ghi jkl mno pqr stu vwx yz",
    "aaaaaaaa bbbbbbbb cccccccc dddddddd",
]


def permute(sent: str, rng: random.Random) -> str:
    """句子内部词序置换（保持大小写与标点外壳）。"""
    core = sent.rstrip(".")
    words = core.split()
    rng.shuffle(words)
    return " ".join(words) + "."


def main() -> int:
    rng = random.Random(SEED)
    for spec in (str(ROOT / "models" / "bert-base-uncased"), str(ROOT / "models" / "pubmedbert")):
        print("=" * 78)
        print(f"embedder = {spec}")
        print("=" * 78)
        emb = build_embedder(spec)
        sent_pool = [s for ss in CORPUS.values() for s in ss]
        texts = list(sent_pool) + GIBBERISH
        cache = encode_all(emb, texts)

        # ---- 1. 词序置换 ----
        print("\n[1] 词序置换控制（词汇相同 / 语义全毁）")
        for domain in ("rna", "economics"):
            ref = " ".join(CORPUS[domain])
            f1s = []
            for _ in range(200):
                hyp = " ".join(permute(s, rng) for s in CORPUS[domain])
                f1s.append(score_pair(emb, hyp, ref, cache)["f1"])
            st = stats(f1s)
            print(f"    {domain:11s} 自身句子全部打乱后 vs 原参考: "
                  f"F1 mean={st['mean']:.4f} std={st['std']:.4f} min={st['min']:.4f} max={st['max']:.4f}")

        # ---- 2. 纯乱码 ----
        print("\n[2] 纯乱码 vs RNA 参考")
        ref = " ".join(CORPUS["rna"])
        for g in GIBBERISH:
            s = score_pair(emb, g, ref, cache)
            print(f"    {g[:38]:40s} F1={s['f1']:.4f}  (doc_cos={floor.doc_cos(emb, g, ref, cache):.4f})")

        # ---- 3. 池子大小 ----
        print("\n[3] 池子大小依赖：参考池从 2 句加到 40 句（跨域无关文本）")
        hypo = " ".join(CORPUS["rna"][:4])
        others = [s for d, ss in CORPUS.items() if d != "rna" for s in ss]
        rows = []
        for k in (2, 4, 8, 16, 32):
            f1s, ps, rs = [], [], []
            for t in range(300):
                r = random.Random(SEED + t * 7919 + k)
                ref = " ".join(r.sample(others, k))
                s = score_pair(emb, hypo, ref, cache)
                f1s.append(s["f1"]); ps.append(s["precision"]); rs.append(s["recall"])
            st = stats(f1s)
            rows.append({"pool_size": k, "f1": st, "P_mean": statistics.fmean(ps), "R_mean": statistics.fmean(rs)})
            print(f"    ref 池 {k:2d} 句  F1 mean={st['mean']:.4f} std={st['std']:.4f} "
                  f"min={st['min']:.4f}   P={statistics.fmean(ps):.4f} R={statistics.fmean(rs):.4f}")

        # 同一实验：固定 ref 池大小，但把 hyp 换成"更多句"
        print("\n[4] hyp 句数依赖（参考池固定 8 句跨域无关）")
        r = random.Random(SEED + 11)
        for nh in (1, 2, 4, 8):
            pool = r.sample(others, 8)
            ref = " ".join(pool)
            hyp = " ".join(CORPUS["rna"][:nh]) if nh <= 8 else " ".join(CORPUS["rna"])
            s = score_pair(emb, hyp, ref, cache)
            print(f"    hyp {nh} 句  F1={s['f1']:.4f}  P={s['precision']:.4f} R={s['recall']:.4f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:
        print("\n[ERROR]", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(3)
