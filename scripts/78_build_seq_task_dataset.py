"""建「已验证模板 × 新 RNA」数据集 —— 合并出错的地方在这里被纠正。

原来错在哪
----------
之前的合并是： **旧 RNA (4,481) × 未验证的新模板 (rfam 家族 / freetext 描述)**
    - freetext 的 4,481 条 RNA 和 seq8 完全重合，新增 RNA = 0
    - 它的答案是 228 字符的文献描述，来自按 RNA 身份查的注释库，不在序列里
    - 14.56% 是标注阶段 LLM 的拒答串

正确的合并是： **新 RNA (24,207) × 已验证的模板**
    - 验证过的是模板，不是 RNA。seq8 那 8 个模板可信，是因为它们的答案
      **能从序列算出来**（77_ 逐条验过 4481/4481）
    - 所以该复制的是模板，该增加的是 RNA

本脚本生成的每一条答案都是序列的确定性函数：
    first 4/6/8 = s[:4/6/8]        revcomp = comp(s[:6])[::-1]
    positions 1-6 = s[:6]          dinucleotide 1-2 = s[:2]
    last 4/6/8  = s[-4/-6/-8]
不需要探针筛、不需要注释库、不可能把答案泄漏进 prompt。

两组模板，一次因果检验
----------------------
    head  (7 个)  探针 0.446~0.810  读序列开头
    tail  (3 个)  探针 0.089~0.149  读序列结尾

同一批 RNA、同一个码信道、同一把尺子，只换了读的位置。
若 head 学得会、tail 学不会，码信道的「前向局部」就从相关变成因果。

global（计数/百分比，探针 0.000~0.037）默认不进训练集 —— 它们的失败有一个
可解释的原因（模型做不了跨码 token 的全局聚合），塞进去只会重复 freetext 的错。
需要时用 --groups head,tail,global 显式打开，且要按组分开报准确率。

用法：
    python scripts/78_build_seq_task_dataset.py
    python scripts/78_build_seq_task_dataset.py --groups head,tail --out data/processed/rnaqa_seqtask.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/processed"
FA = ROOT / "data/external/rfam/all.fa"

COMP = str.maketrans("ACGUacgu", "UGCAugca")   # N 原样保留（和 77_ 校验一致）


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


# (question, kind, 答案函数)  —— 答案函数只依赖序列
TEMPLATES: list[tuple[str, str, object]] = [
    ("What are the first 4 nucleotides of this RNA?", "head", lambda s: s[:4]),
    ("What are the first 6 nucleotides of this RNA?", "head", lambda s: s[:6]),
    ("What are the first 8 nucleotides of this RNA?", "head", lambda s: s[:8]),
    ("What is the reverse complement of the first 4 nucleotides?", "head",
     lambda s: rc(s[:4])),
    ("What is the reverse complement of the first 6 nucleotides?", "head",
     lambda s: rc(s[:6])),
    ("What are the nucleotides at positions 1 through 6 of this RNA?", "head",
     lambda s: s[:6]),
    ("What is the dinucleotide at positions 1 and 2 of this RNA?", "head",
     lambda s: s[:2]),
    ("What are the last 4 nucleotides of this RNA?", "tail", lambda s: s[-4:]),
    ("What are the last 6 nucleotides of this RNA?", "tail", lambda s: s[-6:]),
    ("What are the last 8 nucleotides of this RNA?", "tail", lambda s: s[-8:]),
    ("What is the GC content of this RNA?", "global",
     lambda s: f"{100*(s.count('G')+s.count('C'))/len(s):.1f}%"),
    ("How many G nucleotides are in this RNA?", "global",
     lambda s: str(s.count("G"))),
    ("What percentage of this RNA is A? (to 1 decimal)", "global",
     lambda s: f"{100*s.count('A')/len(s):.1f}%"),
]

MIN_LEN = 8      # tail 模板要 8 个位置


def effective(s: str, pool: int = 4) -> str:
    """模型**实际能看到**的序列。

    FSQCode.forward 里：
        n = L // pool
        x = x[:n * pool]        # 尾部截断
    位置数不是 pool 的倍数时，末尾 L%pool 个位置永远进不了码信道。
    实测 24,207 条 RNA 里 18,313 条（75.6%）不是 4 的倍数。

    所以答案必须按**截断后**的序列算。否则：
      - last-4 问的是原始末尾 4 个，其中 1~3 个模型根本没见过
      - 哪怕信道完美，缺 1 个碱基就要求模型猜中 1/4，准确率天花板被压到 0.25
      - §10 里 last-4=14.9% / last-6=11.3% / last-8=8.9% 就是被这个污染的

    这个耦合关系是硬的：如果哪天 FSQCode 改成补零而不是截断，本脚本必须同步改。
    """
    n = len(s) // pool
    return s[:n * pool]


def key(u: str) -> str:
    return u[4:] if u.startswith("seq:") else u


def read_fa(p: Path) -> dict[str, str]:
    got, cur, buf = {}, None, []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None:
                    got[key(cur)] = "".join(buf)
                cur = line[1:].strip().split()[0]
                buf = []
            else:
                buf.append(line.strip())
    if cur is not None:
        got[key(cur)] = "".join(buf)
    return {k: v.upper() for k, v in got.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="head,tail",
                    help="要生成哪些模板组：head,tail,global")
    ap.add_argument("--out", default="data/processed/rnaqa_seqtask.jsonl")
    ap.add_argument("--sources", default="seq8,rfam")
    ap.add_argument("--full", action="store_true",
                    help="用全部 145,201 条 rfam 序列（需 --split-json，见 80_split_full.py）")
    ap.add_argument("--split-json", default="data/processed/split_full.json")
    ap.add_argument("--full-source-tag", default="rfa",
                    help="全量模式下这一支的 source 标签（区别于旧的 rfam 子集）")
    args = ap.parse_args()

    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    tpl = [(q, k, f) for q, k, f in TEMPLATES if k in groups]
    print(f"[模板] 组 {sorted(groups)} → {len(tpl)} 个模板")
    for q, k, _ in tpl:
        print(f"    {k:<7} {q}")

    # ---------- 收集 RNA：uid -> (sequence, split, source) ----------
    rnas: dict[str, tuple[str, str, str]] = {}
    if "seq8" in args.sources:
        for line in open(D / "rnaqa_seq8.jsonl", encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            sp = r.get("split", "train")
            if sp == "val":                     # 保留 val，和原口径一致
                pass
            elif sp not in ("train", "test"):
                sp = "train"
            rnas.setdefault(r["uid"], (r["sequence"].upper(), sp, "seq8"))
        print(f"[seq8]  {sum(1 for v in rnas.values() if v[2]=='seq8'):,} 条 RNA")

    if "rfam" in args.sources:
        fa = read_fa(FA)
        n = 0
        if args.full:
            # 升级 2 之后：用全部 145,201 条，划分来自 split_full.json（80_ 产出）
            if not args.split_json:
                print("[rfam] --full 需要 --split-json，先跑 80_split_full.py")
                return 2
            split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
            src_tag = args.full_source_tag
            for u, s in fa.items():
                sp = split.get(u)
                if sp is None:
                    continue
                rnas[u] = (s, sp, src_tag)
                n += 1
            print(f"[{src_tag}] {n:,} 条 RNA（全量，划分来自 {Path(args.split_json).name}）")
        else:
            for line in open(D / "rnaqa_rfam.jsonl", encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                u = key(r["uid"])
                s = fa.get(u)
                if not s:
                    continue
                if u in rnas:                   # 理论上不会（两个集合不相交）
                    continue
                rnas[u] = (s, r.get("split", "train"), "rfam")
                n += 1
            print(f"[rfam]  {n:,} 条 RNA")

    # 过滤太短的（按**截断后**的长度，那才是模型能看到的）
    short = [u for u, (s, _, _) in rnas.items() if len(effective(s)) < MIN_LEN]
    for u in short:
        del rnas[u]
    dropped = Counter(len(s) % 4 for s, _, _ in rnas.values())
    print(f"[trunc] 位置数 %4 的分布 " +
          "  ".join(f"%4={k}:{v:,}" for k, v in sorted(dropped.items())))
    print(f"[trunc] 被丢掉 1~3 个位置的有 "
          f"{sum(v for k, v in dropped.items() if k):,} / {len(rnas):,} 条 "
          f"（答案已按截断后序列算）")
    if short:
        print(f"[drop]  截断后长度 < {MIN_LEN} 的 {len(short)} 条")

    # ---------- 生成 ----------
    out = ROOT / args.out
    n_write = 0
    c_src, c_sp, c_kind = Counter(), Counter(), Counter()
    seen_uid_split: dict[str, str] = {}
    conflict = 0
    with open(out, "w", encoding="utf-8") as fo:
        for u, (s, sp, src) in rnas.items():
            if u in seen_uid_split and seen_uid_split[u] != sp:
                conflict += 1
            seen_uid_split[u] = sp
            e = effective(s)                 # 按模型实际能看到的序列算答案
            for q, k, f in tpl:
                a = f(e)
                fo.write(json.dumps({
                    "uid": u, "question": q, "answer": a, "kind": k,
                    "source": src, "template": q, "split": sp,
                }, ensure_ascii=False) + "\n")
                n_write += 1
                c_src[src] += 1
                c_sp[sp] += 1
                c_kind[k] += 1

    print()
    print("=" * 74)
    print(f"  {out.name}   **{n_write:,} 条**   RNA {len(rnas):,} 条")
    print(f"  来源 {dict(c_src)}")
    print(f"  划分 {dict(c_sp)}")
    print(f"  模板组 {dict(c_kind)}")
    if conflict:
        print(f"  ** 有 {conflict} 条 RNA 出现在两个划分里 —— 泄漏")
    # 逐组抽样看答案
    print()
    print("  抽样（每模板组 1 例）:")
    shown = set()
    for u, (s, sp, src) in list(rnas.items()):
        e = effective(s)
        for q, k, f in tpl:
            if k in shown:
                continue
            shown.add(k)
            print(f"    [{k}] {q[:52]}")
            print(f"          raw len={len(s)}  eff len={len(e)}  "
                  f"丢了 {len(s)-len(e)}")
            print(f"          eff[:12]={e[:12]!r}  eff[-12:]={e[-12:]!r}")
            print(f"          answer={f(e)!r}")
    print("=" * 74)
    return 1 if conflict else 0


if __name__ == "__main__":
    raise SystemExit(main())
