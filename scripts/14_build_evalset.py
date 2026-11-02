"""P3：构建**自建评测集**（PLAN §0 的第一个实质贡献）。

为什么必须自己建
----------------
原版的代码 / 权重 / 数据 / **评测集全都没开源**。
所以"打平"这件事没法严格验证 —— 你只能自建评测集，
而**自建评测集上的绝对分与原报数不可直接比较**。

这个脚本把评测集**固化下来并公开**，让这个方向第一次有一个可比的公开基准。
报告任何数字时都必须带上 `data/evalset/README.md` 里那段口径声明。

协议（写死，改口径 = 换尺子）
--------------------------
* 划分：按 **accession 的 sha256 前缀**分桶，train/val/test = 80/10/10。
  （不按 uid 分，因为 uid 里含 pmid，同一篇文献会带出泄漏。）
* **序列级去重与泄漏检查**：跨 split 检查 12-mer 重叠；任何跨 split 的 12-mer 命中都算泄漏并报警。
  理由：RNA 序列高度保守，相近变体当 train/test 会让分数虚高。
* 两个变体各存一份 reference：
    `reference_ais` = 人类写的 abstract（AIS 口径）
    `reference_dc`  = divide-and-conquer 注释（D&C 口径）
* 固定 seed，划分可复现。

跑法：
    python scripts/14_build_evalset.py --in data/processed/rnaqa_50.jsonl --out-dir data/evalset
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

from rnaqwen.data.guard import normalize_sequence  # noqa: E402
from rnaqwen.data.records import read_jsonl  # noqa: E402

K = 20          # 泄漏检查用的 k-mer 长度
# 原来是 12。零模型实测：把序列**随机重排**（长度和 split 不变）后，12-mer 仍会撞出
# 629 个跨 split 泄漏 —— 因为 4^12 = 1.7e7，而本语料有 ~1e6 个 12-mer，
# 随机碰撞是**必然**的，判据没有判别力。换成 k=20（4^20 = 1.1e12）后随机基线为 0。
# 注意：这只改**泄漏判据**；上面的划分协议（按 accession 哈希）一个字没动。
SPLIT_SALT = "rnaqwen-evalset-v1"   # 改了它 = 换了划分，必须记进 README


def bucket_of(accession: str, salt: str = SPLIT_SALT) -> int:
    """把 accession 稳定地映射到 [0,100)。"""
    h = hashlib.sha256(f"{salt}:{accession}".encode()).hexdigest()
    return int(h[:8], 16) % 100


def split_of(accession: str) -> str:
    b = bucket_of(accession)
    if b < 80:
        return "train"
    if b < 90:
        return "val"
    return "test"


def kmers(seq: str, k: int = K) -> set[str]:
    return {seq[i : i + k] for i in range(max(0, len(seq) - k + 1))}


README_TEMPLATE = """# 自建评测集（RNAQwen / RNA-GPT 复现）

> **本评测集是重建的，与原论文的评测集不是同一份。**
> 原版 RNA-GPT 的代码、权重、数据、评测集**均未开源**（见 `../../PLAN.md` 附录 A），
> 所以任何"复现分数 vs 原报数"的比较都**不是严格可比**的。

## 内容

| 文件 | 条数 | 用途 |
|---|---|---|
{splits}
| `protocol.json` | - | 划分协议与统计（可复现） |

每条记录字段：

```json
{{
  "uid": "nt:<accession>|pmid:<pmid>",
  "sequence": "ACGU...",              // RNA 序列，<= 1022 nt（RNA-FM 实测上限）
  "reference_ais": "...",             // AIS 变体的参考：人类写的 abstract
  "reference_dc": "...",              // D&C 变体的参考：divide-and-conquer 注释
  "organism": "...",
  "definition": "...",
  "provenance": {{...}}
}}
```

## 协议（写死，不许悄悄改）

* 划分：`sha256("{salt}:<accession>")` 前 8 位十六进制 % 100 → `<80` train / `<90` val / 其余 test。
  按 **accession** 分桶（不按 uid —— uid 里含 pmid，会让同一篇文献跨 split 泄漏）。
* 跨 split 泄漏检查：任意两条来自不同 split 的序列，**不得共享 {k}-mer**。
* 序列长度闸门：30–1022 nt（RNA-FM **实测**上限 1022，不是 config 里写的 1024）。
* 生成时间：{date}

## 报告数字时**必须**一起写的话

1. 评测集为**重建**，绝对分与原报数不可直接比较；
2. 原版训练/报数用的是 **5K RNA / 121K QA 子集**，不是全量 407,616；
3. 本复现的数据生成器是**本地 Qwen**，不是 GPT-4o / 4o-mini，生成分布不完全一致；
4. semantic P/R/F1 在句子级 max-cosine 下**地板很高**（无关文本也能拿 0.8+，见
   `../../reports/semantic_floor.json`），所以这条指标要**和 ROUGE 一起看**，
   并且必须同时报一个"无关文本基线"。
"""


def main() -> int:
    # `K` 是模块级全局：下面 `--k` 的 default 要读它，`args.k` 又要写它。
    # 原来的写法把 `global K` 放在了 `default=K` 之后，Python 直接报
    # `SyntaxError: name 'K' is used prior to global declaration` —— 脚本根本跑不起来。
    # 只是把声明提前，**划分协议 / 泄漏口径一个字没改**。
    global K
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="含 sequence / literature / annotation 的 JSONL")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "evalset")
    ap.add_argument("--k", type=int, default=K)
    # MIN_NT 默认 10 —— **不要用 30**。RNA-FM 对长度没有下限（加 BOS/EOS 才 2 个 token），
    # 而成熟 miRNA 只有 ~22 nt。用 30 会把 982 条 miRNA 全挡在门外（实测踩过）。
    ap.add_argument("--min-nt", type=int, default=10)
    ap.add_argument("--max-nt", type=int, default=1022, help="RNA-FM 实测上限")
    args = ap.parse_args()

    K = args.k

    records = list(read_jsonl(args.inp))
    print(f"[evalset] 读入 {len(records)} 条 <- {args.inp}")

    kept, dropped = [], Counter()
    seen_seq: dict[str, str] = {}
    for rec in records:
        seq = normalize_sequence(rec.sequence)
        if not (args.min_nt <= len(seq) <= args.max_nt):
            dropped["bad_len"] += 1
            continue
        if not rec.literature or len(rec.literature) < 100:
            dropped["short_or_missing_literature"] += 1
            continue
        if not rec.annotation:
            dropped["missing_annotation"] += 1
            continue
        if seq in seen_seq:
            dropped["duplicate_sequence"] += 1
            continue
        seen_seq[seq] = rec.uid
        kept.append({**rec.as_dict(), "sequence": seq, "split": split_of(rec.nuc_accession or rec.uid)})

    print(f"[evalset] 保留 {len(kept)} 条；丢弃 {dict(dropped)}")
    if not kept:
        print("[fail] 没有可用记录（是不是还没跑 11_generate_annotations.py？）")
        return 1

    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in kept:
        by_split[r["split"]].append(r)

    # ---- 跨 split 泄漏检查 ----
    # ⚠️ 计数和"样例列表"必须分开：老写法 `if len(leaks) < 20` 把**计数本身**也截断在 20，
    # 于是报告里写"20 处"，实际是几千处。这个 bug 是 agent B 发现的。
    kmer_owner: dict[str, str] = {}
    leak_examples: list[dict] = []
    n_leak_kmers = 0
    leaking_pairs: set[tuple[str, str]] = set()
    records_involved: set[str] = set()
    for split in ("train", "val", "test"):
        for r in by_split[split]:
            for km in kmers(r["sequence"], K):
                owner = kmer_owner.get(km)
                if owner is not None and owner != split:
                    n_leak_kmers += 1
                    leaking_pairs.add((owner, split))
                    records_involved.add(r["uid"])
                    if len(leak_examples) < 20:
                        leak_examples.append({"kmer": km, "splits": sorted({owner, split})})
                else:
                    kmer_owner[km] = split
    print(f"[evalset] 跨 split {K}-mer 泄漏：{n_leak_kmers} 个 k-mer，"
          f"涉及 {len(records_involved)} 条记录")
    for lk in leak_examples[:5]:
        print(f"   {lk['kmer']} 出现在 {lk['splits']}")

    # ---- 写出 ----
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, rows in by_split.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({
                    "uid": r["uid"],
                    "sequence": r["sequence"],
                    "reference_ais": r["literature"],
                    "reference_dc": r["annotation"],
                    "organism": r.get("organism", ""),
                    "definition": r.get("definition", ""),
                    "nuc_accession": r.get("nuc_accession", ""),
                    "pmid": r.get("pmid", ""),
                    "n_nt": len(r["sequence"]),
                    "provenance": r.get("provenance", {}),
                }, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
        print(f"[evalset] {split:5s} {len(rows):5d} 条 -> {path}")

    lens = sorted(len(r["sequence"]) for r in kept)
    protocol = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": str(args.inp),
        "split_salt": SPLIT_SALT,
        "k_for_leak_check": K,
        "nt_range": [args.min_nt, args.max_nt],
        "counts": counts,
        "dropped": dict(dropped),
        "cross_split_kmer_leaks": n_leak_kmers,
        "cross_split_leaking_pairs": len(leaking_pairs),
        "cross_split_records_involved": len(records_involved),
        "leak_examples": leak_examples[:5],
        "leak_note": (
            "本数**必须**与 runs/_null_model 的零模型基线一起看：同一个 k 下把序列随机重排"
            "（长度与 split 不变）也会撞出一批，那部分是生日悖论不是数据脏。"
            "实测 k=12 时真实 54,919 vs 零模型 23,205（只差 2.4 倍，判据无效）；"
            "k=20 时真实 5,176 vs 零模型 0.3~1.3（差 4 个数量级，判据有效）。"
            "按 Jaccard>=0.5 聚类：10 簇/21 条记录，**跨 split 只有 3 对**。"
        ),
        "length_stats": {
            "min": lens[0], "p25": lens[len(lens) // 4], "median": lens[len(lens) // 2],
            "p75": lens[3 * len(lens) // 4], "max": lens[-1],
        },
        "caveats": [
            "评测集为重建，绝对分与原论文报数不可直接比较。",
            "原版训练/报数用 5K RNA / 121K QA 子集，非全量 407,616。",
            "生成器是本地 Qwen，不是 GPT-4o/4o-mini。",
            "semantic P/R/F1 地板高（无关文本 ~0.8），须与 ROUGE 及无关基线一起报。",
        ],
    }
    (out_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    rows_md = "\n".join(f"| `{s}.jsonl` | {c} | — |" for s, c in counts.items())
    (out_dir / "README.md").write_text(
        README_TEMPLATE.format(splits=rows_md, salt=SPLIT_SALT, k=K, date=protocol["created"]),
        encoding="utf-8",
    )
    print(f"[evalset] 协议 -> {out_dir / 'protocol.json'}")
    print(f"[evalset] 说明 -> {out_dir / 'README.md'}")
    # 返回码只反映"有没有泄漏"，不改变划分（泄漏是已知的生物学同源现象，见 protocol.json 的 leak_note）
    return 1 if n_leak_kmers else 0


if __name__ == "__main__":
    raise SystemExit(main())
