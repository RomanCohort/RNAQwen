"""系统性质量度量：QA 到底在问"这条 RNA"还是"那篇论文"。

背景
----
换掉 NCBI 链路的根本原因，是那条路生成的问题**问的不是那条 RNA**
（`What is the purpose of the PhyloFish database?`）。所以"换源之后这个问题修好了没有"
必须**量化**，不能靠看两条样例拍脑袋。

判据
----
`ent_id`（如 `MIR423` / `SNORD1B` / `LINC00471`）是 RNA 的名字。
一条 QA 只要在**问题或答案**里提到这个名字，就说明它至少锚定在正确的分子上。
按 variant（ais=文献 / dc=RNAcentral 官方综述）分别统计，差距就是这两个变体的性质差异。

另外统计闭式问答的形式（yes/no vs 四选一）—— `12_generate_qa.py` 的 kind 字段
在模型没返回 `kind` 时会回落成 "closed"，所以这里用文本形态判断，不信任该字段。
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.data.records import read_jsonl  # noqa: E402

_MC_RE = re.compile(r"^\s*[A-D][\).\:]\s*\S", re.M)


def name_forms(ent_id: str) -> list[str]:
    """一个基因名可能的写法：原样 / 去横线 / 小写变体交给大小写无关匹配。"""
    e = (ent_id or "").strip()
    if not e:
        return []
    forms = {e, e.replace("-", ""), e.replace("-", " ")}
    # MIR423 -> miR-423 这类写法也常见；做一个宽松的数字+前后缀形式
    m = re.match(r"^([A-Za-z]+)[-_]?(\d+)$", e)
    if m:
        forms.add(f"{m.group(1)}-{m.group(2)}")
        forms.add(f"{m.group(1)}{m.group(2)}")
    return [f.lower() for f in forms if f]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "qa_quality_metrics.json")
    ap.add_argument("--examples", type=int, default=5)
    args = ap.parse_args()

    recs = list(read_jsonl(args.inp))
    per = collections.defaultdict(lambda: collections.Counter())
    overall = collections.Counter()
    miss_examples: list[dict] = []

    for r in recs:
        ent = r.provenance.get("ent_id", "") if isinstance(r.provenance, dict) else ""
        forms = name_forms(ent)
        for p in r.qa:
            var = p.get("variant", "?")
            kind = p.get("kind", "?")
            q, a = p.get("question", ""), p.get("answer", "")
            blob = (q + " " + a).lower()
            hit = any(f in blob for f in forms)
            both = bool(forms)
            overall["qa"] += 1
            overall["has_ent_id"] += int(both)
            overall["mentions_rna"] += int(hit)
            per[var]["qa"] += 1
            per[var]["mentions_rna"] += int(hit)
            per[var + "/" + kind]["qa"] += 1
            per[var + "/" + kind]["mentions_rna"] += int(hit)
            if kind in ("closed", "yes_no", "multiple_choice"):
                is_mc = bool(_MC_RE.search(a)) or "A." in a[:8] or "B." in a[:8]
                per[var]["closed_mc"] += int(is_mc)
                per[var]["closed_yn"] += int(not is_mc)
            if not hit and len(miss_examples) < 400:
                miss_examples.append({"uid": r.uid, "ent_id": ent, "variant": var,
                                      "kind": kind, "q": q[:170], "a": a[:170]})

    n = max(overall["qa"], 1)
    report = {
        "records": len(recs), "qa": overall["qa"],
        "mentions_rna_pct": round(100 * overall["mentions_rna"] / n, 1),
        "has_ent_id_pct": round(100 * overall["has_ent_id"] / n, 1),
        "by_variant": {k: dict(v) for k, v in sorted(per.items())},
        "by_variant_pct": {
            k: round(100 * v["mentions_rna"] / max(v["qa"], 1), 1)
            for k, v in sorted(per.items()) if "mentions_rna" in v
        },
        "miss_examples": miss_examples[: args.examples * 4],
    }
    print(json.dumps({k: v for k, v in report.items() if k != "miss_examples"},
                     indent=2, ensure_ascii=False))
    print("\n--- 没提到 RNA 名字的样例（抽查用）---")
    for e in miss_examples[: args.examples]:
        print(f"  [{e['variant']}/{e['kind']}] ent={e['ent_id']}")
        print(f"    Q: {e['q']}")
        print(f"    A: {e['a']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[metrics] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
