"""rna_literature_1k 的 definition 够不够当 annotation。

现有 3488 条的 annotation 是 RNAcentral 的**策展描述**（金标准）。
如果 1k 这批的 definition 也是同类东西，就能直接补上；否则只能重新生成。
"""
import json
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")


def load(p):
    with open(ROOT / p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


print("=" * 92)
print("definition / annotation 覆盖率与样例")
print("=" * 92)
for name in ["data/processed/rnaqa_rnacentral_all.jsonl",
             "data/raw/rna_literature_1k.jsonl"]:
    recs = load(name)
    print("\n--- %s (%d 条) ---" % (name.split("/")[-1], len(recs)))
    for field in ["annotation", "definition", "literature", "rna_type", "organism"]:
        have = sum(1 for r in recs if (r.get(field) or "").strip() if isinstance(r.get(field), str))
        print("   %-12s 有 %4d / %d" % (field, have, len(recs)))
    for field in ["annotation", "definition"]:
        ex = next((r.get(field) for r in recs if (r.get(field) or "").strip()
                   if isinstance(r.get(field), str)), None)
        if ex:
            print("   %s 样例: %s" % (field, ex[:200].replace("\n", " ")))
        else:
            print("   %s 样例: （无）" % field)

    # definition 长度分布
    defs = [len(r.get("definition") or "") for r in recs if isinstance(r.get("definition"), str)]
    if defs:
        defs.sort()
        print("   definition 长度: min %d  median %d  max %d"
              % (defs[0], defs[len(defs) // 2], defs[-1]))

print()
print("=" * 92)
print("结论")
print("=" * 92)
print("""
 如果 1k 的 definition 长度和 3488 的 annotation 量级相当（几十~几百字符），
 那它就是同一类东西，可以当 Stage 1 的目标；否则要另想办法。
""")
