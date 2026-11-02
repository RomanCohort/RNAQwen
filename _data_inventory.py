"""数据盘点：每条源各有多少 RNA / QA，哪些还没用上。"""
import json
from pathlib import Path

ROOT = Path(r"C:\qwen-api\RNAQwen")


def peek(path: Path):
    n, keys = 0, None
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n += 1
                if keys is None:
                    try:
                        d = json.loads(line)
                        keys = sorted(d.keys())
                    except Exception:
                        keys = ["<JSON 解析失败>"]
    except Exception as e:
        return None, [f"读取失败 {e}"]
    return n, keys or []


print("=" * 96)
print("raw / processed 文件盘点")
print("=" * 96)
for d in ["data/raw", "data/processed"]:
    print(f"\n--- {d} ---")
    for p in sorted((ROOT / d).glob("*.jsonl")):
        n, keys = peek(p)
        print("  %-46s %8s 条  %8.1f MB" % (p.name, n, p.stat().st_size / 1e6))
        if keys:
            print("      字段: %s" % (", ".join(keys[:12])))

print()
print("=" * 96)
print("关键文件的 RNA/QA 统计")
print("=" * 96)
for name in ["data/processed/rnaqa_rnacentral_all.jsonl",
             "data/processed/rnaqa_n6.jsonl",
             "data/processed/rnaqa_rnacentral.jsonl",
             "data/processed/rnaqa_rnacentral_new.jsonl"]:
    p = ROOT / name
    if not p.exists():
        continue
    n_rna = n_qa = 0
    variants, kinds = {}, {}
    urs = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n_rna += 1
            urs.add(d.get("urs_id") or d.get("id") or d.get("sequence", "")[:40])
            qa = d.get("qa") or []
            n_qa += len(qa)
            for q in qa:
                variants[q.get("variant", "?")] = variants.get(q.get("variant", "?"), 0) + 1
                kinds[q.get("kind", "?")] = kinds.get(q.get("kind", "?"), 0) + 1
    print("\n  %s" % name)
    print("    RNA %-6d  QA %-7d  唯一 URS %-6d  QA/RNA %.2f" % (n_rna, n_qa, len(urs), n_qa / max(n_rna, 1)))
    print("    variant %s" % variants)
    print("    kind    %s" % kinds)
