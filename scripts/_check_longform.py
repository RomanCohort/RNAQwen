import json, collections, statistics as st
# 1) n6_merged: annotation / definition / literature 可用性
n=0; has_def=has_ann=has_lit=0; dl=[]; al=[]; ll=[]
with open(r"data/processed/rnaqa_n6_merged.jsonl", encoding="utf-8") as fh:
    for line in fh:
        r=json.loads(line); n+=1
        d=(r.get("definition") or "").strip()
        a=(r.get("annotation") or "").strip()
        l=(r.get("literature") or "").strip()
        if d: has_def+=1; dl.append(len(d))
        if a: has_ann+=1; al.append(len(a))
        if l: has_lit+=1; ll.append(len(l))
print(f"[n6_merged] n={n}")
print(f"  definition {has_def} ({100*has_def/n:.0f}%)  median {int(st.median(dl)) if dl else 0} chars")
print(f"  annotation {has_ann} ({100*has_ann/n:.0f}%)  median {int(st.median(al)) if al else 0} chars")
print(f"  literature {has_lit} ({100*has_lit/n:.0f}%)  median {int(st.median(ll)) if ll else 0} chars")

# 2) cont_clean: 按 source 看答案长度
src=collections.defaultdict(list)
with open(r"data/processed/rnaqa_cont_clean.jsonl", encoding="utf-8") as fh:
    for line in fh:
        r=json.loads(line)
        src[r.get("source","?")].append(len(r.get("answer") or ""))
print("\n[cont_clean] source -> n / median answer chars / p95")
for k,v in sorted(src.items(), key=lambda x:-len(x[1])):
    v.sort()
    print(f"  {k:12s} {len(v):>8,}  median {v[len(v)//2]:>5}  p95 {v[int(len(v)*0.95)]:>5}")
