import json, collections
p = r"data/processed/rnaqa_v3_l234.jsonl"
n=0; keys=None; tpl=collections.Counter(); sp=collections.Counter(); alen=collections.defaultdict(list)
with open(p, encoding="utf-8") as fh:
    for line in fh:
        r=json.loads(line); n+=1
        if keys is None: keys=list(r.keys())
        tpl[r["template"]]+=1; sp[r["split"]]+=1
        alen[r["template"]].append(len(r["answer"]))
print("rows", f"{n:,}", "keys", keys)
print("split", dict(sp))
for t,c in tpl.most_common():
    v=sorted(alen[t]); print(f"  {t:20s} {c:>8,}  answer median {v[len(v)//2]:>4} chars")
print("\nsample long answer:")
with open(p, encoding="utf-8") as fh:
    for line in fh:
        r=json.loads(line)
        if r["template"]=="l4_full_para":
            print(" Q:", r["question"]); print(" A:", r["answer"]); break
