import json, collections, os
p = r"C:\qwen-api\RNAQwen\data\processed\rnaqa_v3_pilot.jsonl"
rows=0; tpl=collections.Counter(); sp=collections.Counter(); uids=set(); ans=collections.defaultdict(list)
tok=collections.Counter(); seq={}
for line in open(p,encoding="utf-8"):
    r=json.loads(line); rows+=1
    tpl[r["template"]]+=1; sp[r["split"]]+=1; uids.add(r["uid"]); ans[r["template"]].append(len(r["answer"]))
    seq[r["uid"]]=r["sequence"]
    tok["answer"]+=max(1,len(r["answer"])//4); tok["question"]+=max(1,len(r["question"])//4)
code=sum(len(s)//4 for s in seq.values())*sum(tpl.values())/len(uids)
print("file", os.path.basename(p), f"{os.path.getsize(p)/1e6:.1f} MB")
print("rows", f"{rows:,}", "sequences", f"{len(uids):,}", f"q/seq {rows/len(uids):.2f}")
print("split", dict(sp))
print("templates", len(tpl))
for t,c in tpl.most_common():
    v=sorted(ans[t]); print(f"  {t:20s} {c:>7,}  {100*c/rows:>5.1f}%  ans p50 {v[len(v)//2]:>4}")
tot=tok["answer"]+tok["question"]+code
print(f"tokens: code {code/1e6:.1f}M ({100*code/tot:.1f}%)  question {tok['question']/1e6:.1f}M  answer {tok['answer']/1e6:.1f}M ({100*tok['answer']/tot:.1f}%)")
print(f"total {tot/1e6:.1f}M")
