import json, collections, statistics as st, os
p = r"C:\qwen-api\RNAQwen\data\processed\rnaqa_v3_l234.jsonl"
rows = 0
by_tpl = collections.Counter()
by_split = collections.Counter()
by_kind = collections.Counter()
ans_len = collections.defaultdict(list)
seq_by_uid = {}
tok = collections.Counter()
for line in open(p, encoding="utf-8"):
    r = json.loads(line); rows += 1
    t = r["template"]; by_tpl[t] += 1; by_split[r["split"]] += 1; by_kind[r["kind"]] += 1
    ans_len[t].append(len(r["answer"]))
    seq_by_uid[r["uid"]] = r["sequence"]
    tok["answer"] += max(1, len(r["answer"])//4)
    tok["question"] += max(1, len(r["question"])//4)
print("file", p, f"{os.path.getsize(p)/1e6:.1f} MB")
print("rows", f"{rows:,}", "unique sequences", f"{len(seq_by_uid):,}")
print("split", dict(by_split))
tot = sum(tok.values())
print("tokens", {k: f"{v/1e6:.1f}M" for k,v in tok.items()}, f"total {tot/1e6:.1f}M")
print()
print(f"{'template':22s} {'rows':>9s} {'share':>7s} {'answer p50':>11s} {'p95':>5s}")
for t,c in by_tpl.most_common():
    v = sorted(ans_len[t]); 
    print(f"{t:22s} {c:>9,} {100*c/rows:>6.1f}% {v[len(v)//2]:>11d} {v[int(len(v)*0.95)]:>5d}")
