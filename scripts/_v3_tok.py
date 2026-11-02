import json, collections
p = r"C:\qwen-api\RNAQwen\data\processed\rnaqa_v3_l234.jsonl"
q_per_seq = collections.Counter()
seq = {}
for line in open(p, encoding="utf-8"):
    r = json.loads(line)
    q_per_seq[r["uid"]] += 1
    seq[r["uid"]] = r["sequence"]
code_tok = sum((len(seq[u])//4) * n for u, n in q_per_seq.items())
ans = q = 0
for line in open(p, encoding="utf-8"):
    r = json.loads(line)
    ans += max(1, len(r["answer"])//4)
    q += max(1, len(r["question"])//4)
tot = ans + q + code_tok
print(f"code tokens   {code_tok/1e6:6.1f} M  ({100*code_tok/tot:4.1f}%)")
print(f"question      {q/1e6:6.1f} M  ({100*q/tot:4.1f}%)")
print(f"answer        {ans/1e6:6.1f} M  ({100*ans/tot:4.1f}%)")
print(f"total         {tot/1e6:6.1f} M")
print(f"questions/seq mean {sum(q_per_seq.values())/len(q_per_seq):.2f}")
lens = sorted(len(s) for s in seq.values())
print(f"seq len p10 {lens[len(lens)//10]} p50 {lens[len(lens)//2]} p90 {lens[len(lens)*9//10]} max {lens[-1]}")
