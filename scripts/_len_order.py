import json, statistics as st
p = r"C:\qwen-api\RNAQwen\data\processed\rnacentral_2m.jsonl"
N = 2187209
step = 2048
marks = [0, 100_000, 300_000, 600_000, 1_000_000, 1_500_000, 2_000_000]
buckets = {}
cur = 0
lens = []
with open(p, encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        r = json.loads(line)
        L = len(r["sequence"])
        lens.append(L)
        if i in marks:
            buckets[i] = L
        if i >= marks[-1]:
            break
print("length at file offsets:", buckets)
# average length in each 100k block
import itertools
blocks = [lens[i:i+100_000] for i in range(0, len(lens), 100_000)]
for bi, b in enumerate(blocks[:8]):
    print(f"  block {bi} (rows {bi*100000}-{bi*100000+len(b)}) mean_len={st.mean(b):.0f} max={max(b)}")
print("overall mean", round(st.mean(lens),1))
