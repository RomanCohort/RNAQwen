import gzip, hashlib, numpy as np, time
p = r"C:\qwen-api\RNAQwen\data\raw\rnacentral\rnacentral_active.fasta.gz"
hs = []; lens = []; cur = []
t0 = time.time()
def flush():
    global cur
    if cur:
        s = "".join(cur).upper()
        hs.append(int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little"))
        lens.append(len(s))
    cur = []
with gzip.open(p, "rt", errors="replace") as fh:
    for line in fh:
        if line[0] == ">":
            flush()
        else:
            cur.append(line.strip())
flush()
a = np.array(hs, dtype=np.uint64); L = np.array(lens, dtype=np.int32)
u = np.unique(a)
print(f"[raw]   seqs={len(a):,}  total_nt={int(L.sum()):,} ({L.sum()/1e9:.2f} Gnt)")
print(f"[dedup] unique≈{len(u):,}  ({100*len(u)/len(a):.1f}% of raw)")
for cap in (256, 512, 1024):
    m = L <= cap
    print(f"[len<={cap}] seqs={int(m.sum()):,}  nt={int(L[m].sum()):,}")
print(f"[time] {time.time()-t0:.0f}s")
