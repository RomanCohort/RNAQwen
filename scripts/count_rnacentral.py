import gzip, sys, time
p = r"C:\qwen-api\RNAQwen\data\raw\rnacentral\rnacentral_active.fasta.gz"
n = 0; nt = 0; t0 = time.time()
with gzip.open(p, "rt", errors="replace") as fh:
    for line in fh:
        if line[0] == ">":
            n += 1
            if n % 2000000 == 0:
                print(f"[count] {n} seqs  {nt/1e9:.1f} Gnt  {time.time()-t0:.0f}s", flush=True)
        else:
            nt += len(line.strip())
print(f"[done] seqs={n} nt={nt} ({nt/1e9:.2f} Gnt) in {time.time()-t0:.0f}s", flush=True)
