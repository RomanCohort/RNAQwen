import os, time, requests
p = r"C:\qwen-api\RNAQwen\data\external\rnacentral_1024.parquet"
url = "https://huggingface.co/datasets/multimolecule/rnacentral.1024/resolve/main/data.parquet"
target = 3695300000
for attempt in range(60):
    have = os.path.getsize(p) if os.path.exists(p) else 0
    if have >= 3690000000:
        print("DONE", have, flush=True); break
    try:
        r = requests.get(url, headers={"Range": f"bytes={have}-"}, stream=True, timeout=(15, 15))
        mode = "ab" if r.status_code == 206 else "wb"
        if mode == "wb":
            have = 0
        with open(p, mode) as fh:
            for c in r.iter_content(512 * 1024):
                fh.write(c); have += len(c)
        print("attempt", attempt, os.path.getsize(p), flush=True)
    except Exception as e:
        print("retry", type(e).__name__, os.path.getsize(p), flush=True)
    time.sleep(1)
print("FINAL", os.path.getsize(p), flush=True)
