import os, time, hashlib, requests
p = r"C:\qwen-api\RNAQwen\data\external\rnacentral_1024.parquet"
url = "https://huggingface.co/datasets/multimolecule/rnacentral.1024/resolve/main/data.parquet"
TARGET = 3695330389
WANT = "cf1f389a2accda43ee3d118c5e9f485be584c7511df12bca1d7b9fe5816ba7b4"
for attempt in range(200):
    have = os.path.getsize(p) if os.path.exists(p) else 0
    if have >= TARGET:
        break
    try:
        r = requests.get(url, headers={"Range": f"bytes={have}-"}, stream=True, timeout=(15, 20))
        mode = "ab" if r.status_code == 206 else "wb"
        if mode == "wb":
            have = 0
        with open(p, mode) as fh:
            for c in r.iter_content(1024 * 1024):
                fh.write(c); have += len(c)
        print("attempt", attempt, os.path.getsize(p), flush=True)
    except Exception as e:
        print("retry", type(e).__name__, os.path.getsize(p), flush=True)
    time.sleep(1)
size = os.path.getsize(p)
print("SIZE", size, "/", TARGET, flush=True)
h = hashlib.sha256()
with open(p, "rb") as fh:
    while True:
        b = fh.read(16 * 1024 * 1024)
        if not b:
            break
        h.update(b)
print("SHA256", h.hexdigest(), flush=True)
print("MATCH", h.hexdigest() == WANT, flush=True)
