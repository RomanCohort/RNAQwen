"""Fetch C-Eval via hf_hub_download (retries + on-disk cache) in parallel.

The direct resolve/ URL redirects to the real HF CDN, which was dropping
connections; hf_hub_download retries and caches, so a flaky link only costs time
on the file that failed.
"""
import io
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
warnings.filterwarnings("ignore")

import pandas as pd
import requests
from huggingface_hub import hf_hub_download

OUT = r"C:\qwen-api\RNAQwen\data\external\bench"
REPO = "ceval/ceval-exam"

meta = requests.get("https://hf-mirror.com/api/datasets/ceval/ceval-exam", timeout=60).json()
files = [x["rfilename"] for x in meta["siblings"]]
subjects = sorted({f.split("/")[0] for f in files if "/" in f})
print("subjects:", len(subjects), flush=True)


def try_pull(name, attempts=4):
    for i in range(attempts):
        try:
            path = hf_hub_download(REPO, name, repo_type="dataset")
            return name, pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001
            if i == attempts - 1:
                print("  give up %s: %r" % (name, exc), flush=True)
                return name, None
            time.sleep(2 * (i + 1))


for split, out in [("val", "ceval_val.parquet"), ("dev", "ceval_dev.parquet")]:
    names = ["%s/%s-00000-of-00001.parquet" % (s, split) for s in subjects]
    frames, bad = [], []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for name, df in ex.map(try_pull, names):
            if df is None:
                bad.append(name)
                continue
            df = df.copy()
            df["subject"] = name.split("/")[0]
            frames.append(df)
    if bad:
        print("FAILED %d: %s" % (len(bad), bad[:6]), flush=True)
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(os.path.join(OUT, out), index=False)
    print("saved %-20s %5d rows cols=%s" % (out, len(full), list(full.columns)), flush=True)

print("answers:", sorted(full["answer"].astype(str).unique()), flush=True)
print("DONE", flush=True)
