"""Download the MMLU + C-Eval question banks used by the RNAQwen benchmark suite.

Everything lands as parquet under data/external/bench so the harness never has to
touch the network again.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import pandas as pd
from datasets import load_dataset, get_dataset_config_names

OUT = r"C:\qwen-api\RNAQwen\data\external\bench"
os.makedirs(OUT, exist_ok=True)

def dump(rows, name):
    pd.DataFrame(rows).to_parquet(os.path.join(OUT, name), index=False)
    print("  saved %-22s %6d rows" % (name, len(rows)), flush=True)

# ---------------- MMLU ----------------
print("== MMLU ==", flush=True)
for split, name in [("test", "mmlu_test.parquet"),
                    ("dev", "mmlu_dev.parquet"),
                    ("validation", "mmlu_validation.parquet")]:
    ds = load_dataset("cais/mmlu", "all", split=split)
    rows = [{"subject": r["subject"], "question": r["question"],
             "choices": list(r["choices"]), "answer": int(r["answer"])} for r in ds]
    dump(rows, name)

# ---------------- C-Eval ----------------
print("== C-Eval ==", flush=True)
subs = get_dataset_config_names("ceval/ceval-exam")
for split, name in [("dev", "ceval_dev.parquet"), ("val", "ceval_val.parquet")]:
    rows = []
    for s in subs:
        ds = load_dataset("ceval/ceval-exam", s, split=split)
        for r in ds:
            d = dict(r)
            d["subject"] = s
            rows.append(d)
    dump(rows, name)

print("fields sample:", json.dumps({k: str(v)[:40] for k, v in rows[0].items()}, ensure_ascii=False), flush=True)
print("DONE", flush=True)
