"""Validate the C-Eval Chinese template before the full sweep gets there.

Pulls whatever ceval/ceval-exam subjects are already in the HF cache, builds the
prompt with the exact functions the real harness uses, and scores a handful of
questions against the 8098 GPU instance.
"""
import os, sys, warnings, argparse, collections
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, r"C:\qwen-api\RNAQwen\scripts\_scratch")
import _mc_bench as H
from datasets import load_dataset, get_dataset_config_names

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8098)
ap.add_argument("--subjects", default="computer_network,high_school_physics,college_programming")
ap.add_argument("--limit", type=int, default=8)
ap.add_argument("--nshot", type=int, default=5)
args = ap.parse_args()

cn = H.load_cn_names()
class A:  # stand-in for the argparse namespace run_one expects
    port = args.port
    topk = 40
    ngen = 4
    timeout = 600.0
    dump_top = True
    dump_n = 2

ok = n = 0
for s in [x for x in args.subjects.split(",") if x]:
    try:
        dev = load_dataset("ceval/ceval-exam", s, split="dev")
        val = load_dataset("ceval/ceval-exam", s, split="val")
    except Exception as e:
        print("skip", s, repr(e)); continue
    shots = [H.ceval_block(dev[i], str(dev[i]["answer"]).strip().upper())
             for i in range(min(args.nshot, len(dev)))]
    head = H.CEVAL_HEAD.format(cn.get(s, s)) + "\n"
    for i in range(min(args.limit, len(val))):
        row = dict(val[i])
        gold = str(row["answer"]).strip().upper()
        task = {"bench": "ceval", "subject": s, "idx": i, "gold": gold,
                "strict_prefix": "", "prompt": head + "\n\n".join(shots + [H.ceval_block(row)])}
        r = H.run_one(task, A)
        n += 1
        ok += r["ok"]
        print("%-24s %2d gold=%s gen=%s cloze=%s strict=%s  n_letters=%d  top=%s  gens=%s"
              % (s, i, gold, r["gen"], r["cloze"], r["strict"], r["n_letters"],
                 r.get("top"), r.get("gen_tokens")), flush=True)
print("smoke acc: %d/%d = %.3f" % (ok, n, ok / max(1, n)))
