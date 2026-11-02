"""Does the shipped GGUF answer, or does it keep talking?

The exact-match number in 2.4 is depressed by a decoder that does not stop when the
answer turn ends: a typical A-arm miss is an answer followed by another turn, so the
model knew the answer and scored zero for it.  Offline re-analysis only re-read those
stored strings; this does the actual fix, on the artifact we ship.

Every prompt goes through the *same* Q4_K_M server twice:

  nostop  the way the eval has always called it - generate until max_new_tokens
  stop    the same call with the turn boundary declared as a stop string

So the two exact-match numbers differ by nothing but the decoder rule.

Pool: test split of the seqtask corpus, restricted to the uids whose sequence is in
rnaqa_n6_merged.jsonl (the frontend recomputes the codes from the sequence), evenly
spread per template.  The question-only prior floor is recomputed on the very same
picks, because a floor quoted from another sample is not a floor.

    python scripts/_scratch/_gguf_stop_eval.py --per-template 12 --n-predict 24
"""
import argparse
import importlib.util
import json
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GG = ROOT / "gguf"
TASK = ROOT / "data" / "processed" / "rnaqa_seqtask_full.jsonl"
SEQF = ROOT / "data" / "processed" / "rnaqa_n6_merged.jsonl"
STOP = ["\n", "<|im_end|>", "<|im_start|>"]


def norm(s: str) -> str:
    return "".join(c for c in s.strip().upper() if c.isalnum() or c in ".%").rstrip(".")


def char_f1(pred: str, gold: str) -> float:
    a, b = Counter(norm(pred)), Counter(norm(gold))
    if not a or not b:
        return 0.0
    ov = sum((a & b).values())
    if ov == 0:
        return 0.0
    p, r = ov / sum(a.values()), ov / sum(b.values())
    return 2 * p * r / (p + r)


def spread(lst: list, n: int) -> list:
    if n <= 0 or len(lst) <= n:
        return lst
    st = len(lst) / n
    return [lst[min(int(i * st), len(lst) - 1)] for i in range(n)]


def clean(txt: str) -> str:
    """The rule the eval has always used: the answer turn opens with "A:".

    The model does emit that prefix (the GGUF returns " A: CUGG" for gold "CUGG"),
    and 76_acc_by_source strips everything up to the last "A:" before scoring, so
    this has to strip it too or the two paths are not on the same ruler.
    """
    t = txt.strip()
    if "A:" in t:
        t = t.split("A:")[-1].strip()
    return t


def post(url: str, body: dict, timeout: int = 3600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8099")
    ap.add_argument("--per-template", type=int, default=12)
    ap.add_argument("--n-predict", type=int, default=24)
    ap.add_argument("--out", default="reports/gguf_stop_eval.json")
    ap.add_argument("--limit-templates", type=int, default=0)
    ap.add_argument("--skip-nostop", action="store_true",
                    help="only make the stopping call (the server already stops on "
                         "its own, so the second call doubles the bill for nothing)")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("rp", GG / "rna_prompt.py")
    rp = importlib.util.module_from_spec(spec)
    sys.modules["rp"] = rp
    spec.loader.exec_module(rp)

    seq_of = {}
    with SEQF.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                seq_of[r["uid"]] = r["sequence"]
    print(f"[data] uid with sequence {len(seq_of):,}", flush=True)

    byq: dict[str, list] = {}
    train: dict[str, Counter] = {}
    with TASK.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") == "test":
                if r["uid"] in seq_of:
                    byq.setdefault(r["question"], []).append(r)
            elif r.get("split") == "train":
                train.setdefault(r["question"], Counter())[norm(r["answer"])] += 1
    mode = {q: c.most_common(1)[0][0] for q, c in train.items()}

    picks = []
    for q in sorted(byq):
        picks += spread(byq[q], args.per_template)
    if args.limit_templates:
        keep = sorted(byq)[: args.limit_templates]
        picks = [r for r in picks if r["question"] in keep]
    floor = sum(1 for r in picks if norm(r["answer"]) == mode[r["question"]]) / len(picks)
    print(f"[picks] {len(picks)} items / {len({r['question'] for r in picks})} templates"
          f"   prior floor (gold == train mode) {floor:.2%}", flush=True)

    print("first call loads ERNIE ...", flush=True)
    recs = []
    for i, r in enumerate(picks):
        seq = seq_of[r["uid"]].strip().upper()
        ids, ncode = rp.build(seq, r["question"])
        base = {"prompt": ids, "n_predict": args.n_predict, "temperature": 0.0,
                "cache_prompt": False}
        raw_b = post(args.url + "/completion", dict(base, stop=STOP)).get("content") or ""
        raw_a = "" if args.skip_nostop else (
            post(args.url + "/completion", base).get("content") or "")
        if args.skip_nostop:
            raw_a = raw_b
        a, b = clean(raw_a), clean(raw_b)
        gold = norm(r["answer"])
        rec = dict(template=r["question"], uid=r["uid"], gold=gold, codes=ncode,
                   nt=len(seq), nostop=a, stop=b, raw_nostop=raw_a, raw_stop=raw_b,
                   prior=gold == mode[r["question"]],
                   exact_nostop=norm(a) == gold, exact_stop=norm(b) == gold,
                   prefix_nostop=bool(gold) and norm(a).startswith(gold),
                   prefix_stop=bool(gold) and norm(b).startswith(gold),
                   f1_stop=char_f1(b, gold))
        recs.append(rec)
        verdict = "OK" if rec["exact_stop"] else (
            "prefix" if rec["prefix_stop"] else "MISS")
        print(f"[{i + 1}/{len(picks)}] {r['question'][:34]:<36} gold {gold:<10} "
              f"nostop {norm(a)[:12]:<12} stop {norm(b)[:12]:<12} {verdict}", flush=True)

    def rate(key: str, sub: list) -> float:
        return sum(1 for x in sub if x[key]) / max(len(sub), 1)

    print(f"\n{'arm':>7} {'n':>4} {'exact':>7} {'prefix':>7} {'charF1':>7} {'over':>5}")
    for arm in ("nostop", "stop"):
        over = sum(1 for x in recs if len(norm(x[arm])) > len(x["gold"]))
        f1 = sum(char_f1(x[arm], x["gold"]) for x in recs) / len(recs)
        print(f"{arm:>7} {len(recs):>4} {rate('exact_' + arm, recs):>6.1%} "
              f"{rate('prefix_' + arm, recs):>6.1%} {f1:>7.3f} {over:>5}")
    print(f"{'prior':>7} {len(recs):>4} {rate('prior', recs):>6.1%}")

    print(f"\n{'template':<52} {'n':>3} {'nostop':>7} {'stop':>7} {'prior':>6}")
    for q in sorted({x["template"] for x in recs}):
        sub = [x for x in recs if x["template"] == q]
        print(f"{q[:50]:<52} {len(sub):>3} {rate('exact_nostop', sub):>6.1%} "
              f"{rate('exact_stop', sub):>6.1%} {rate('prior', sub):>5.1%}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(n=len(recs), n_predict=args.n_predict,
                                   stop=STOP, records=recs), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n[save] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
