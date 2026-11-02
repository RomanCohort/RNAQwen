"""MMLU / C-Eval multiple-choice harness for a llama.cpp server.

Why /completion and not /v1/chat/completions: the raw endpoint takes the prompt
verbatim, so no chat template is applied and the model never gets a chance to
emit a reasoning block.  There is nothing to disable -- that is exactly the
"no thinking" condition we want.

One request per question.  The prompt ends at the answer marker; we ask for a
handful of greedy tokens plus top-K logprobs, which yields three decisions from
the same forward pass:

  gen    -- the first generated token whose stripped text is a single A-D letter.
            This is what the model actually answers.  English prompts put the
            letter straight after "Answer:", while the Chinese template tends to
            emit a newline first, so the marker is allowed to be a step away.
            Headline metric.
  cloze  -- best logprob among every token in the FIRST position whose stripped
            text is a single A-D letter.  Matches the classic lm-eval cloze
            protocol; only meaningful when the letter really is the next token.
  strict -- as cloze but restricted to the canonical surface form (" A" for the
            English template, "A" for the Chinese one).

Results are appended to JSONL as they arrive, so a killed run resumes by
pointing the same --out at it again.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(ROOT, "data", "external", "bench")
MAPPING = os.path.join(ROOT, "data", "external", "ceval_subject_mapping.json")

LETTERS = ("A", "B", "C", "D")
NEG = -1e30

# MMLU marker is literally "Answer:"; the C-Eval ones are the answer-colon
# "\u7b54\u6848\uff1a" and the stem header
# "\u4ee5\u4e0b\u662f\u4e2d\u56fd\u5173\u4e8e{}\u7684\u5355\u9879\u9009\u62e9\u9898\uff0c\
# \u8bf7\u9009\u51fa\u5176\u4e2d\u7684\u6b63\u786e\u7b54\u6848\u3002"
MMLU_MARK = "Answer:"
CEVAL_MARK = "\u7b54\u6848\uff1a"
CEVAL_HEAD = ("\u4ee5\u4e0b\u662f\u4e2d\u56fd\u5173\u4e8e{}\u7684\u5355\u9879\u9009\u62e9\u9898\uff0c"
              "\u8bf7\u9009\u51fa\u5176\u4e2d\u7684\u6b63\u786e\u7b54\u6848\u3002")

_local = threading.local()


def sess() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = requests.Session()
        s.headers.update({"Content-Type": "application/json"})
        _local.s = s
    return s


def load_cn_names() -> dict:
    try:
        with open(MAPPING, encoding="utf-8") as f:
            return {k: v[1] for k, v in json.load(f).items()}
    except Exception:
        return {}


# ---------------------------------------------------------------- prompt text

def mmlu_block(question: str, choices, answer_letter=None) -> str:
    lines = [question.strip()]
    for letter, choice in zip(LETTERS, choices):
        lines.append("%s. %s" % (letter, str(choice).strip()))
    lines.append(MMLU_MARK + (answer_letter or ""))
    return "\n".join(lines)


def ceval_block(row, answer_letter=None) -> str:
    lines = [str(row["question"]).strip()]
    for letter in LETTERS:
        lines.append("%s. %s" % (letter, str(row[letter]).strip()))
    lines.append(CEVAL_MARK + (answer_letter or ""))
    return "\n".join(lines)


def build_tasks(bench, df, dev, nshot, limit, subjects=None):
    tasks = []
    cn = load_cn_names() if bench == "ceval" else {}
    for subject, grp in df.groupby("subject", sort=True):
        if subjects and subject not in subjects:
            continue
        grp = grp.reset_index(drop=True)
        if limit:
            grp = grp.head(limit)
        shots = []
        if nshot:
            pool = dev[dev["subject"] == subject].head(nshot)
            for _, srow in pool.iterrows():
                if bench == "mmlu":
                    shots.append(mmlu_block(srow["question"], srow["choices"],
                                            LETTERS[int(srow["answer"])]))
                else:
                    shots.append(ceval_block(srow, str(srow["answer"]).strip().upper()))
        for i, row in grp.iterrows():
            if bench == "mmlu":
                body = mmlu_block(row["question"], row["choices"])
                head, gold, strict_prefix = "", LETTERS[int(row["answer"])], " "
            else:
                body = ceval_block(row)
                head = CEVAL_HEAD.format(cn.get(subject, subject)) + "\n"
                gold = str(row["answer"]).strip().upper()
                strict_prefix = ""
            prompt = head + "\n\n".join(shots + [body]) if shots else head + body
            tasks.append({"bench": bench, "subject": subject, "idx": int(i),
                          "gold": gold, "strict_prefix": strict_prefix,
                          "prompt": prompt})
    return tasks


# ------------------------------------------------------------------- scoring

def score(prompt, port, topk, ngen, timeout, retries=3):
    body = {
        "prompt": prompt,
        "n_predict": ngen,
        "temperature": 0.0,
        "logprobs": topk,
        "cache_prompt": True,
        "stream": False,
    }
    url = "http://127.0.0.1:%d/completion" % port
    last = None
    for attempt in range(retries):
        try:
            r = sess().post(url, data=json.dumps(body), timeout=timeout)
            r.raise_for_status()
            cp = r.json().get("completion_probabilities") or []
            if not cp:
                raise RuntimeError("no completion_probabilities")
            return cp
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("scoring failed: %r" % (last,))


def letter_of(tok):
    bare = tok.strip()
    if len(bare) == 1 and bare.upper() in LETTERS:
        return bare.upper()
    return None


def decide(cp, strict_prefix):
    gen = None
    for entry in cp:
        gen = letter_of(entry.get("token", ""))
        if gen:
            break
    loose, strict = {}, {}
    for entry in (cp[0].get("top_logprobs") or []):
        tok = entry.get("token", "")
        letter = letter_of(tok)
        if not letter:
            continue
        lp = float(entry.get("logprob", NEG))
        if lp > loose.get(letter, NEG):
            loose[letter] = lp
        if tok == strict_prefix + letter and lp > strict.get(letter, NEG):
            strict[letter] = lp
    pick = lambda d: max(d, key=d.get) if d else None  # noqa: E731
    return gen, pick(loose), pick(strict), loose


def run_one(task, args):
    t0 = time.time()
    try:
        cp = score(task["prompt"], args.port, args.topk, args.ngen, args.timeout)
        gen, cloze, strict, loose = decide(cp, task["strict_prefix"])
        gold = task["gold"]
        row = {
            "bench": task["bench"], "subject": task["subject"], "idx": task["idx"],
            "gold": gold, "gen": gen, "cloze": cloze, "strict": strict,
            "ok": int(gen == gold), "ok_cloze": int(cloze == gold),
            "ok_strict": int(strict == gold),
            "n_letters": len(loose), "secs": round(time.time() - t0, 3),
        }
        if (not loose or args.dump_top) and task["idx"] < args.dump_n:
            row["top"] = [e.get("token") for e in (cp[0].get("top_logprobs") or [])
                          ][:12]
            row["gen_tokens"] = [e.get("token") for e in cp]
        return row
    except Exception as exc:  # noqa: BLE001
        return {"bench": task["bench"], "subject": task["subject"],
                "idx": task["idx"], "gold": task["gold"], "gen": None,
                "cloze": None, "strict": None, "ok": 0, "ok_cloze": 0,
                "ok_strict": 0, "n_letters": 0,
                "secs": round(time.time() - t0, 3), "error": repr(exc)}


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=["mmlu", "ceval"])
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--nshot", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="per-subject cap, 0=all")
    ap.add_argument("--subjects", default="", help="comma separated subset")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--ngen", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-top", action="store_true")
    ap.add_argument("--dump-n", type=int, default=3)
    args = ap.parse_args()

    if args.bench == "mmlu":
        df = pd.read_parquet(os.path.join(DATA, "mmlu_test.parquet"))
        dev = pd.read_parquet(os.path.join(DATA, "mmlu_dev.parquet"))
        if args.nshot:
            need, have = set(df["subject"]), set(dev["subject"])
            if not need <= have:
                print("WARN: filling %d subjects missing from dev" % len(need - have))
                fill = [df[df["subject"] == s].head(1) for s in sorted(need - have)]
                dev = pd.concat([dev] + fill)
    else:
        df = pd.read_parquet(os.path.join(DATA, "ceval_val.parquet"))
        dev = pd.read_parquet(os.path.join(DATA, "ceval_dev.parquet"))

    subjects = [s for s in args.subjects.split(",") if s] or None
    tasks = build_tasks(args.bench, df, dev, args.nshot, args.limit, subjects)
    if not tasks:
        print("no tasks")
        return 1

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["bench"], r["subject"], r["idx"]))
                except Exception:
                    pass
    todo = [t for t in tasks if (t["bench"], t["subject"], t["idx"]) not in done]
    print("tasks=%d already_done=%d todo=%d nshot=%d workers=%d port=%d"
          % (len(tasks), len(done), len(todo), args.nshot, args.workers, args.port),
          flush=True)

    lock = threading.Lock()
    fh = open(args.out, "a", encoding="utf-8")
    t_start = time.time()
    state = {"n": 0, "ok": 0, "ok_cloze": 0, "ok_strict": 0}

    def work(task):
        row = run_one(task, args)
        with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            state["n"] += 1
            state["ok"] += row["ok"]
            state["ok_cloze"] += row["ok_cloze"]
            state["ok_strict"] += row["ok_strict"]
            n = state["n"]
            if n % 100 == 0 or n == len(todo):
                rate = n / max(1e-9, time.time() - t_start)
                print("  %6d/%d  gen=%.4f cloze=%.4f  %.2f q/s  eta=%.1f min"
                      % (n, len(todo), state["ok"] / n, state["ok_cloze"] / n, rate,
                         (len(todo) - n) / max(1e-9, rate) / 60), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()

    summarize(args)
    return 0


def summarize(args):
    rows = []
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    d = pd.DataFrame(rows)
    print("\n=== %s  n=%d  %d-shot  port=%d ===" % (args.bench, len(d), args.nshot,
                                                    args.port))
    # cloze is the headline: it is the classic single-position loglikelihood
    # protocol, so it lines up with published MMLU / C-Eval numbers.  gen is the
    # greedy-generation cross-check.
    for col, label in [("ok_cloze", "cloze*"), ("ok", "gen   "), ("ok_strict", "strict")]:
        print("  %s acc=%.4f" % (label, d[col].mean()))
    print("  parse_fail(gen)=%d" % int(d["gen"].isna().sum()))
    per = d.groupby("subject")[["ok", "ok_cloze", "ok_strict"]].mean()
    per["n"] = d.groupby("subject")["ok"].size()
    print("\nworst 10 by gen acc:")
    print(per.sort_values("ok").head(10).to_string(float_format=lambda v: "%.3f" % v))
    summary = {
        "bench": args.bench, "nshot": args.nshot, "port": args.port,
        "n": int(len(d)), "acc_gen": float(d["ok"].mean()),
        "acc_cloze": float(d["ok_cloze"].mean()),
        "acc_strict": float(d["ok_strict"].mean()),
        "parse_fail": int(d["gen"].isna().sum()),
        "per_subject": {s: {"acc_gen": float(v["ok"]), "acc_cloze": float(v["ok_cloze"]),
                            "acc_strict": float(v["ok_strict"]), "n": int(v["n"])}
                        for s, v in per.iterrows()},
    }
    with open(args.out + ".summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nsummary ->", args.out + ".summary.json")


if __name__ == "__main__":
    sys.exit(main())
