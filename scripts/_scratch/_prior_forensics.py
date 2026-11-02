"""Is the zeroed-code arm a question-only prior predictor, or something worse?

Three claims get tested here, all offline, against the 300 stored generations
(`reports/acc_by_source_D_qwen36_sos_final.json.raw.json`) plus the training
corpus the run actually used (`data/processed/rnaqa_seqtask_full.jsonl`):

  1. what the question-only predictor is worth.  "Always answer each template
     with its most frequent training answer" is checkable without an LLM:
     count the train answers, then score the corpus test split.
  2. whether the *zeroed* arm's near-constant answer is that predictor.  The
     honest test is string identity: does arm C emit the template's mode?
  3. how much of arm A's score is prior.  Restricted to items whose gold is
     NOT the mode, a prior predictor is wrong by construction, so anything
     the model gets there it got from the code.

Writes reports/prior_forensics.json.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/processed/rnaqa_seqtask_full.jsonl"
RAW = ROOT / "reports/acc_by_source_D_qwen36_sos_final.json.raw.json"
OUT = ROOT / "reports/prior_forensics.json"
ARMS = ("A", "B", "C")


def norm(s: str) -> str:
    return "".join(c for c in s if c.isalpha()).upper()


def main() -> int:
    train: dict[str, Counter] = {}
    test: dict[str, Counter] = {}
    with CORPUS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sp, q = r.get("split"), r.get("question")
            if sp == "train":
                train.setdefault(q, Counter())[norm(r["answer"])] += 1
            elif sp == "test":
                test.setdefault(q, Counter())[norm(r["answer"])] += 1

    d = json.loads(RAW.read_text(encoding="utf-8"))
    templates = [q for q in d if q in train]
    mode = {q: train[q].most_common(1)[0][0] for q in templates}

    # ---- 1. question-only prior, corpus wide ----
    n_te = hit_te = 0
    rows = []
    for q in templates:
        m = sum(test[q].values())
        h = test[q].get(mode[q], 0)
        n_te += m
        hit_te += h
        rows.append(dict(template=q, mode=mode[q],
                         train_share=train[q].most_common(1)[0][1] / sum(train[q].values()),
                         n_test=m, prior_acc=h / m))
    print(f"question-only prior, 10 templates, corpus test split: "
          f"{hit_te / n_te:.2%} ({hit_te:,}/{n_te:,})")

    # ---- 2/3. the stored generations ----
    stat = {}
    for arm in ARMS:
        s = dict(n=0, slots=0, gold_is_mode=0, gold_is_mode_px=0,
                 nonmode=0, nonmode_ex=0, nonmode_px=0, prior_err=0,
                 slot_eq_mode=0, distinct_per_template=[])
        for q in templates:
            mg = mode[q]
            slots = Counter()
            for _stored, gold, _f1, raw, _q in d[q][arm]:
                g = norm(gold)
                slot = norm(raw)[:len(g)]
                slots[slot] += 1
                s["n"] += 1
                s["slots"] += 1
                s["slot_eq_mode"] += (slot == mg)
                if g == mg:
                    s["gold_is_mode"] += 1
                    s["gold_is_mode_px"] += (slot == g)
                else:
                    s["nonmode"] += 1
                    s["nonmode_ex"] += (norm(raw) == g)
                    s["nonmode_px"] += bool(g) and norm(raw).startswith(g)
                    s["prior_err"] += (slot == mg)
            top = slots.most_common(1)[0]
            s["distinct_per_template"].append(dict(
                template=q, distinct=len(slots), mode=top[0], share=top[1] / 30,
                corpus_mode=mg, is_corpus_mode=top[0] == mg))
        stat[arm] = s

    print(f"\n{'arm':>4} {'slot=#mode':>10} {'gold=#mode':>10} {'px on those':>12} "
          f"{'gold!=mode':>11} {'exact':>7} {'prefix':>7} {'emit mode anyway':>17}")
    for arm, s in stat.items():
        print(f"{arm:>4} {s['slot_eq_mode']:>10} {s['gold_is_mode']:>10} "
              f"{s['gold_is_mode_px'] / max(s['gold_is_mode'], 1):>11.1%} "
              f"{s['nonmode']:>11} {s['nonmode_ex'] / s['nonmode']:>6.1%} "
              f"{s['nonmode_px'] / s['nonmode']:>6.1%} {s['prior_err']:>17}")

    print("\nzeroed arm (C), most frequent answer slot per template, n=30 each:")
    print(f"{'template':<52} {'distinct':>8} {'mode':<10} {'share':>6} "
          f"{'corpus mode':<10} same?")
    for r in stat["C"]["distinct_per_template"]:
        print(f"{r['template'][:50]:<52} {r['distinct']:>8} {r['mode']:<10} "
              f"{r['share']:>5.0%} {r['corpus_mode']:<10} "
              f"{'yes' if r['is_corpus_mode'] else 'NO'}")

    OUT.write_text(json.dumps(dict(
        prior=dict(n_test=n_te, prior_acc=hit_te / n_te, per_template=rows),
        arms={a: {k: v for k, v in stat[a].items() if k != "distinct_per_template"}
              for a in ARMS},
        arm_c_modal=stat["C"]["distinct_per_template"],
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
