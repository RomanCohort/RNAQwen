"""自由文本生物学 QA —— 只改写，不生成。用 requests + 线程池（openai 包没装）。

素材：4,481 条带 annotation 的 RNA。其余 145,201 条没有描述，**不能用** —— 只能编。

三条硬规矩：
  1. 问题里不出现 RNA 名字 / 类型 / 物种
  2. 答案必须有多样性 —— 模板化的扔掉（用 52_ 的思路：唯一率 + 熵 + 弃答率）
  3. 提示词写死「只能用 annotation 里的事实」

先跑试点（--n 20），看每个模板的分布再决定全量。
"""
from __future__ import annotations

import argparse
import os
import json
import math
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"
OUT = ROOT / "data/processed/rnaqa_freetext.jsonl"
URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

QUESTIONS = [
    "Describe this RNA.",
    "What is the biological function of this RNA?",
    "Where in the cell is this RNA localized?",
    "Is this RNA associated with any disease or condition?",
    "What RNA-binding proteins or complexes interact with this RNA?",
]

SYSTEM = ("You write concise, factual answers about RNA using ONLY the provided "
          "curated annotation.")

PROMPT = """Below is a curated annotation of an RNA, extracted from the scientific literature.

<annotation>
{ann}
</annotation>

Question: {q}

Write the answer using ONLY facts explicitly present in the annotation above.

Hard rules:
1. Do NOT introduce any gene name, disease, pathway, organism, protein, or fact that
   is not explicitly written in the annotation.
2. If the annotation does not contain information that answers the question, output
   exactly this and nothing else: The provided annotation does not state this.
3. Do not speculate, do not generalize from the RNA type, do not use outside knowledge.
4. Answer in at most 60 words.

Answer:"""

_lock = threading.Lock()
_stat = Counter()


def entropy(counts):
    t = sum(counts)
    return -sum((c / t) * math.log(c / t) for c in counts if c > 0) if t else 0.0


def call(key, r, q, retries=4, timeout=90):
    ann = (r.get("annotation") or "").strip()
    if len(ann) < 80:
        return None
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": PROMPT.format(ann=ann[:4000], q=q)}],
        "temperature": 0.3, "max_tokens": 220,
        "thinking": {"type": "disabled"},
    }
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for a in range(retries):
        try:
            resp = requests.post(URL, json=body, headers=hdr, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(3 * (a + 1)); continue
            resp.raise_for_status()
            j = resp.json()
            txt = (j["choices"][0]["message"]["content"] or "").strip()
            with _lock:
                u = j.get("usage") or {}
                _stat["in"] += u.get("prompt_tokens", 0)
                _stat["out"] += u.get("completion_tokens", 0)
                _stat["ok"] += 1
            if not txt:
                return None
            return dict(uid=r["uid"], sequence=r["sequence"], question=q, answer=txt,
                        template=q, source="freetext_llm", kind="open_ended",
                        annotation_chars=len(ann))
        except Exception as e:
            if a == retries - 1:
                with _lock:
                    _stat[f"err:{type(e).__name__}"] += 1
                return None
            time.sleep(2 ** a + random.random())
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--keep", nargs="*", default=[
        "Describe this RNA.",
        "What is the biological function of this RNA?",
        "Is this RNA associated with any disease or condition?"],
        help="只对这几个模板生成（上一轮筛选出来的）")
    args = ap.parse_args()
    if not args.key:
        import os
        args.key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not args.key:
        print("缺少 key"); return 1

    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("sequence") and (r.get("annotation") or "").strip():
                recs.append(r)
    print(f"有 annotation 的 RNA {len(recs):,}")
    random.Random(20261010).shuffle(recs)
    recs = recs[:args.n]

    # 只对保留的模板调用 —— 之前对全部 5 个调用、只过滤输出，2/5 的调用是白花的
    tasks = [(r, q) for r in recs for q in args.keep]
    # 增量写盘：上次 22,405 次调用跑到 79% 卡住，结果全在内存里，杀掉就全丢。
    # 现在每拿到一条就 append 一次（keep 的判定用 pilot 的结果，全量时直接用上轮的 keep）。
    # 断点续跑：读已有的输出，跳过已完成的 (uid, question)。
    # 上次卡住时结果全在内存里，杀掉全丢，只能从零重跑 -- 那次白花了钱。
    done_keys = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    r0 = json.loads(line)
                    done_keys.add((r0["uid"], r0["question"]))
                except Exception:
                    pass
        print(f"[resume] 已完成 {len(done_keys):,} 条，将跳过")
    tasks = [t for t in tasks if (t[0]["uid"], t[1]) not in done_keys]
    print(f"[resume] 本轮需跑 {len(tasks):,} 次调用")
    fout = open(args.out, "a", encoding="utf-8")
    print(f"样本 {len(recs)} x {len(QUESTIONS)} 模板 = {len(tasks)} 次调用   并发 {args.concurrency}")
    t0 = time.time()
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(call, args.key, r, q) for r, q in tasks]
        for fu in as_completed(futs):
            res = fu.result()
            done += 1
            if res:
                results.append(res)
                if res["question"] in args.keep:
                    fout.write(json.dumps(res, ensure_ascii=False) + "\n")
                    fout.flush()
            if done % 20 == 0 or done == len(tasks):
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  成功 {len(results)}  {done/el:.1f}/s  "
                      f"ETA {(len(tasks)-done)/max(done/el,1e-9)/60:.1f} min", flush=True)
    el = time.time() - t0
    print(f"\n完成  成功 {len(results)}  用时 {el:.0f}s  ({len(tasks)/el:.1f}/s)")
    print(f"tokens  in {_stat['in']:,}  out {_stat['out']:,}")
    errs = {k: v for k, v in _stat.items() if k.startswith("err:")}
    if errs:
        print("错误:", errs)

    print()
    print("=" * 82)
    print("每个模板的答案分布 —— 模板化的必须扔掉")
    print("=" * 82)
    byq = defaultdict(list)
    for r in results:
        byq[r["question"]].append(r["answer"])
    print(f"{'问题':<50} {'n':>4} {'唯一':>6} {'熵':>6} {'唯一率':>7} {'弃答%':>7}  判定")
    keep = []
    for q in QUESTIONS:
        ans = byq.get(q, [])
        if not ans:
            continue
        c = Counter(ans)
        uniq = len(c) / len(ans)
        refusal = sum(1 for a in ans if "does not state" in a.lower()) / len(ans)
        ok = uniq > 0.7 and refusal < 0.3
        if ok:
            keep.append(q)
        print(f"  {q[:48]:<48} {len(ans):>4} {len(c):>6} {entropy(list(c.values())):>6.2f} "
              f"{uniq:>7.2f} {100*refusal:>6.1f}%  {'收' if ok else '扔'}")
    print(f"\n保留 {len(keep)}/{len(QUESTIONS)}")

    fout.close()
    print(f"[save] {args.out}  {sum(1 for r in results if r['question'] in keep)} 条")

    ann = {r["uid"]: (r.get("annotation") or "") for r in recs}
    print()
    print("=" * 82)
    print(f"抽查 {args.show} 条 —— 逐字对照 annotation")
    print("=" * 82)
    for r in results[:args.show]:
        print(f"\nQ: {r['question']}\nA: {r['answer'][:220]}")
        print(f"@: {ann.get(r['uid'],'')[:220]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
