"""General-purpose chat ability: RNAQwen-preview vs the stock base, same battery.

Why this exists
---------------
The service RNAQwen-preview would replace (qwen-api :8001 -> llama-server :8091) is
NOT a different model family.  It serves

    C:\qwen-api\models\Qwen3.6-35B-A3B-msc\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

which is the same base RNAQwen-preview was built from.  So the question "can it take
over the general chat model" reduces to one thing: did 7500 steps of RNA-only
fine-tuning damage the general ability of the base?  That is a clean A/B, and this
script is that A/B.

Both arms get byte-identical prompts and identical sampling, so a difference in the
output is attributable to the weights and nothing else.

    python scripts/_scratch/_general_qa_ab.py --out reports/general_qa_ab.json

Read the result as a screen, not a verdict: a dozen prompts can show a collapsed
model or a clearly weaker one, and cannot show that a model is as good as another.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# (id, prompt, max_tokens, checker-name or "")
# The checker is a cheap automatic read on the prompts that have one right answer.
# Everything else is left to a human eye.
BATTERY = [
    ("fact",     "珠穆朗玛峰的海拔是多少米？用一句话回答。", 120, ""),
    ("math",     "一个水池，甲管单独注满需要 6 小时，乙管单独注满需要 4 小时。"
                 "两管同时开，多少小时注满？请给出计算过程，最后一行只写结果。", 300, "math"),
    ("arith",    "17 × 23 = ? 只输出数字。", 40, "arith"),
    ("code",     "用 Python 写一个函数 is_palindrome(s)，忽略大小写和非字母数字字符，"
                 "返回布尔值。只给代码，不要解释。", 300, "code"),
    ("debug",    "下面这段 Python 有 bug，指出错在哪并给出修正后的代码：\n"
                 "def mean(xs):\n    total = 0\n    for x in xs:\n"
                 "        total += x\n    return total / len(xs)\nprint(mean([]))", 400, "code"),
    ("json",     "列出三本中国科幻小说，用 JSON 数组输出，每个对象字段为 "
                 "title / author / year。不要输出 JSON 之外的内容。", 400, "json"),
    ("translate", "把下面这句翻译成英文，只输出译文：\n今天下午图书馆三点开门。", 150, ""),
    ("write",    "写一段 80 字左右的推荐语，介绍一杯手冲咖啡。", 300, ""),
    ("summarise", "把下面这段压缩成一句话：\n"
                  "城际铁路的新线路今天开始试运行，全程用时比既有线路缩短约四十分钟，"
                  "沿线共设七个车站，其中三个为新建车站，初期每天开行十六对列车。", 150, ""),
    ("logic",    "所有会飞的动物都有翅膀。企鹅没有翅膀吗？请先判断前提是否为真，再回答"
                 "「企鹅会不会飞」，并说明推理过程。", 700, "logic"),
    ("trap",     "「林黛玉倒拔垂杨柳」这个情节出自哪本书的哪一回？如果这个情节并不存在，"
                 "请直接说不存在。", 250, "trap"),
    ("en",       "What is the capital of Australia? Answer with the city name only.", 40, ""),
    ("self",     "你是谁？你的训练者是谁？", 200, ""),
    ("follow",   "给我三条睡前放松的建议，每条不超过 15 个字，用编号列表输出。", 250, "follow"),
]


def chat(endpoint: str, prompt: str, max_tokens: int, thinking: bool = False,
         timeout: int = 900) -> dict:
    # Thinking is off on both arms, and that is a deliberate choice rather than a
    # convenience.  Measured 2026-10-16 on the stock model: with thinking on it spends
    # the whole budget on reasoning_content and returns an EMPTY content field (600
    # tokens, 2295 chars of reasoning, finish_reason=length, content="").  The battery
    # budgets are sized for answers, so thinking on would score both arms at zero.
    #
    # It is also the only fair setting for this comparison: RNAQwen-preview was trained
    # with no_think, so giving the stock arm a reasoning loop it does not have would be
    # measuring the budget, not the weights.  "--thinking" exists to check that
    # difference on purpose, not for the main run.
    #
    # A "/no_think" suffix in the prompt does NOT work here (measured: still 300 tokens
    # of reasoning).  chat_template_kwargs is what actually reaches the template.
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "top_p": 1.0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as f:
        out = json.loads(f.read().decode("utf-8"))
    dt = time.time() - t0
    ch = out.get("choices") or [{}]
    msg = ch[0].get("message") or {}
    usage = out.get("usage") or {}
    return dict(text=(msg.get("content") or "").strip(),
                reasoning_chars=len(msg.get("reasoning_content") or ""),
                finish=ch[0].get("finish_reason"),
                completion_tokens=usage.get("completion_tokens"),
                prompt_tokens=usage.get("prompt_tokens"), seconds=round(dt, 2),
                tps=round((usage.get("completion_tokens") or 0) / dt, 2) if dt > 0 else None)


def check(name: str, text: str) -> dict:
    """Cheap automatic reads. -1 means the checker could not decide; that is not a fail."""
    t = text.strip()
    if name == "arith":
        nums = set(re.findall(r"\d+", t))
        return {"ok": 391 in {int(n) for n in nums}}
    if name == "math":
        return {"ok": any(k in t for k in ("2.4", "12/5", "2 小时 24", "2小时24")),
                "note": "want 12/5 = 2.4 h"}
    if name == "code":
        return {"ok": "def " in t and ("return" in t or "total" in t),
                "note": "has a def with a body"}
    if name == "json":
        m = re.search(r"\[.*\]", t, re.S)
        if not m:
            return {"ok": False, "note": "no array"}
        try:
            arr = json.loads(m.group(0))
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "note": f"unparseable: {exc}"}
        need = {"title", "author", "year"}
        return {"ok": all(isinstance(o, dict) and need <= set(o) for o in arr),
                "note": f"{len(arr)} objects"}
    if name == "trap":
        return {"ok": any(k in t for k in ("不存在", "没有", "并非", "未出现")),
                "note": "should refuse the premise"}
    if name == "logic":
        return {"ok": ("没有翅膀" in t or "有翅膀" in t) and "不会飞" in t,
                "note": "premise is false: penguins do not fly"}
    if name == "follow":
        return {"ok": t.count("\n") >= 2, "note": "3 lines"}
    return {"ok": -1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rnaqwen", default="http://127.0.0.1:8099/v1/chat/completions")
    ap.add_argument("--stock", default="http://127.0.0.1:8091/v1/chat/completions")
    ap.add_argument("--only", default="", help="comma-separated battery ids")
    ap.add_argument("--arms", default="both",
                    choices=("both", "rnaqwen", "stock"),
                    help="run one arm while the other side is not available yet; "
                         "tokens/s from a single-arm run is NOT comparable across runs")
    ap.add_argument("--thinking", action="store_true",
                    help="leave the reasoning loop on (off by default, see chat())")
    ap.add_argument("--out", default="reports/general_qa_ab.json")
    args = ap.parse_args()

    want = {s.strip() for s in args.only.split(",") if s.strip()}
    arms = [a for a in ("rnaqwen", "stock") if args.arms in (a, "both")]
    rows = []
    for bid, prompt, mx, ck in BATTERY:
        if want and bid not in want:
            continue
        row = {"id": bid, "prompt": prompt, "max_tokens": mx}
        for arm in arms:
            ep = args.rnaqwen if arm == "rnaqwen" else args.stock
            try:
                r = chat(ep, prompt, mx, thinking=args.thinking)
                r["check"] = check(ck, r["text"])
            except Exception as exc:  # noqa: BLE001
                r = {"error": f"{type(exc).__name__}: {exc}", "check": {"ok": -1}}
            row[arm] = r
        for a in ("rnaqwen", "stock"):
            row.setdefault(a, {"error": "not run", "check": {"ok": -1}})
        rows.append(row)
        print(f"[{bid}] " + "  ".join(
            f"{a} ok={row[a]['check'].get('ok')}" for a in arms), flush=True)

    auto = [r for r in rows if r[arms[0]]["check"].get("ok") in (0, 1, True, False)]
    def score(arm):
        vals = [1 if r[arm]["check"].get("ok") else 0 for r in auto]
        return f"{sum(vals)}/{len(vals)}" if vals else "n/a"

    print(f"\n可自动判分的 {len(auto)} 项：" +
          "   ".join(f"{a} {score(a)}" for a in arms))
    print("平均出词速度：" + "   ".join(
        f"{a} {sum(r[a].get('tps') or 0 for r in rows)/len(rows):.2f} tok/s" for a in arms))
    (ROOT / args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
