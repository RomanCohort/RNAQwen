"""实测 DeepSeek 标注的 token 消耗，然后外推成本。

**不猜，量。**

做法：
  1. 从 litsumm 里取 N 条真实样本（`summary` 当 D&C 源文本、`context` 当 AIS 源文本）
  2. 用和 `12_generate_qa.py` **完全相同**的 prompt 走一次真实调用
  3. 把每次响应的 `usage` 累加（input / cache-hit / output）
  4. 按当前价目表外推到"全量"的三种规模

价目（2026-10 抓的官方文档，单位 USD / 1M tokens）：
    deepseek-flash      输入(缓存未命中) 0.15 空闲 / 0.30 高峰
                        输入(缓存命中)   0.003 空闲 / 0.006 高峰
                        输出             0.60 空闲 / 1.20 高峰
    deepseek-v4-pro     输入(未命中) 0.66 / 1.32      输出 1.98 / 3.96
空闲时段价格是高峰的一半；高峰 = 北京时间 周一至周五 9:00-12:00、14:00-18:00。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

from rnaqwen import secrets  # noqa: E402
from rnaqwen.data.prompts import QA_CLOSED, QA_FROM_TEXT, SYSTEM_RNA_EXPERT, fill  # noqa: E402

# USD / 1M tokens
PRICES = {
    "deepseek-flash":    {"in_miss": (0.15, 0.30), "in_hit": (0.003, 0.006), "out": (0.60, 1.20)},
    "deepseek-v4-pro":   {"in_miss": (0.66, 1.32), "in_hit": (0.022, 0.044), "out": (1.98, 3.96)},
}
USD_CNY = 7.1   # 粗略汇率，只用于给人民币量级


def call(model: str, messages: list[dict], max_tokens: int, key: str, temperature: float = 0.4,
         thinking: str = "disabled", effort: str | None = None) -> dict:
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature}
    if thinking is not None:
        payload["thinking"] = {"type": thinking}
    if effort:
        payload["reasoning_effort"] = effort
    body = json.dumps(payload).encode()
    req = urllib.request.Request("https://api.deepseek.com/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--litsumm", type=Path, default=ROOT / "data/external/litsumm_v1_5_data.parquet")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--thinking", default="disabled", choices=["disabled", "enabled"],
                    help="DeepSeek 默认开思考且 effort=high；数据生成必须关掉（否则 JSON 被截断）")
    ap.add_argument("--effort", default=None, choices=[None, "low", "high", "max"])
    ap.add_argument("--n", type=int, default=8, help="抽多少条真实样本")
    ap.add_argument("--n-pairs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/deepseek_cost_estimate.json")
    args = ap.parse_args()

    key = secrets.resolve_api_key(provider="deepseek")
    print(f"[cost] 密钥 {secrets.mask(key)}  模型 {args.model}")

    import pyarrow.parquet as pq
    tbl = pq.read_table(args.litsumm, columns=["urs_taxid", "ent_id", "context", "summary", "truthful"])
    d = tbl.to_pydict()
    picks = [i for i in range(len(d["urs_taxid"])) if d["truthful"][i] is not False][: args.n]
    print(f"[cost] 取样 {len(picks)} 条\n")

    calls: list[dict] = []
    t0 = time.time()
    for i in picks:
        for variant, src in (("dc", d["summary"][i]), ("ais", d["context"][i])):
            for kind, tpl in (("open", QA_FROM_TEXT), ("closed", QA_CLOSED)):
                src = (src or "").strip()
                if not src:
                    continue
                msgs = [{"role": "system", "content": SYSTEM_RNA_EXPERT},
                        {"role": "user", "content": fill(tpl, text=src, n=args.n_pairs)}]
                try:
                    r = call(args.model, msgs, 1100, key, thinking=args.thinking, effort=args.effort)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [fail] {d['ent_id'][i]} {variant}/{kind}: {type(exc).__name__} {str(exc)[:80]}")
                    continue
                u = r.get("usage") or {}
                calls.append({
                    "ent_id": d["ent_id"][i], "variant": variant, "kind": kind,
                    "prompt_chars": len(src),
                    "prompt_tokens": u.get("prompt_tokens", 0),
                    "cache_hit": u.get("prompt_cache_hit_tokens", 0),
                    "cache_miss": u.get("prompt_cache_miss_tokens", 0),
                    "completion_tokens": u.get("completion_tokens", 0),
                    "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
                })
                print(f"  {d['ent_id'][i]:>18s} {variant}/{kind:6s} src={len(src):6d}字符 "
                      f"in={u.get('prompt_tokens',0):5d} (hit {u.get('prompt_cache_hit_tokens',0)} / "
                      f"miss {u.get('prompt_cache_miss_tokens',0)}) out={u.get('completion_tokens',0):5d}"
                      f"{'  think='+str((u.get('completion_tokens_details') or {}).get('reasoning_tokens')) if (u.get('completion_tokens_details') or {}).get('reasoning_tokens') else ''}")
    dt = time.time() - t0

    if not calls:
        print("[fail] 一个成功调用都没有")
        return 1

    def mean(k: str) -> float:
        return sum(c[k] for c in calls) / len(calls)

    per_call = {k: mean(k) for k in ("prompt_tokens", "cache_hit", "cache_miss", "completion_tokens")}
    per_call_miss_nonzero = sum(c["cache_miss"] for c in calls) / len(calls)
    by_variant: dict[str, dict] = {}
    for v in ("dc", "ais"):
        sub = [c for c in calls if c["variant"] == v]
        if sub:
            by_variant[v] = {
                "n": len(sub),
                "src_chars": round(sum(c["prompt_chars"] for c in sub) / len(sub)),
                "in_miss": round(sum(c["cache_miss"] for c in sub) / len(sub)),
                "out": round(sum(c["completion_tokens"] for c in sub) / len(sub)),
            }

    print(f"\n[cost] {len(calls)} 次调用，用时 {dt:.1f}s")
    print(f"[cost] 单次平均：输入 {per_call['prompt_tokens']:.0f} tok"
          f"（缓存命中 {per_call['cache_hit']:.0f} / 未命中 {per_call_miss_nonzero:.0f}），"
          f"输出 {per_call['completion_tokens']:.0f} tok")
    for v, s in by_variant.items():
        print(f"[cost]   {v}: 源文本 {s['src_chars']} 字符 → 输入 {s['in_miss']} tok，输出 {s['out']} tok")

    # ---- 外推 ----
    # 每个 RNA：2 变体 × 2 类型 = 4 次调用
    per_record = {
        "in_miss": sum(c["cache_miss"] for c in calls) / len({c["ent_id"] for c in calls}),
        "out": sum(c["completion_tokens"] for c in calls) / len({c["ent_id"] for c in calls}),
    }

    def cost(tokens_in: float, tokens_out: float, model: str, peak: bool) -> dict:
        p = PRICES[model]
        i = 1 if peak else 0
        cin = tokens_in / 1e6 * p["in_miss"][i]
        cout = tokens_out / 1e6 * p["out"][i]
        return {"usd": cin + cout, "cny": (cin + cout) * USD_CNY}

    # 两轮蒸馏那步（验真筛选）也要算：每生成 1 条 QA 要多 1 次"判定"调用，
    # 输入 ≈ 源文本 + QA，输出极短
    verify_in_per_call = per_call["prompt_tokens"] * 0.6
    verify_out_per_call = 60

    scales = [
        ("我们的 litsumm 子集", 3800),
        ("原版训练子集 5K RNA", 5000),
        ("原版全量 407,616 RNA", 407616),
    ]

    report: dict = {
        "model": args.model,
        "measured_calls": len(calls),
        "avg_per_call": per_call,
        "avg_cache_miss_per_call": round(per_call_miss_nonzero, 1),
        "by_variant": by_variant,
        "per_record": {"in_miss": round(per_record["in_miss"], 1), "out": round(per_record["out"], 1)},
        "price_table_usd_per_mtok": PRICES,
        "extrapolation": [],
    }

    print(f"\n[cost] 每个 RNA（2 变体 × 2 类型 = 4 次调用）："
          f"输入 {per_record['in_miss']:.0f} tok（未命中），输出 {per_record['out']:.0f} tok")
    print(f"\n{'规模':28s} {'输入 tok':>14s} {'输出 tok':>12s} {'空闲 $':>9s} {'高峰 $':>9s} {'空闲 ¥':>9s}")
    print("-" * 92)
    for label, n in scales:
        tin = per_record["in_miss"] * n
        tout = per_record["out"] * n
        c_off = cost(tin, tout, args.model, peak=False)
        c_pk = cost(tin, tout, args.model, peak=True)
        row = {"label": label, "n_rna": n, "tokens_in": round(tin), "tokens_out": round(tout),
               "offpeak": c_off, "peak": c_pk,
               "offpeak_with_verifier": cost(tin + verify_in_per_call * n * 4,
                                             tout + verify_out_per_call * n * 4, args.model, peak=False)}
        report["extrapolation"].append(row)
        print(f"{label:28s} {tin:14,.0f} {tout:12,.0f} {c_off['usd']:9.2f} {c_pk['usd']:9.2f} {c_off['cny']:9.1f}")

    print("\n[cost] 加上「验真筛选」那一轮（每生成 1 条 QA 多 1 次判定调用）后的空闲时段价：")
    for row in report["extrapolation"]:
        v = row["offpeak_with_verifier"]
        print(f"        {row['label']:28s} ${v['usd']:8.2f}  ¥{v['cny']:8.1f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[cost] 报告 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
