"""带 **usage 计量 + 余额前后对比** 地跑 `12_generate_qa.py`（不改 src，只在运行时打补丁）。

为什么不用直接跑 12
------------------
`OpenAICompatGenerator._one()` 只返回文本，把响应里的 `usage` 丢掉了，
所以算不出真实花费、也验证不了"思考模式确实关了"。
这里在运行时把 `_one` 换成一个**逐字镜像 llm.py 逻辑但顺带记账**的版本
（不是改 src —— 只是测试侧的 monkeypatch），同时前后各查一次 `/user/balance`。

真金白银的判据是**余额差**；token 计数用来算"每个 RNA 的成本"，便于外推。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen import llm, secrets  # noqa: E402

_lock = threading.Lock()
USAGE: dict[str, float] = {}


def _bump(key: str, val: float) -> None:
    USAGE[key] = USAGE.get(key, 0) + val


def balance() -> dict:
    key = secrets.resolve_api_key(provider="deepseek")
    req = urllib.request.Request("https://api.deepseek.com/user/balance",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _one_with_usage(self, messages, max_new_tokens, temperature, top_p):
    """镜像 `llm.OpenAICompatGenerator._one`，额外累计 usage。"""
    body = self._body(messages, max_new_tokens, temperature, top_p)
    last: Exception | None = None
    for attempt in range(self.retries):
        try:
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode())
            u = payload.get("usage") or {}
            with _lock:
                _bump("calls", 1)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens",
                          "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
                    if u.get(k) is not None:
                        _bump(k, int(u[k]))
                det = u.get("completion_tokens_details") or {}
                rt = det.get("reasoning_tokens") or u.get("reasoning_tokens") or 0
                _bump("reasoning_tokens", int(rt))
                if rt:
                    _bump("calls_with_reasoning", 1)
                fr = payload.get("choices", [{}])[0].get("finish_reason")
                if fr and fr != "stop":
                    _bump(f"finish_{fr}", 1)
            msg = payload["choices"][0]["message"]
            text = msg.get("content") or ""
            if not text.strip():
                text = msg.get("reasoning_content") or ""
            return text
        except Exception as exc:  # noqa: BLE001
            last = exc
            with _lock:
                _bump("http_errors", 1)
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"生成失败: {last}") from last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--model", default="deepseek:deepseek-flash")
    ap.add_argument("--variants", default="ais,dc")
    ap.add_argument("--kinds", default="open,closed")
    ap.add_argument("--n-pairs", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--keep-dropped", action="store_true")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--no-patch", action="store_true", help="不装 usage 补丁（纯跑）")
    args = ap.parse_args()

    if not args.no_patch:
        llm.OpenAICompatGenerator._one = _one_with_usage

    b0 = None
    try:
        b0 = balance()
        print("[cost] 起始余额:", json.dumps(b0.get("balance_infos", b0), ensure_ascii=False), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("[cost] 余额查询失败:", type(exc).__name__, flush=True)

    # 直接调用 12 的 main()，保证口径一字不改
    spec = importlib.util.spec_from_file_location("genqa", ROOT / "scripts" / "12_generate_qa.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    argv = ["12_generate_qa.py", "--in", str(args.inp), "--out", str(args.out),
            "--model", args.model, "--variants", args.variants, "--kinds", args.kinds,
            "--n-pairs", str(args.n_pairs), "--concurrency", str(args.concurrency)]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    if args.keep_dropped:
        argv += ["--keep-dropped"]
    old = sys.argv
    sys.argv = argv
    t0 = time.time()
    rc = 0
    try:
        rc = mod.main()
    finally:
        sys.argv = old

    el = time.time() - t0
    b1 = None
    try:
        b1 = balance()
        print("\n[cost] 结束余额:", json.dumps(b1.get("balance_infos", b1), ensure_ascii=False), flush=True)
    except Exception as exc:  # noqa: BLE001
        print("[cost] 余额查询失败:", type(exc).__name__, flush=True)

    spent = None
    if b0 and b1:
        try:
            spent = float(b0["balance_infos"][0]["total_balance"]) - float(b1["balance_infos"][0]["total_balance"])
        except Exception:  # noqa: BLE001
            pass

    u = dict(sorted(USAGE.items()))
    print(f"\n[cost] 用时 {el:.1f}s；用量 {json.dumps(u, ensure_ascii=False)}")
    if spent is not None:
        print(f"[cost] 实测花费 ¥{spent:.4f}")
        n_rec = args.limit
        if n_rec:
            print(f"[cost] 每条 RNA ¥{spent/n_rec:.4f} → 外推 3,800 条 ¥{spent/n_rec*3800:.1f}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "model": args.model, "limit": args.limit, "variants": args.variants,
            "kinds": args.kinds, "n_pairs": args.n_pairs, "concurrency": args.concurrency,
            "elapsed_s": round(el, 1), "usage": u,
            "balance_before": b0, "balance_after": b1, "spent_cny": spent,
            "exit_code": rc,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[cost] -> {args.report}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
