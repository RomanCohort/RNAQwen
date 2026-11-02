"""DeepSeek 账户探针：列出可用模型 / 查余额。**只打印掩码，不打印密钥。**

    python scripts/_ds_usage.py --list-models
    python scripts/_ds_usage.py --balance
    python scripts/_ds_usage.py --both
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen import secrets  # noqa: E402

API = "https://api.deepseek.com"


def _get(path: str) -> dict:
    key = secrets.resolve_api_key(provider="deepseek")
    req = urllib.request.Request(API + path, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--both", action="store_true")
    args = ap.parse_args()
    if not (args.list_models or args.balance or args.both):
        args.both = True

    print("key:", secrets.mask(secrets.resolve_api_key(provider="deepseek")))
    if args.list_models or args.both:
        try:
            m = _get("/models")
            print("models:", [d.get("id") for d in m.get("data", [])])
        except Exception as exc:  # noqa: BLE001
            print("models: 查询失败", type(exc).__name__, str(exc)[:160])
    if args.balance or args.both:
        try:
            b = _get("/user/balance")
            print("balance:", json.dumps(b, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            print("balance: 查询失败", type(exc).__name__, str(exc)[:160])
    return 0


if __name__ == "__main__":
    sys.exit(main())
