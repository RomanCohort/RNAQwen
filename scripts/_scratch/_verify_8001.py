"""End-to-end check of the real-scenario chain: 8001 gateway -> 8091 llama-server.

Reads an API key straight out of the gateway's sqlite db, sends one chat
completion, and prints a short digest of the answer plus the timing.
"""
import json
import sqlite3
import time

import requests

DB = r"C:\qwen-api\data\qwen-api.db"

con = sqlite3.connect(DB)
tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print("tables:", tables)
key = None
for t in tables:
    cols = [d[1] for d in con.execute("PRAGMA table_info(%s)" % t)]
    if any(c in ("key", "api_key", "token") for c in cols):
        col = [c for c in ("key", "api_key", "token") if c in cols][0]
        rows = list(con.execute("SELECT %s FROM %s LIMIT 3" % (col, t)))
        print("  %s.%s -> %d rows" % (t, col, len(rows)))
        if rows and key is None:
            key = rows[0][0]
if not key:
    raise SystemExit("no api key found")
print("using key %s...%s" % (key[:6], key[-4:]))

t0 = time.time()
r = requests.post(
    "http://127.0.0.1:8001/v1/chat/completions",
    headers={"Authorization": "Bearer %s" % key, "Content-Type": "application/json"},
    json={
        "model": "qwen3.6-35b-a3b-base",
        "messages": [
            {"role": "system", "content": "You are a helpful RNA biology assistant."},
            {"role": "user", "content": "用两句话解释什么是 riboswitch。"},
        ],
        "max_tokens": 300,
        "temperature": 0.7,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    timeout=300,
)
dt = time.time() - t0
print("HTTP", r.status_code, "in %.2fs" % dt)
js = r.json()
if "choices" in js:
    msg = js["choices"][0]["message"]
    print("reasoning:", repr(msg.get("reasoning_content"))[:200])
    print("content  :", msg.get("content"))
    print("finish   :", js["choices"][0].get("finish_reason"))
    print("usage    :", js.get("usage"))
else:
    print(json.dumps(js, ensure_ascii=False)[:600])
