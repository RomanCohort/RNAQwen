"""看一眼 memory.db 的 schema、embedding 格式、以及 engineering 类记忆的写法。"""
import sqlite3, json
from pathlib import Path

DB = Path(r"C:\Users\颜子壹\deploy\character_chat\data\memory.db")
print("DB:", DB, DB.exists(), "%.1f MB" % (DB.stat().st_size / 1e6))

conn = sqlite3.connect(str(DB))
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("\n=== 表 ===")
for r in cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"):
    print(" ", r["name"])
    print("   ", (r["sql"] or "")[:900].replace("\n", "\n    "))

print("\n=== memories 列 ===")
cols = [c[1] for c in cur.execute("PRAGMA table_info(memories)")]
print(" ", cols)

print("\n=== 计数 ===")
print("  总数:", cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
for r in cur.execute("SELECT category, COUNT(*) c FROM memories GROUP BY category"):
    print("   category=%-14s %d" % (r["category"], r["c"]))
for r in cur.execute("SELECT is_milestone, COUNT(*) c FROM memories GROUP BY is_milestone"):
    print("   milestone=%-8s %d" % (r["is_milestone"], r["c"]))

print("\n=== embedding 格式 ===")
r = cur.execute("SELECT id, embedding, LENGTH(embedding) L FROM memories WHERE embedding IS NOT NULL LIMIT 1").fetchone()
if r:
    print("   sqlite type:", type(r["embedding"]).__name__, " 长度:", r["L"], "字节 (=512 float32 应=2048)")

print("\n=== 最近 5 条 engineering（看写法）===")
for r in cur.execute("""SELECT id, content, importance, emotion_label, scene, time_phase,
                               created_at, category, emotional_valence, is_milestone
                        FROM memories WHERE category='engineering'
                        ORDER BY id DESC LIMIT 5"""):
    d = {k: r[k] for k in r.keys()}
    print(json.dumps(d, ensure_ascii=False)[:600])
    print("  ---")

print("\n=== 内容里提到 换源/下载/校验/哈希 的 ===")
for r in cur.execute("""SELECT id, category, importance, substr(content,1,160) c
                        FROM memories
                        WHERE content LIKE '%换源%' OR content LIKE '%哈希%'
                           OR content LIKE '%sha256%' OR content LIKE '%校验%'
                           OR content LIKE '%下载%'
                        ORDER BY id DESC LIMIT 15"""):
    print("  #%s [%s] imp=%s  %s" % (r["id"], r["category"], r["importance"], r["c"]))

print("\n=== 最大 id ===")
print(" ", cur.execute("SELECT MAX(id) FROM memories").fetchone()[0])
conn.close()
