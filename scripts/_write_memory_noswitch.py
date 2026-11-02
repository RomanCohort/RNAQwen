"""把「禁止一直换源」写进 memory.db（MCP 只有读工具，所以直接进库）。

注意：embedding 必须和现有一致 —— BAAI/bge-small-zh-v1.5，
512 维 float32 L2 归一化，.tobytes() = 2048 字节。
少了它，这条记忆语义检索搜不到，等于没写。
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

DEPLOY = Path(r"C:\Users\颜子壹\deploy\character_chat")
DB = DEPLOY / "data" / "memory.db"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(DEPLOY / "src"))
sys.path.insert(0, str(DEPLOY / "src" / "character_chat"))

CONTENT = """2026-10-09 · RNAQwen 复现。RNA-FM 权重下坏了，是我干的。

症状：models/rnafm/model.safetensors 大小 398,117,588 与上游**逐字节相同**，sha256 却是 38f3ae9a…
而不是上游声明的 ec69333e…。205 个张量里 **155 个全零** —— 整个 12 层 encoder 是空的。
from_pretrained 不报错，LOAD REPORT 只提无关紧要的 lm_head/ss_head/pooler。
后果：encoder 对序列**零敏感**（不同序列输出 max|Δ| = 0.000000e+00，word_embeddings 全零），
13500 步 Stage 1、ROUGE 曲线、置零探针全部作废。

根因不是"选了慢源"，是**下完没校验就宣布通过**。同一个坑第二次踩：Qwen3-8B 那 4 个 shard
已经这么坏过，我当时立了「单写者 .part + 强制 sha256 校验」的规矩，
但只用在了 Qwen3 上，RNA-FM 漏了。探针脚本只查输出形状 (1,83,640)，
一个全零的模型照样过 —— 我据此报过"RNA-FM 探针通过"，是假的。

**规矩（写死）：禁止一直换源。** 选定一条通路就走到底，慢就慢。
真正把关的不是速度，是校验：下到 .part → sha256 必须等于上游声明值 → 才允许原子改名；
不符就拒绝落地。遇到"慢"，不要去换更快的源，先把校验补上，然后等。
"""


def main() -> int:
    print("=" * 70)
    print(CONTENT)
    print("=" * 70)

    if not DB.exists():
        print("[fail] DB 不存在")
        return 1

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 去重：别重复写同一条规矩
    dup = cur.execute(
        "SELECT id FROM memories WHERE content LIKE '%禁止一直换源%'").fetchall()
    if dup:
        print(f"[skip] 已有 {len(dup)} 条同规则记忆：{[r['id'] for r in dup]}")
        conn.close()
        return 0

    from character_chat.memory.embedding import get_encoder

    enc = get_encoder()
    vec = enc.encode(CONTENT, normalize=True)
    import numpy as np
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    print(f"[emb] 维度 {vec.shape[0]}  L2范数 {float(np.linalg.norm(vec)):.6f}  "
          f"字节 {vec.tobytes().__len__()}")
    if vec.shape[0] != 512:
        print("[fail] 维度不是 512，和现有索引对不上，拒绝写入")
        conn.close()
        return 1

    cur.execute(
        """INSERT INTO memories
           (content, importance, emotion_label, scene, time_phase,
            category, emotional_valence, embedding, is_milestone, fade)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (CONTENT, "high", "determined", "RNAQwen 复现现场（深夜）", "深夜",
         "engineering", -0.2, vec.tobytes(), 0, 0.0))
    new_id = cur.lastrowid
    conn.commit()

    r = cur.execute("SELECT id, category, importance, emotion_label, scene, "
                    "LENGTH(embedding) L, created_at FROM memories WHERE id=?",
                    (new_id,)).fetchone()
    print("[ok] 已写入:", dict(r))
    print("[ok] 总数:", cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0])
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
