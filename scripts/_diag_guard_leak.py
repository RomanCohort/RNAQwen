"""诊断：闸门报的 53 处「序列泄漏」到底是真泄漏还是假阳性。

怀疑点：`find_sequence_leak` 先用 `normalize_sequence(text)` 把文本里**所有非 ACGU 字符删掉**，
再在剩下的一串字母里找 k-mer。问题是 **英文散文里充满了 a/c/g/u 这四个字母** ——
"can act as a competing" 过滤完就是 "cacacacgc..."，天然会撞上低复杂度序列。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

from rnaqwen.data.guard import normalize_sequence  # noqa: E402

path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/processed/rnaqa_rnacentral.jsonl"
report = json.loads((ROOT / "reports/guard_rnacentral.json").read_text(encoding="utf-8"))
leaks = report.get("leaks", [])
print(f"报告里记了 {len(leaks)} 处泄漏\n")

rows = {json.loads(l)["uid"]: json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()}

# 只看前 5 条，把"命中的 k-mer"在答案文本里的**原始位置**找出来
for lk in leaks[:5]:
    uid = lk["uid"]
    rec = rows.get(uid)
    if not rec:
        continue
    hit = (lk.get("exact") or [None])[0]
    answer = lk.get("a", "")
    print("=" * 78)
    print(f"{uid}  seq_len={len(rec['sequence'])}")
    print(f"命中的 k-mer: {hit}")
    print(f"答案片段: {answer[:150]}")
    # 在归一化文本里定位，然后回映射到原文
    norm = normalize_sequence(answer)
    pos = norm.find(hit) if hit else -1
    print(f"归一化文本长度 {len(norm)}（原文 {len(answer)} 字符）")
    if pos >= 0:
        print(f"命中位置 {pos}，归一化上下文: ...{norm[max(0,pos-30):pos+40]}...")
    # 真正的判据：这段文本里有没有**独立成词**的核苷酸串
    runs = re.findall(r"(?<![A-Za-z0-9])[ACGTUacgtu]{8,}(?![A-Za-z0-9])", answer)
    print(f"独立成词的核苷酸串(>=8): {runs[:5] if runs else '无'}")
    # 反向：序列里有没有这个 k-mer（当然有，因为是从序列里取的）
    print(f"该 k-mer 在序列中的位置: {rec['sequence'].find(hit)}")
