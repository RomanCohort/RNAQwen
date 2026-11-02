"""用 litsumm 的真实 context/summary 造一个 3 条记录的**假输入**，只为验证管线通不通。

真数据要等 parquet 下完，但 `12_generate_qa.py` / 分片驱动 / jsonl 口径现在就能测。
序列是随便凑的（不参与生成），uid 前缀 `smoke:` 明确标记这不是真数据。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow.parquet as pq  # noqa: E402

from rnaqwen.data.records import RnaQaRecord, write_jsonl  # noqa: E402

OUT = ROOT / "work" / "smoke" / "smoke_input.jsonl"
LIT = ROOT / "data" / "external" / "litsumm_v1_5_data.parquet"


def main() -> int:
    d = pq.read_table(LIT, columns=["urs_taxid", "ent_id", "context", "summary"]).to_pydict()
    picks = []
    for i in range(len(d["urs_taxid"])):
        if len(d["context"][i]) > 400 and len(d["summary"][i]) > 200:
            picks.append(i)
        if len(picks) >= 3:
            break
    recs = []
    for i in picks:
        urs_taxid = d["urs_taxid"][i]
        urs, taxid = urs_taxid.split("_", 1)
        seq = ("ACGU" * 30)[:120]
        recs.append(RnaQaRecord(
            uid=f"smoke:{urs}|taxid:{taxid}",
            sequence=seq,
            organism=f"taxid:{taxid}",
            rna_type="smoke",
            definition=d["ent_id"][i],
            literature=d["context"][i],
            annotation=d["summary"][i],
            provenance={"source": "SMOKE-TEST (非真实数据)", "ent_id": d["ent_id"][i]},
        ))
    n = write_jsonl(OUT, recs)
    print(f"[smoke] 写入 {n} 条 -> {OUT}")
    for r in recs:
        print("  ", r.uid, "| lit", len(r.literature), "| ann", len(r.annotation))
    return 0


if __name__ == "__main__":
    sys.exit(main())
