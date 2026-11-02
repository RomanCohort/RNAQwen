"""只算一件事：投影表示的「公共分量」有多大、去掉它之后还能不能区分不同 RNA。

为什么要单独跑：原探针的 inter=0.94 可能是**公共偏置分量**造成的假象——
所有 RNA 的表示都含一个很大的共同方向，平均之后看起来都像。
必须把跨序列均值减掉，再看剩下的变化量有没有区分力。
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from rnaqwen.data.records import read_jsonl  # noqa: E402
from rnaqwen.models.alignment import Stage1Aligner, Stage1Config  # noqa: E402

CKPT = ROOT / "checkpoints/stage1_ernierna_c64_full_b2/projection_step14500.pt"
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"
N = 32
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    recs = [r.as_dict() for r in read_jsonl(DATA)]
    seqs, seen = [], set()
    for r in recs:
        s = r.get("sequence") or ""
        b = len(s) // 150
        if s and b not in seen:
            seen.add(b)
            seqs.append(s)
        if len(seqs) >= N:
            break
    print(f"[probe] {len(seqs)} 条序列，长度 {min(map(len,seqs))}–{max(map(len,seqs))}")

    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    known = {f.name for f in dataclasses.fields(Stage1Config)}
    cfg = Stage1Config(**{k: v for k, v in (blob.get("config") or {}).items() if k in known})
    m = Stage1Aligner(cfg)
    m.projection.load_state_dict(blob["projection"])
    if m.compression is not None:
        m.compression.load_state_dict(blob["compression"])
    for mod in (m.encoder, m.llm, m.projection):
        mod.to(DEV)
    if m.compression is not None:
        m.compression.to(DEV)
    m.eval()
    print(f"[probe] checkpoint step={blob.get('step')}  arch={m.encoder.arch.name}")

    with torch.no_grad():
        enc = m.encoder(seqs)
        h, mk = enc.hidden.to(DEV), enc.mask.to(DEV)
        if m.compression is not None:
            h, mk = m.compression(h, mk)
        p = m.projection(h).float()                      # (N, L', 4096)
        w = (mk > 0).to(p.dtype).unsqueeze(-1)
        r = (p * w).sum(1) / w.sum(1).clamp_min(1.0)     # (N, 4096) 序列级表示

    # 1) 逐 token 层面：所有 RNA 的所有 token 混在一起，看"同一个 token 槽位"跨序列是否相似
    Nn, L, D = p.shape
    valid = (mk > 0)
    Lmin = int(valid.sum(1).min())
    pt = p[:, :Lmin, :]                                   # (N, Lmin, D) 对齐前 Lmin 个位置
    norm_pt = F.normalize(pt, dim=-1)
    sim_pos = torch.einsum("nld,mld->nm", norm_pt, norm_pt) / Lmin   # (N,N) 逐位置平均
    off = sim_pos[~torch.eye(Nn, dtype=torch.bool, device=sim_pos.device)]
    print(f"\n  逐位置（前 {Lmin} 个位置）跨序列平均相似度 : {float(off.mean()):.4f}")

    # 2) 序列级表示：原始 vs 去均值 vs L2 归一化后
    rn = F.normalize(r, dim=-1)
    s_raw = rn @ rn.t()
    off_raw = float(s_raw[~torch.eye(Nn, dtype=torch.bool, device=r.device)].mean())

    mu = r.mean(0, keepdim=True)
    rc = r - mu
    mu_norm = float(mu.norm())
    r_norm_mean = float(r.norm(dim=-1).mean())
    # 去掉公共分量后，还剩多少能量
    per_seq_energy = rc.norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-9)
    print(f"  序列级表示 平均模长              : {r_norm_mean:.2f}")
    print(f"  跨序列均值向量 模长（公共分量）    : {mu_norm:.2f}   "
          f"占平均模长 {100*mu_norm/r_norm_mean:.1f}%")
    print(f"  去掉公共分量后 保留能量            : "
          f"{100*float(per_seq_energy.mean()):.1f}%")

    rcn = F.normalize(rc, dim=-1)
    s_c = rcn @ rcn.t()
    off_c = float(s_c[~torch.eye(Nn, dtype=torch.bool, device=r.device)].mean())
    print(f"\n  序列级 inter（原始）              : {off_raw:.4f}")
    print(f"  序列级 inter（去公共分量后）       : {off_c:.4f}")

    print()
    if off_c < 0.5:
        print("  -> 去掉公共分量后**区分得开**：信息在，只是被一个大偏置盖住了。")
        print("     那问题不是'投影没送信息'，而是'LLM 要先学会减掉那个偏置'。")
    else:
        print("  -> 去掉公共分量后**仍然分不开**：RNA 特异的信息真的没送过去。")
        print("     投影是瓶颈，DeCo 那条路线有直接证据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
