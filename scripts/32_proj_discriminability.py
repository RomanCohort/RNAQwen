"""投影可分辨性探针 —— 回答"投影到底有没有把 RNA 的差异送过去"。

为什么需要这个而不是看 val_loss / proj_std：
  * val_loss 是"最终答案对不对",投影只是链路中的一环，它坏了也可能靠语言先验补上
  * proj_std 只说明输出**幅度**在变，不说明**可分辨性**
  * Δzeroed 衰减说明"用得少了"，但没说"送过去的东西还分不分得开"

测什么：
  对 N 条**不同**的 RNA，算投影后的表示，
    inter = 不同 RNA 之间 表示的平均相似度   <- 越低越好（说明区分得开）
    intra = 同一条 RNA 内部 不同位置的相似度 <- 作为量纲参照
  关键指标 gap = intra - inter。gap 掉到 0 = 投影把所有 RNA 映射到同一个点。

同时沿 checkpoint 扫一遍，看**可分辨性随训练怎么演化** ——
如果早期分得开、后期反而糊了，那就是"训练在把信息抹掉"的直接证据。
"""
from __future__ import annotations

import argparse
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


@torch.no_grad()
def rep_of(model, sequences, device):
    """返回每条序列投影后的表示，按 mask 做平均 -> (N, D)。"""
    enc = model.encoder(sequences)
    rna = enc.hidden.to(device)
    m = enc.mask.to(device)
    if model.compression is not None:
        rna, m = model.compression(rna, m)
    proj = model.projection(rna)                      # (N, L', D)
    w = (m > 0).to(proj.dtype).unsqueeze(-1)
    return (proj * w).sum(1) / w.sum(1).clamp_min(1.0)


def probe_one(ckpt: Path, sequences: list[str], device: str) -> dict:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    import dataclasses
    known = {f.name for f in dataclasses.fields(Stage1Config)}
    saved = {k: v for k, v in (blob.get("config") or {}).items() if k in known}
    model = Stage1Aligner(Stage1Config(**saved))
    model.projection.load_state_dict(blob["projection"])
    if model.compression is not None:
        if "compression" not in blob:
            raise SystemExit(f"{ckpt} 配置要压缩器但 checkpoint 里没有")
        model.compression.load_state_dict(blob["compression"])
    model.encoder.to(device); model.llm.to(device); model.projection.to(device)
    if model.compression is not None:
        model.compression.to(device)
    model.eval()

    r = rep_of(model, sequences, device).float()
    rn = F.normalize(r, dim=-1)

    # inter：不同序列之间（对角之外）
    sim = rn @ rn.t()
    n = sim.shape[0]
    off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
    inter = float(off.mean())

    # ⚠️ 原始 inter 会被**公共偏置分量**污染：如果所有 RNA 的表示都含一个很大的共同方向，
    #    平均后大家看起来就都很像。所以必须再做一次**去均值（中心化）**版本，
    #    单独看「RNA 之间不同的那部分」还有多少能量、还能不能区分。
    rc = r - r.mean(0, keepdim=True)
    energy = float(rc.norm(dim=-1).mean() / r.norm(dim=-1).mean().clamp_min(1e-9))
    if float(rc.norm()) > 1e-6:
        rcn = F.normalize(rc, dim=-1)
        simc = rcn @ rcn.t()
        offc = simc[~torch.eye(n, dtype=torch.bool, device=simc.device)]
        inter_c = float(offc.mean())
    else:
        inter_c = float("nan")

    # intra：同一条序列内部，逐位置表示的相似度
    enc = model.encoder(sequences)
    h = enc.hidden.to(device)
    mk = enc.mask.to(device)
    if model.compression is not None:
        h, mk = model.compression(h, mk)
    p = model.projection(h).float()
    intra_vals = []
    for i in range(p.shape[0]):
        v = p[i][mk[i] > 0]
        if v.shape[0] < 4:
            continue
        vn = F.normalize(v, dim=-1)
        s = vn @ vn.t()
        m = ~torch.eye(s.shape[0], dtype=torch.bool, device=s.device)
        intra_vals.append(float(s[m].mean()))
    intra = sum(intra_vals) / len(intra_vals) if intra_vals else float("nan")

    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    return {"ckpt": ckpt.name, "step": int(blob.get("step", 0)),
            "inter": inter, "intra": intra, "gap": intra - inter,
            "n": n, "proj_std": float(r.std())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/rnaqa_n6_merged.jsonl")
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--every", type=int, default=2, help="每 N 个 checkpoint 测一个")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/proj_discriminability.json")
    args = ap.parse_args()

    recs = [r.as_dict() for r in read_jsonl(args.data)]
    seqs = []
    seen_len = set()
    for r in recs:
        s = r.get("sequence") or ""
        # 故意按长度分层取，避免全抽到同一类长度
        b = len(s) // 200
        if s and b not in seen_len:
            seen_len.add(b)
            seqs.append(s)
        if len(seqs) >= args.n_seq:
            break
    if len(seqs) < 4:
        seqs = [r["sequence"] for r in recs[:args.n_seq]]
    lens = sorted(len(s) for s in seqs)
    print(f"[probe] {len(seqs)} 条序列，长度 min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")

    cks = sorted(args.ckpt_dir.glob("projection_step*.pt"),
                 key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    cks = cks[::args.every]
    if not cks:
        print("[fail] 没有 checkpoint")
        return 1
    print(f"[probe] 测 {len(cks)} 个 checkpoint\n")

    rows = []
    print("  %8s %10s %10s %10s %10s" % ("step", "inter", "intra", "gap", "rep_std"))
    for c in cks:
        try:
            d = probe_one(c, seqs, args.device)
        except Exception as e:
            print("  %8s  失败 %s: %s" % (c.name, type(e).__name__, str(e)[:50]))
            continue
        rows.append(d)
        print("  %8d %10.4f %10.4f %10.4f %10.5f"
              % (d["step"], d["inter"], d["intra"], d["gap"], d["proj_std"]))

    if rows:
        g0, g1 = rows[0]["gap"], rows[-1]["gap"]
        print()
        print(f"  gap 首 {g0:.4f} -> 末 {g1:.4f}   （{100*g1/max(g0,1e-9):.0f}%）")
        if g1 < 0.05:
            print("  -> **投影基本无分辨力**：不同 RNA 的表示几乎重合。投影是瓶颈。")
        elif g1 < 0.5 * g0:
            print("  -> 分辨力在**显著衰减**，训练在把差异抹掉。支持「冗余抽象丢信息」。")
        else:
            print("  -> 分辨力保持住了，投影没塌。瓶颈可能在别处。")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[probe] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
