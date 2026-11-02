"""逐 token 模态相关性探针（MAPO 的仪器的 RNA 版）。

MAPO（arXiv 2605.27741）的做法：拿一个 **ablated reference** 和真实策略比，
用跨模态差分熵找出"哪些 token 真的依赖源模态"，然后据此加权策略梯度。

我们的对应物：
    参照   = 编码器置零的前向（encoder-ablated）
    度量   = 答案位置上两个分布的 **KL(真实 || 置零)**
    粒度   = **逐 token**，不是一个全局标量

为什么必须升级到逐 token：
    全局 Δloss 只说"整体上依赖变少了"，说不出**哪里**变少。
    而模态崩溃的关键信息恰恰在分布上——是"所有位置均匀地依赖下降"（= 整体遗忘），
    还是"只有套话位置不依赖、关键事实位置仍然依赖"（= 正常的任务结构）？
    这两种情况的全局标量可能一样，但对下一步该改什么，结论完全相反。
"""
from __future__ import annotations

import argparse
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
from rnaqwen.models.encoder import EncoderOutput  # noqa: E402


@torch.no_grad()
def answer_kl(model, seq: str, tgt: str, device) -> dict:
    """返回该样本答案位置上，真实 RNA 与置零 RNA 两个分布的逐 token KL。"""
    # ⚠️ build_batch 必须在**补丁内部**重新调用。
    #    踩过：如果在补丁之前先算好 inputs，置零那个 forward 用的还是旧 inputs，
    #    补丁根本没参与计算 —— 结果是 KL 全 0.0000（包括 max），看起来像"模态完全崩了"，
    #    实际是我的探针没生效。全零且 max 也是零，这就是信号：先怀疑自己。
    def fwd(zero: bool):
        orig = model.encoder.forward
        if zero:
            def zf(seqs):
                o = orig(seqs)
                return EncoderOutput(hidden=torch.zeros_like(o.hidden),
                                     mask=o.mask, lengths=o.lengths)
            model.encoder.forward = zf
        try:
            inputs, attn, labels = model.build_batch([seq], [tgt], device)
            out = model.llm(inputs_embeds=inputs, attention_mask=attn, labels=labels)
        finally:
            model.encoder.forward = orig
        return out.logits, labels

    lr, labels = fwd(False)
    lz, _ = fwd(True)
    lr, lz = lr.float(), lz.float()

    # 只在答案位置比：labels != -100 且需要 shift 对齐
    shift_lab = labels[:, 1:]
    mask = shift_lab != -100
    if not bool(mask.any()):
        return {}
    pr = F.log_softmax(lr[:, :-1][mask], dim=-1)
    pz = F.log_softmax(lz[:, :-1][mask], dim=-1)
    kl = (pr.exp() * (pr - pz)).sum(-1)                 # 逐 token KL(real||zero)
    idx = shift_lab[mask]
    dlogp = (pr.gather(-1, idx.unsqueeze(-1)) - pz.gather(-1, idx.unsqueeze(-1))).squeeze(-1)
    return {"kl": kl.detach().cpu(), "dlogp": dlogp.detach().cpu(), "n": int(mask.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed/rnaqa_rnacentral_all.jsonl")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/token_modality_relevance.json")
    args = ap.parse_args()

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    known = {f.name for f in dataclasses.fields(Stage1Config)}
    cfg = Stage1Config(**{k: v for k, v in (blob.get("config") or {}).items() if k in known})
    model = Stage1Aligner(cfg)
    model.projection.load_state_dict(blob["projection"])
    if model.compression is not None:
        model.compression.load_state_dict(blob["compression"])
    for m in (model.encoder, model.llm, model.projection):
        m.to(args.device)
    if model.compression is not None:
        model.compression.to(args.device)
    model.eval()
    print(f"[tok] ckpt={args.ckpt.name} step={blob.get('step')} arch={model.encoder.arch.name}")

    recs = [r.as_dict() for r in read_jsonl(args.data)][: args.n]
    all_kl, all_dlogp, per_sample = [], [], []
    for i, r in enumerate(recs):
        seq = r.get("sequence") or ""
        tgt = (r.get("annotation") or "").strip()
        if not seq or not tgt:
            continue
        try:
            d = answer_kl(model, seq, tgt, args.device)
        except Exception as e:
            print(f"  [{i}] 失败 {type(e).__name__}: {str(e)[:60]}")
            continue
        if not d:
            continue
        kl, dl = d["kl"], d["dlogp"]
        all_kl.append(kl); all_dlogp.append(dl)
        per_sample.append({
            "i": i, "n_tok": d["n"],
            "kl_mean": float(kl.mean()), "kl_median": float(kl.median()),
            "kl_max": float(kl.max()), "kl_p90": float(kl.quantile(0.9)),
            "kl_frac_above_0.1": float((kl > 0.1).float().mean()),
            "dlogp_mean": float(dl.mean()),
            "dlogp_neg_frac": float((dl < 0).float().mean()),
        })

    if not all_kl:
        print("[fail] 没有有效样本")
        return 1

    K = torch.cat(all_kl); D = torch.cat(all_dlogp)
    print(f"\n  样本 {len(per_sample)}  答案 token {K.numel()}")
    print(f"  KL(real||zero)  mean {K.mean():.4f}  median {K.median():.4f} "
          f" p90 {K.quantile(0.9):.4f}  max {K.max():.4f}")
    print(f"  依赖占比：KL>0.1 的 token 占 {100*float((K>0.1).float().mean()):.1f}%")
    print(f"                  KL>0.5 的 token 占 {100*float((K>0.5).float().mean()):.1f}%")
    print(f"  Δlogp(正确token)  mean {D.mean():+.4f}   **负值占比 {100*float((D<0).float().mean()):.1f}%**")
    print()
    print("  %4s %7s %9s %9s %9s %11s" % ("idx", "ntok", "KL均值", "KL中位", "KL_p90", "KL>0.1占比"))
    for s in per_sample[:12]:
        print("  %4d %7d %9.4f %9.4f %9.4f %10.1f%%"
              % (s["i"], s["n_tok"], s["kl_mean"], s["kl_median"], s["kl_p90"],
                 100 * s["kl_frac_above_0.1"]))

    print()
    frac = float((K > 0.1).float().mean())
    neg = float((D < 0).float().mean())
    if frac > 0.5:
        print("  -> **大部分答案 token 都依赖 RNA**：模型确实在逐位置读序列。模态未崩。")
    elif frac > 0.1:
        print("  -> **依赖集中在少数 token**：典型结构——套话位置不依赖，事实位置依赖。")
        print("     这不算崩溃；要看的是那几个关键位置**随训练有没有掉**。")
    else:
        print("  -> **几乎没有 token 依赖 RNA**：模态已经崩了。")
    if neg > 0.3:
        print(f"  -> ⚠️ {100*neg:.0f}% 的 token 上，置零后正确 token 的 logp **反而更高**。")
        print("     即编码器在这些位置的贡献是**负的**，不只是没用。")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"ckpt": str(args.ckpt), "step": blob.get("step"),
         "kl_mean": float(K.mean()), "kl_median": float(K.median()),
         "frac_kl_gt_0.1": frac, "dlogp_neg_frac": neg,
         "per_sample": per_sample}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[tok] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
