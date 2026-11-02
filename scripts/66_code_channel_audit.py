"""路线 C 缺失的那一次验证：e2e 训练之后，码表还是不是那个「可复用词表」？

为什么必须做
------------
判据②（跨序列 6-mer 迁移）**只在 stage-0 的冻结 tokenizer 上量过**：2.63x
（48_stage1_full_tokenizer.py，逐核苷酸一个码）。

但 e2e 实际接进 LLM 的是**另一个** tokenizer（50_routeC_e2e.py 的 FSQCode）：

  · 前面多了一个 pool=4 的学习式压缩（初始化为平均池化）—— 48_ 是逐核苷酸的；
  · enc 从 48_ 的检查点载入后**继续被 LLM 的 CE 梯度训练**（日志 |grad_fsq| 0.03~0.7）。

后果已经看得见：用码 4092/4096(99.9%) -> 评测集上只剩 ~350/4096，熵 5.86/8.32。
而**判据②对这版 tokenizer 从来没测过**。它是「码的身份值不值得学」的唯一非循环证据 ——
掉下去的话，准确率不显著就不再是「训得不够」，而是「码本身没东西可学」。

三路对照（同一批隐藏态、同一套 6-mer 靶子、同一个划分）
-------------------------------------------------------
  A. stage-0 冻结 tokenizer，逐核苷酸     <- 2.63x 就是它（本脚本必须先复现出来，否则尺子不可信）
  B. 同一个冻结 enc + 平均池化 4x          <- 隔离「池化」的代价（从没单独量过）
  C. e2e 训练后的 FSQ（pool=4）            <- 实际接进 LLM 的那一版

另附两个诊断
------------
  · 熵正则作用在**软**使用率上（tau 决定的 softmax 分布本来就散），塌缩发生在**硬**码上。
    两者差多少 = 「复用压力」量错对象的程度。
  · 训练后码 vs 冻结码在窗口首位的**一致率** = 码级漂移的直接读数；
    这也是 --init-code-emb（按 stage-0 码建的）还有没有意义的前提。

跑法（不占显存，可与训练并行）:
    python scripts/66_code_channel_audit.py \
        --ckpt runs/e2e_A_v3/ckpt/step_016000.pt \
        --out reports/code_channel_audit.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tok48 = _load("t48", ROOT / "scripts/48_stage1_full_tokenizer.py")
rc = _load("rc50", ROOT / "scripts/50_routeC_e2e.py")


def digits_to_code(k: torch.Tensor, levels: int) -> torch.Tensor:
    """与 48_/50_ 里那个循环逐位同构：code = code*levels + k[:,d]（高位在前）。"""
    c = torch.zeros(k.shape[0], dtype=torch.long, device=k.device)
    for d in range(k.shape[1]):
        c = c * levels + k[:, d].long()
    return c


def packed_entropy(codes: torch.Tensor, vocab: int) -> dict:
    cnt = torch.bincount(codes.long(), minlength=vocab)
    used = int((cnt > 0).sum())
    p = cnt.float() / max(float(cnt.sum()), 1.0)
    ent = float(-(p * (p + 1e-9).log()).sum())
    return dict(n_tokens=int(codes.numel()), used=used, used_pct=used / vocab,
                entropy=ent, entropy_max=math.log(vocab), top_share=float(p.max()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=ROOT / "runs/e2e_A_v3/ckpt/step_016000.pt")
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--hold", type=float, default=0.2, help="跨序列留出比例（与 48_ 一致）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chunk", type=int, default=200_000, help="逐核苷酸编码的分块大小")
    ap.add_argument("--soft-n", type=int, default=60_000, help="算软熵用多少个位置")
    ap.add_argument("--out", type=Path, default=ROOT / "reports/code_channel_audit.json")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("路线 C 码信道审计：e2e 之后码表还是词表吗")
    print("=" * 78)

    # ---------------------------------------------------------------- 数据
    recs = tok48.load_all()
    seqs = [r["sequence"] for r in recs]
    print(f"[data] {len(recs)} 条 RNA（与 stage-0 同一批）")

    try:
        d = torch.load(tok48.POS_CACHE, map_location="cpu", weights_only=False, mmap=True)
        print(f"[cache] mmap 载入 {tok48.POS_CACHE.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"[cache] mmap 不可用（{type(exc).__name__}），整块载入")
        d = torch.load(tok48.POS_CACHE, map_location="cpu", weights_only=False)
    x, lens, offsets = d["x"], d["lens"], d["offsets"]
    N, DIM = int(x.shape[0]), int(x.shape[1])
    assert int(lens.sum()) == N, f"缓存自检失败 {int(lens.sum())} != {N}"
    assert d.get("n") == len(recs), f"缓存覆盖 {d.get('n')} 条 != {len(recs)} 条"
    print(f"[pos] {N:,} 个位置  dim {DIM}  中位长 {int(lens.median())}")

    spans, o = [], 0
    for L in lens.tolist():                 # 每条序列的位置区间（顺序与 48_ 的 load_all 一致）
        spans.append((o, L))
        o += L
    span_start = torch.tensor([s[0] for s in spans], dtype=torch.long)

    # ---------------------------------------------------------------- 两个 head
    tc = torch.load(ROOT / "checkpoints/fsq_full/tokenizer.pt", map_location="cpu",
                    weights_only=False)
    LAT, LEV = int(tc["latent"]), int(tc["levels"])
    VOC = LEV ** LAT
    frozen = tok48.FSQTokenizer(DIM, LAT, LEV, int(tc.get("hidden", 256)))
    frozen.load_state_dict(tc["state"])
    frozen.eval()
    print(f"[head-A/B] stage-0 冻结 tokenizer  latent {LAT} levels {LEV} vocab {VOC}")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    trained = rc.FSQCode(DIM, LAT, LEV, DIM, pool=args.pool)
    cur = trained.state_dict()
    sd = {k: v for k, v in ck["fsq"].items()
          if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
    missing = [k for k in cur if k not in sd]
    trained.load_state_dict(sd, strict=False)
    trained.eval()
    print(f"[head-C] {args.ckpt.name}  step {ck['step']}  pool {args.pool}  "
          f"载入 {len(sd)}/{len(cur)} 个张量" + (f"  **缺 {missing}**" if missing else ""))
    if missing:
        raise SystemExit("训练后的 FSQ 权重不完整，读数不可信，先查检查点")

    d_enc = trained.enc.weight.detach() - frozen.enc.weight.detach()
    drift_enc = float(d_enc.norm() / frozen.enc.weight.detach().norm())
    avgpool = torch.eye(DIM).repeat(1, args.pool) / args.pool      # 与 FSQCode 的初始化同构
    d_cmp = trained.compress.weight.detach().float() - avgpool
    print(f"[drift] enc 相对漂移 {100*drift_enc:.1f}%   compress 距平均池化 "
          f"{float(d_cmp.norm()):.3f} / 平均池化自身范数 {float(avgpool.norm()):.3f}")

    # ---------------------------------------------------------------- A: 逐核苷酸（冻结）
    @torch.no_grad()
    def codes_frozen_pernt() -> torch.Tensor:
        out = torch.empty(N, dtype=torch.int32)
        for i in range(0, N, args.chunk):
            xb = x[i:i + args.chunk].float()
            _, k = tok48.fsq(frozen.enc(xb), LEV)
            out[i:i + args.chunk] = digits_to_code(k, LEV).to(torch.int32)
        return out

    t1 = time.time()
    codes_A = codes_frozen_pernt()
    print(f"[A] 逐核苷酸编码完 {time.time()-t1:.0f}s")

    # ---------------------------------------------------------------- B/C: 池化
    @torch.no_grad()
    def codes_pooled(head: str):
        """(窗口码, 窗口所属序列号, 窗口在序列内的起始位置)。逐条序列做，池化不跨序列。"""
        cs, sids, pstr = [], [], []
        for si, (o, L) in enumerate(spans):
            n = L // args.pool
            if n == 0:
                continue
            xs = x[o:o + n * args.pool].float().reshape(n * args.pool, DIM)
            if head == "B":
                z = xs.reshape(n, args.pool, DIM).mean(dim=1)
                _, k = tok48.fsq(frozen.enc(z), LEV)
                c = digits_to_code(k, LEV)
            else:
                c, _, _, _ = trained(xs)        # 50_ 的实现（含被训练过的 compress）
            cs.append(c.to(torch.int32))
            sids.append(torch.full((n,), si, dtype=torch.long))
            pstr.append(torch.arange(n, dtype=torch.long) * args.pool)
        return torch.cat(cs), torch.cat(sids), torch.cat(pstr)

    pooled = {}
    for name in ("B", "C"):
        t1 = time.time()
        pooled[name] = codes_pooled(name)
        c = pooled[name][0]
        print(f"[{name}] 池化 {args.pool}x 编码完 {c.numel():,} 个窗口  {time.time()-t1:.0f}s")

    # ---------------------------------------------------------------- 6-mer 靶子
    mers = tok48.mer_of(seqs)                                        # 逐核苷酸，末尾 K-1 个为 -1
    seq_id_nt = torch.repeat_interleave(torch.arange(len(lens)), lens)
    assert mers.numel() == N

    def anchor_mers(pstr, sids) -> torch.Tensor:
        """窗口的靶子 = 窗口首位那个核苷酸的局部 6-mer（与 48_ 同源，不循环）。"""
        return mers[span_start[sids] + pstr]

    def run_transfer(codes, sids, pstr, tag) -> dict:
        t1 = time.time()
        r = tok48.transfer_metric(codes.long(), anchor_mers(pstr, sids), sids, len(spans), VOC,
                                  hold=args.hold, seed=args.seed)
        r["seconds"] = round(time.time() - t1, 1)
        r["codes_used"] = int(torch.bincount(codes.long(), minlength=VOC).gt(0).sum())
        print(f"  [{tag}] 迁移 {100*r['acc']:.2f}%  vs 打乱 {100*r['ctrl']:.2f}%  "
              f"vs 常数 {100*r['base']:.2f}%  ->  **{r['lift']:.2f}x**  "
              f"用码 {r['codes_used']}/{VOC}  ({r['seconds']:.0f}s)")
        return r

    print("\n=== 跨序列 6-mer 迁移（唯一非循环的判据，46_/48_ 同一口径）===")
    tr_nt = tok48.transfer_metric(codes_A.long(), mers, seq_id_nt, len(spans), VOC,
                                  hold=args.hold, seed=args.seed)
    print(f"  [A 冻结·逐核苷酸] 迁移 {100*tr_nt['acc']:.2f}%  vs 打乱 {100*tr_nt['ctrl']:.2f}%  "
          f"vs 常数 {100*tr_nt['base']:.2f}%  ->  **{tr_nt['lift']:.2f}x**   "
          f"(48_ 报 2.63x —— 本行是尺子自检)")
    tr_B = run_transfer(pooled["B"][0], pooled["B"][1], pooled["B"][2], f"B 冻结+池化{args.pool}x")
    tr_C = run_transfer(pooled["C"][0], pooled["C"][1], pooled["C"][2], f"C e2e训练后·池化{args.pool}x")

    # ---------------------------------------------------------------- 码级漂移
    codes_C, sids_C, pstr_C = pooled["C"]
    same = codes_C == codes_A[span_start[sids_C] + pstr_C]
    agree = float(same.float().mean())
    print(f"\n[漂移] 窗口首位：训练后码 == 冻结码 的比例 {100*agree:.1f}%  "
          f"=> --init-code-emb 的前提{'还在' if agree > 0.5 else '**已经不成立**'}")

    # ---------------------------------------------------------------- 软/硬熵
    print("\n=== 熵正则量的是软使用率，塌缩发生在硬码上 ===")
    soft_ent, soft_used = None, None
    tau = LAT / 2
    with torch.no_grad():
        parts, taken = [], 0
        for (o, L) in spans:
            if taken >= args.soft_n:
                break
            n = L // args.pool
            if n == 0:
                continue
            xs = x[o:o + n * args.pool].float().reshape(n * args.pool, DIM)
            _, q_ste, _, _ = trained(xs)
            for s in range(0, q_ste.shape[0], 4096):
                q = q_ste[s:s + 4096]
                d2 = (q.unsqueeze(1) - trained.grid.unsqueeze(0)).pow(2).sum(-1)
                parts.append(torch.softmax(-d2 / tau, dim=-1).mean(0))
            taken += n
        u = torch.stack(parts).mean(0)
        p = u / u.sum().clamp(min=1e-9)
        soft_ent = float(-(p * (p + 1e-9).log()).sum())
        soft_used = int((u > 1e-6).sum())
    hard = packed_entropy(codes_C, VOC)
    print(f"  软熵 {soft_ent:.2f}/{math.log(VOC):.2f}（{taken} 个位置，tau {tau}）"
          f"   软分布里 >1e-6 的码 {soft_used}/{VOC}")
    print(f"  硬码 熵 {hard['entropy']:.2f}/{hard['entropy_max']:.2f}  "
          f"用码 {hard['used']}/{VOC}  最大码占比 {100*hard['top_share']:.2f}%")
    print(f"  => 熵正则在软分布上看到的覆盖是硬码的 {soft_used/max(hard['used'],1):.1f} 倍")

    # ---------------------------------------------------------------- 汇总
    stats = {
        "ckpt": str(args.ckpt), "step": int(ck["step"]), "pool": args.pool,
        "A_frozen_pernt": dict(**packed_entropy(codes_A, VOC), transfer=tr_nt),
        "B_frozen_pooled": dict(**packed_entropy(pooled["B"][0], VOC), transfer=tr_B),
        "C_trained_pooled": dict(**hard, transfer=tr_C,
                                 soft_entropy=soft_ent, soft_used=soft_used),
        "code_agreement_windowstart": agree,
        "enc_relative_drift": drift_enc,
        "compress_dist_from_avgpool": float(d_cmp.norm()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {args.out}   用时 {(time.time()-t0)/60:.1f} min")

    # ---------------------------------------------------------------- 判读
    print("\n" + "=" * 78)
    print("判读")
    print("=" * 78)
    lc, lb, la = tr_C["lift"], tr_B["lift"], tr_nt["lift"]
    if lc >= 2.0:
        print(f"  ✅ 训练后码表仍是词表（{lc:.2f}x）。准确率不显著不能怪 tokenizer ——")
        print("     码里有可迁移的东西，问题在「LLM 会不会读」。下一步该动对齐，不是重训 tokenizer。")
    elif lc >= 1.5:
        print(f"  ⚠️ 训练后码表**变钝**：{lc:.2f}x（冻结逐核苷酸 {la:.2f}x / 冻结+池化 {lb:.2f}x）。")
        print("     池化吃掉多少看 B-A，e2e 训练吃掉多少看 C-B。")
    else:
        print(f"  ❌ 训练后码表**不再是词表**：{lc:.2f}x（冻结逐核苷酸 {la:.2f}x / 冻结+池化 {lb:.2f}x）。")
        print("     CE 梯度把码改成了「每条序列一串各自不同的片段」——")
        print("     判据②对实际接进 LLM 的这版 tokenizer 从来没成立过。")
        print("     修法：冻结 FSQ 头（只训码表 embedding），或把重建/迁移目标一起加进损失。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
