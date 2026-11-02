"""Export the RNAQwen-preview pack: nf4 experts + bf16 rest + SOS adapter.

What it produces
----------------
    runs/RNAQwen-preview/
      weights/model-000NN-of-000MM.safetensors   nf4 experts + bf16 rest, ~23 GB
      weights/index.json                         key -> shard, sha256 per shard
      config.json           VLM config used to rebuild the meta model
      meta.json             version stamp, provenance, key counts
      sos.json              how to rebuild the SOS layers (absent if merged)
      sos_adapter.pt        the 2480 adapter tensors load_pack re-attaches
      merge_drift.json      what merging *would* cost -- measured, not guessed
      fsq.pt                FSQ encoder (offline code precompute)
      code_emb.pt           code table 4096 x 2048
      tokenizer.pt          RNA tokenizer (<RNA> placeholder)
      load_preview.py       loader
      README.md

Why the adapter is not merged by default (measured 2026-10-15, D_qwen36_sos)
--------------------------------------------------------------------------
Folding dw = sum_k s_k B_k A_k into the 310 bf16 attention / shared_expert
weights moves the probe logits by max|dlogit| 2.32 (9.05e-2 of the max logit)
and flips top-1 on 4 of 128 probe positions.  That is neither run-to-run noise
(two identical forwards differ by exactly 0.0) nor a storage problem
(per-element std(dw) is 0.0008..0.0066, three orders above a bf16 step): bf16
keeps 7 mantissa bits, so bf16(W + dw) is a *different function*, not a
rounding of the same one.  The pack therefore stores the pre-merge weights plus
the adapter and load_pack rebuilds the SOS layers -- bitwise identical to what
was trained and evaluated.  `--merge` restores the lossy merged form.

Checks that must pass before we ship
------------------------------------
1. adapter coverage: the fresh model's trainable set must equal the checkpoint's
   key set.  (76_ lesson: unexpected_keys alone silently accepts a mismatch.)
2. merge safety: every SOS target must be a plain nn.Linear.  On the 35B path
   only the MoE experts are bnb-quantized; merging into a Linear4bit would
   corrupt it silently.
3. pack round trip: drop the in-memory model, reload the pack (weights +
   adapter) into a fresh meta model, logits must come back bitwise identical.
4. --merge additionally demands the drift stay under max(1e-2, 3x the measured
   run-to-run floor).

Usage
    python scripts/97_export_preview.py --ckpt runs/e2e_D_qwen36_sos/ckpt/final.pt
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import importlib.util  # noqa: E402

import torch  # noqa: E402

import rnaqwen  # noqa: F401,E402  必须先于 transformers/peft

spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(spec)
sys.modules["rc"] = rc
spec.loader.exec_module(rc)

import bitsandbytes as bnb  # noqa: E402
from rnaqwen.models.sos_lora import SOSLoRALayer, merge_sos_layers  # noqa: E402
from rnaqwen.preview import load_pack, save_pack  # noqa: E402
from rnaqwen.qwen36 import (base_spec_from_ckpt_args, commit_gb,  # noqa: E402
                            lora_targets)

DEV = rc.DEV


def _parts(llm):
    body = llm.model
    if hasattr(body, "language_model"):
        return body.language_model, llm.lm_head
    return body, llm.lm_head


def _logits(llm, ids, am):
    body, head = _parts(llm)
    emb = body.embed_tokens(ids)
    hs = body(inputs_embeds=emb, attention_mask=am, use_cache=False).last_hidden_state
    return head(hs).float().cpu()


def _cmp(a, b):
    """(max|d|, mean|d|, max relative to |a|.max, top-1 agreement)."""
    d = (a - b).abs()
    mx = d.max().item()
    ag = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    return mx, d.mean().item(), mx / a.abs().max().item(), ag


def write_readme(out: Path, meta: dict) -> None:
    tot = meta["weights"]["total_bytes"] / 2 ** 30
    if meta["adapter"]["merged"]:
        head = ("专家按 nf4 量化，SOS 适配器**已合并**进 bf16 的注意力 / "
                "shared_expert 权重（合并会改变模型行为，见 merge_drift.json）。")
        adapt = ("| `sos_adapter.pt` | 未合并的原始适配器，只作存档，"
                 "不要在 `weights/` 上再挂一次 |")
    else:
        head = ("专家按 nf4 量化，SOS 适配器**不合并**：`weights/` 是合并前的权重，"
                "`load_pack()` 按 `sos.json` 重建 SOS 层并加载 `sos_adapter.pt`，"
                "与训练 / 评测时**逐位一致**。")
        adapt = ("| `sos.json` + `sos_adapter.pt` | SOS 重建配置与适配器张量，"
                 "`load_pack()` 自动使用 |\n"
                 "| `merge_drift.json` | 实测的合并代价（合并前后 logits / top-1 一致率）|")
    (out / "README.md").write_text(f"""# {meta['name']} (preview)

RNA-QA 预览模型包。底座 {meta['base']['dir']}，{head}全包 {tot:.1f} GiB。

## 怎么用

```python
from rnaqwen.preview import load_pack
model = load_pack("runs/{out.name}")
```

`load_preview.py` 是同一条路径的最小示例。**这不是一个 vanilla transformers 权重目录**：
MoE 专家在盘上就是 20,992 个预量化 nf4 Linear，必须配合本仓库 `src/rnaqwen` 的 patch
（per-expert Linear 化 + Conv1d 替换 + SOS 层重建）才能装载。

## 里面是什么

| 文件 | 说明 |
|---|---|
| `weights/` | 量化权重分片 + `index.json`（key→分片、每片 sha256） |
{adapt}
| `fsq.pt` | FSQ 编码器，离线预计算码时用 |
| `code_emb.pt` | 码表 {meta['codec']['vocab']} x {meta['codec']['latent']} |
| `tokenizer.pt` | RNA tokenizer（含 `<RNA>` 占位） |
| `meta.json` | 版本戳与来源 |

## 版本耦合（PLAN §17.3）

`(encoder, code_table, adapter, base)` 必须整体一起动。任何不匹配都**静默降级**：
不报错、不崩溃，只是答得变差。本包的戳在 `meta.json`。
""", encoding="utf-8")


def write_loader(out: Path) -> None:
    (out / "load_preview.py").write_text('''"""Load this RNAQwen-preview pack.

    python load_preview.py        # rebuild + one forward pass

Needs this repo's src/rnaqwen on the path: the MoE experts are stored as
pre-quantized nf4 Linear layers and the SOS adapter is re-attached by
load_pack(), so a vanilla transformers load will not work.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnaqwen.preview import load_pack  # noqa: E402

if __name__ == "__main__":
    import torch

    model = load_pack(HERE)
    body = model.model.language_model
    ids = torch.arange(3, 19, dtype=torch.long).unsqueeze(0).to("cuda")
    with torch.no_grad():
        out = model(inputs_embeds=body.embed_tokens(ids))
    print("logits", tuple(out.logits.shape), "sum", float(out.logits.float().sum()))
''', encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/e2e_D_qwen36_sos/ckpt/final.pt")
    ap.add_argument("--fsq", default="checkpoints/fsq_stream/tokenizer.pt",
                    help="与新 checkpoint 配套的 FSQ tokenizer")
    ap.add_argument("--out", default="runs/RNAQwen-preview")
    ap.add_argument("--name", default="RNAQwen-preview")
    ap.add_argument("--shard-gb", type=float, default=3.0)
    ap.add_argument("--skip-reload", action="store_true")
    ap.add_argument("--merge", action="store_true",
                    help="把 SOS 折进 bf16 权重（会改变模型行为，见文件头）")
    args = ap.parse_args()

    t0 = time.time()
    ckpt_path = ROOT / args.ckpt
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cargs = ck["args"]
    base_dir, mk_kw = base_spec_from_ckpt_args(cargs)
    LORA_R = int(cargs.get("lora_r", 8))
    fsq_path = ROOT / args.fsq
    if not fsq_path.is_file():
        raise FileNotFoundError(fsq_path)
    tkc = torch.load(fsq_path, map_location="cpu", weights_only=False)
    LAT = int(ck["fsq"]["enc.weight"].shape[0])
    LEV = int(tkc["levels"])
    rc.LATENT, rc.LEVELS, rc.VOCAB = LAT, LEV, LEV ** LAT
    print(f"[cfg] ckpt {ckpt_path.name} step={ck.get('step')} "
          f"adapter={mk_kw['adapter']} preset={mk_kw['lora_preset']} "
          f"mode={mk_kw['base_mode']} base={base_dir.name}")
    print(f"[cfg] LATENT={LAT} LEVELS={LEV} VOCAB={LEV ** LAT} "
          f"pool={mk_kw['pool']} r={LORA_R}")

    print(f"\n[model] 装载 {base_dir.name} + adapter + 码表 …")
    model = rc.E2EModel(LORA_R, cargs.get("rna_emb", "table"),
                        latent=LAT, levels=LEV, **mk_kw)
    model.fsq.load_state_dict(ck["fsq"])
    if ck.get("rna_emb") is not None:
        model.rna_emb.weight.data.copy_(
            ck["rna_emb"].to(model.rna_emb.weight.device,
                             model.rna_emb.weight.dtype))
    miss = model.llm.load_state_dict(ck["lora"], strict=False)
    bad = list(getattr(miss, "unexpected_keys", None) or [])
    want = {n for n, p in model.llm.named_parameters() if p.requires_grad}
    got = set(ck["lora"].keys())
    print(f"[adapter] {len(got)} 张量  unexpected={len(bad)}  可训={len(want)}")
    if bad or (want - got):
        print(f"  ** adapter 与底座对不上：unexpected {bad[:3]}  "
              f"缺失 {sorted(want - got)[:3]}")
        return 3
    model.eval()
    cu, cl = commit_gb()
    print(f"[mem] commit {cu:.1f}/{cl:.1f} GB  装载 {time.time() - t0:.1f}s")

    sos = [m for m in model.llm.modules() if isinstance(m, SOSLoRALayer)]
    if not sos:
        print("  ** 模型里一个 SOSLoRALayer 都没有")
        return 3
    bad_q = [m for m in sos
             if isinstance(m.base, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))]
    if bad_q:
        print(f"  ** {len(bad_q)} 个 SOS 目标是量化层，合并会写坏量化权重，中止")
        return 3
    print(f"[sos] 目标 {len(sos)} 个，全部 nn.Linear，"
          f"dtype {sorted({str(m.base.weight.dtype) for m in sos})}")

    # 先量本底：同一份权重、同一个输入，连算两次。这一项不为 0 的话，
    # 后面任何 A/B 对比都说不清是合并的锅还是模型自己的抖动。
    ids = torch.arange(3, 131, dtype=torch.long).unsqueeze(0).to(DEV)
    am = torch.ones_like(ids)
    with torch.no_grad():
        ref = _logits(model.llm, ids, am)
        ref2 = _logits(model.llm, ids, am)
    mx_s, mean_s, rel_s, ag_s = _cmp(ref, ref2)
    print(f"[noise] 同权重同输入连算两次：max|Δ| {mx_s:.3e}  均值 {mean_s:.3e}  "
          f"相对 {rel_s:.3e}  top1 一致 {ag_s:.4f}")

    if args.merge:
        n_merged = merge_sos_layers(model.llm)
        with torch.no_grad():
            after = _logits(model.llm, ids, am)
        left = sum(1 for m in model.llm.modules() if isinstance(m, SOSLoRALayer))
        mx_m, mean_m, rel_m, ag_m = _cmp(ref, after)
        lim = max(1e-2, 3.0 * rel_s)
        print(f"[merge] 合并 {n_merged} 层，剩 {left} 个；max|Δ| {mx_m:.3e}  "
              f"均值 {mean_m:.3e}  相对 {rel_m:.3e}  top1 一致 {ag_m:.4f}")
        if left or rel_m > lim:
            print(f"  ** 合并漂移 {rel_m:.3e} > {lim:.3e}，中止")
            return 4
        ref = after              # 存下去的就是合并后的权重，校验基准换掉
        sos = []
        del after
    gc.collect()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n[pack] 写出 {out}"
          + ("（适配器已折进权重）" if args.merge else "（适配器不合并，另行重建）"))
    index = save_pack(model.llm, out, shard_gb=args.shard_gb)
    model.llm.config.to_json_file(str(out / "config.json"))
    torch.save(dict(state=ck["fsq"], latent=LAT, levels=LEV,
                    d_model=(int(ck["rna_emb"].shape[-1])
                             if ck.get("rna_emb") is not None else None)),
               out / "fsq.pt")
    if ck.get("rna_emb") is not None:
        torch.save(ck["rna_emb"], out / "code_emb.pt")
    torch.save(ck["lora"], out / "sos_adapter.pt")
    shutil.copyfile(fsq_path, out / "tokenizer.pt")

    sos_cfg = None
    if args.merge:
        stale = out / "sos.json"
        if stale.exists():
            stale.unlink()
    else:
        sos_cfg = {"file": "sos_adapter.pt",
                   "targets": lora_targets(mk_kw["lora_preset"]),
                   "r_tot": LORA_R, "k": int(mk_kw["sos_k"]),
                   "alpha": float(LORA_R * 2),
                   "gamma_max": float(mk_kw["sos_gamma_max"]),
                   "dropout": float(mk_kw["sos_dropout"])}
        (out / "sos.json").write_text(json.dumps(sos_cfg, indent=1),
                                      encoding="utf-8")
        print(f"[pack] sos.json  targets={len(sos_cfg['targets'])} 项  "
              f"K={sos_cfg['k']} r_tot={sos_cfg['r_tot']} alpha={sos_cfg['alpha']}")

    meta = {
        "name": args.name,
        "kind": "preview",
        "format": index["format"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base": {"dir": base_dir.name, "mode": mk_kw["base_mode"],
                 "quant": "nf4 experts only (bnb 0.50.2, compress_statistics=True)"},
        "adapter": {"type": mk_kw["adapter"], "preset": mk_kw["lora_preset"],
                    "r_tot": LORA_R, "sos_k": mk_kw["sos_k"],
                    "sos_gamma_max": mk_kw["sos_gamma_max"],
                    "merged": bool(args.merge),
                    "sos_json": "sos.json" if sos_cfg else None},
        "codec": {"latent": LAT, "levels": LEV, "vocab": LEV ** LAT,
                  "pool": mk_kw["pool"], "fsq": "fsq.pt",
                  "code_table": "code_emb.pt"},
        "train": {"ckpt": str(args.ckpt), "step": ck.get("step"),
                  "elapsed_h": round(float(ck.get("t_elapsed", 0.0)) / 3600, 2)},
        "weights": index,
        "note": (("weights/ are the *pre-merge* weights; load_pack() rebuilds the "
                  "SOS layers from sos.json and loads sos_adapter.pt. Exactly "
                  "equal to what was trained/evaluated. Merging is available "
                  "(97_ --merge) but measurably changes the function -- see "
                  "merge_drift.json. PLAN 17.3: encoder / code table / adapter / "
                  "base versions must move together.") if sos_cfg else
                 ("merged: the SOS delta is folded into the bf16 attention / "
                  "shared_expert weights. sos_adapter.pt is provenance only -- do "
                  "not apply it on top of weights/."))
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    write_readme(out, meta)
    write_loader(out)
    print(f"[pack] {len(index['shards'])} 分片  "
          f"{index['total_bytes'] / 2 ** 30:.2f} GiB  {index['total_keys']} keys")

    if args.skip_reload:
        print("\n[verify] 跳过回装校验")
        return 0
    print("\n[verify] 丢掉内存里的模型，从包重新装载 …")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    t1 = time.time()
    m2 = load_pack(out, dev=DEV)
    with torch.no_grad():
        again = _logits(m2, ids, am)
    same = torch.equal(ref, again)
    d2 = (ref - again).abs().max().item()
    print(f"[verify] 回装 {time.time() - t1:.1f}s  logits max|Δ| {d2:.3e}  "
          f"bitwise_equal={same}")
    if not same:
        print("  ** 回装后与内存里的权重不一致（应当逐位相同）")
        return 5

    if sos_cfg:
        print("\n[drift] 在回装出来的这一份上量：如果合并，会漂多少")
        n_merged = merge_sos_layers(m2)
        with torch.no_grad():
            merged = _logits(m2, ids, am)
        mx_d, mean_d, rel_d, ag_d = _cmp(again, merged)
        drift = {"n_merged": n_merged, "probe_tokens": int(ids.shape[1]),
                 "max_abs": mx_d, "mean_abs": mean_d, "rel_max": rel_d,
                 "top1_agree": ag_d, "noise_max_abs": mx_s,
                 "noise_top1_agree": ag_s,
                 "max_abs_logit": float(again.abs().max())}
        (out / "merge_drift.json").write_text(json.dumps(drift, indent=1),
                                              encoding="utf-8")
        print(f"[drift] 合并 {n_merged} 层；max|Δ| {mx_d:.3e}  均值 {mean_d:.3e}  "
              f"相对 {rel_d:.3e}  top1 一致 {ag_d:.4f}")
        if rel_d > 1e-2:
            print("  ** 合并会改变这个模型的输出 —— 所以包默认不合并")
        del merged
        gc.collect()

    cu, cl = commit_gb()
    print(f"[mem] commit {cu:.1f}/{cl:.1f} GB")
    print(f"[done] {out}  {index['total_bytes'] / 2 ** 30:.2f} GiB  "
          f"用时 {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
