"""把训练检查点导出成 artifacts 布局，好复用 64_acc_nothink.py。

训练检查点（save_ckpt）的格式：
    dict(step, fsq=state_dict, rna_emb=tensor, lora={lora_ 键}, opt, log, args, ...)
artifacts 布局：
    fsq.pt  code_emb.pt  meta.json  lora_adapter/{adapter_config.json, adapter_model.safetensors}
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora-config", default="runs/A_v2_4096/artifacts/lora_adapter/adapter_config.json",
                    help="从已有的 artifacts 借一份 adapter_config.json（同一套 LoRA 配置）")
    args = ap.parse_args()

    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    print(f"[ckpt] step={ck.get('step')}")
    for k in ("fsq", "rna_emb", "lora"):
        v = ck.get(k)
        if v is None:
            print(f"  **缺 {k}**")
        elif isinstance(v, dict):
            print(f"  {k}: {len(v)} 个张量")
        else:
            print(f"  {k}: {tuple(v.shape)}")

    lat = int(ck["fsq"]["enc.weight"].shape[0])
    tk = torch.load(ROOT / "checkpoints/fsq_full/tokenizer.pt", map_location="cpu",
                    weights_only=False)
    lev = int(tk["levels"])
    print(f"[cfg] 从 fsq 权重推 LATENT={lat}   从 tokenizer 读 LEVELS={lev}  VOCAB={lev**lat}")

    out = ROOT / args.out
    (out / "lora_adapter").mkdir(parents=True, exist_ok=True)

    torch.save(dict(state=ck["fsq"], latent=lat, levels=lev), out / "fsq.pt")
    torch.save(ck["rna_emb"], out / "code_emb.pt")
    (out / "meta.json").write_text(json.dumps(
        dict(rna_emb_mode="table", vocab=lev ** lat, latent=lat, levels=lev,
             from_ckpt=args.ckpt, step=ck.get("step")), indent=2), encoding="utf-8")

    src_cfg = ROOT / args.lora_config
    if src_cfg.exists():
        shutil.copy(src_cfg, out / "lora_adapter" / "adapter_config.json")
    else:
        raise SystemExit(f"缺 adapter_config.json: {src_cfg}")
    save_file({k: v.contiguous() for k, v in ck["lora"].items()},
              str(out / "lora_adapter" / "adapter_model.safetensors"))
    print(f"[save] {out}")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(out)}  {f.stat().st_size/1024/1024:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
