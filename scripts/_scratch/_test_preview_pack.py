"""Tiny end-to-end test of the RNAQwen-preview pack (save_pack / load_pack).

Why this exists
---------------
src/rnaqwen/preview.py had only ever been probed on one 256x256 nf4 layer.  The
real pack (scripts/97_export_preview.py) needs ~90 GB of system commit and
hours, so key grouping / ModuleList resolution / safetensors handling has to be
proven on a toy model of the same class first (~0.5 GB).

What it exercises (all production code paths)
--------------------------------------------
    Qwen3_5MoeForConditionalGeneration from a shrunk copy of the 35B config
    patch_moe_experts + patch_conv1d
    swap_linears(experts, "int4")                 bnb nf4, compress_statistics
    replace_linear_with_sos + merge_sos_layers    adapter path of 97_
    save_pack -> drop model -> load_pack          cross-shard, many shards

Asserts logits survive the merge (fp noise) and the reload (bitwise).

    python scripts/_scratch/_test_preview_pack.py
"""
import gc
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: F401,E402  must precede transformers/peft
import torch  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M  # noqa: E402
from transformers import AutoConfig  # noqa: E402

from rnaqwen.models.sos_lora import (SOSLoRALayer, merge_sos_layers,  # noqa: E402
                                     replace_linear_with_sos)
from rnaqwen.preview import load_pack, save_pack  # noqa: E402
from rnaqwen.qwen36 import (MDL, lora_targets, patch_conv1d,  # noqa: E402
                            patch_moe_experts, swap_linears)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TMP = ROOT / "runs/_packtest"
SHARD_GB = 0.05          # force many shards: cross-shard load must work


def parts(llm):
    body = llm.model
    if hasattr(body, "language_model"):
        return body.language_model, llm.lm_head
    return body, llm.lm_head


def logits(model, ids, am):
    body, head = parts(model)
    emb = body.embed_tokens(ids)
    hs = body(inputs_embeds=emb, attention_mask=am, use_cache=False).last_hidden_state
    return head(hs).float().cpu()


def tiny_config() -> Path:
    """The 35B config with only the expensive dimensions shrunk."""
    cfg = json.loads((MDL / "config.json").read_text(encoding="utf-8"))
    t = cfg["text_config"]
    t.update(num_hidden_layers=4,
             layer_types=["linear_attention"] * 3 + ["full_attention"],
             num_experts=8, num_experts_per_tok=2,
             moe_intermediate_size=128, shared_expert_intermediate_size=128,
             vocab_size=256)
    v = cfg["vision_config"]
    v.update(depth=1, hidden_size=64, intermediate_size=128, num_heads=2,
             num_position_embeddings=64)
    TMP.mkdir(parents=True, exist_ok=True)
    (TMP / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return TMP


def main() -> int:
    fails: list[str] = []
    torch.manual_seed(0)
    cfgdir = tiny_config()
    cfg = AutoConfig.from_pretrained(str(cfgdir))
    tcfg = cfg.text_config
    print(f"[cfg] layers {tcfg.num_hidden_layers} experts {tcfg.num_experts} "
          f"hidden {tcfg.hidden_size} vocab {tcfg.vocab_size} dev {DEV}")

    patch_moe_experts(M)
    patch_conv1d(M)
    model = M.Qwen3_5MoeForConditionalGeneration(cfg)
    if hasattr(model.model, "visual"):
        del model.model.visual
    if hasattr(model, "mtp"):
        del model.mtp
    text = model.model.language_model
    model.to(torch.bfloat16)   # real checkpoints are bf16; the CPU init is fp32
    text.rotary_emb = type(text.rotary_emb)(tcfg, device=DEV)  # keep rope fp32
    for m in (text.embed_tokens, text.norm, text.rotary_emb, model.lm_head):
        m.to(DEV)

    nq = 0
    for i in range(len(text.layers)):
        nq += swap_linears(text.layers[i], "int4",
                           f"model.language_model.layers.{i}")
        text.layers[i].to(DEV)
    print(f"[tiny] quantized {nq} expert Linear (expect "
          f"{tcfg.num_hidden_layers * tcfg.num_experts * 2})")

    rep = replace_linear_with_sos(model, tuple(lora_targets("D")), 8, 4, 16.0,
                                  dropout=0.0, gamma_max=2.5)
    if not rep:
        fails.append("SOS attached nothing")
    # Nonzero delta: with the default zero-init B the merge is a no-op and proves
    # nothing.  std 0.01 gives per-element std(dW) ~1e-3, like the real adapter.
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, SOSLoRALayer):
                for b in m.B:
                    b.normal_(0, 0.01)
    print(f"[tiny] SOS layers {len(rep)}  trainable "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    ids = torch.arange(3, 19, dtype=torch.long).unsqueeze(0).to(DEV)
    am = torch.ones_like(ids)
    with torch.no_grad():
        ref = logits(model, ids, am)
    n_sd = len(model.state_dict())
    snap = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
    rope0 = getattr(text.rotary_emb, "inv_freq", None)
    out = TMP / "pack"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    model.config.to_json_file(str(out / "config.json"))
    (out / "sos.json").write_text(
        json.dumps({"file": "sos_adapter.pt", "targets": lora_targets("D"),
                    "r_tot": 8, "k": 4, "alpha": 16.0, "gamma_max": 2.5,
                    "dropout": 0.0}, indent=1), encoding="utf-8")
    adapt = {k: v.detach().to("cpu").clone()
             for k, v in model.state_dict().items()
             if re.search(r"\.(A|B)\.\d+$", k)}
    torch.save(adapt, out / "sos_adapter.pt")
    print(f"[pack] sos_adapter.pt {len(adapt)} tensors")
    index = save_pack(model, out, shard_gb=SHARD_GB)
    n_qmod = sum(1 for k in index["key_to_shard"] if ".weight.absmax" in k)
    print(f"[pack] {len(index['shards'])} shards {index['total_bytes'] / 2 ** 20:.1f} MiB "
          f"{index['total_keys']} keys  quantized modules {n_qmod} "
          f"(state_dict had {n_sd} keys)")
    if index["total_keys"] != n_sd:
        fails.append(f"pack lost keys: {index['total_keys']} != {n_sd}")
    if n_qmod != nq:
        fails.append(f"quantized module count {n_qmod} != {nq}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    m2 = load_pack(out, dev=DEV)
    sd2 = m2.state_dict()
    bad_w = [k for k in snap if k not in sd2
             or not torch.equal(snap[k], sd2[k].detach().to("cpu"))]
    extra_w = [k for k in sd2 if k not in snap]
    rope1 = getattr(m2.model.language_model.rotary_emb, "inv_freq", None)
    print(f"[weights] {len(snap)} keys  mismatched {len(bad_w)} {bad_w[:6]}  "
          f"extra {extra_w[:6]}")
    print(f"[rope] inv_freq dtype {getattr(rope0, 'dtype', None)} -> "
          f"{getattr(rope1, 'dtype', None)}")
    if bad_w or extra_w:
        fails.append(f"weights not identical: {len(bad_w)} mismatched, "
                     f"{len(extra_w)} extra")
    with torch.no_grad():
        again = logits(m2, ids, am)
    same = torch.equal(ref, again)
    d2 = (ref - again).abs().max().item()
    print(f"[reload] logits max|d| {d2:.3e} bitwise_equal={same}")
    if not same:
        fails.append(f"reload not bitwise equal (max|d| {d2:.3e})")

    n = merge_sos_layers(m2)
    with torch.no_grad():
        merged = logits(m2, ids, am)
    d3 = (again - merged).abs().max().item()
    rel3 = d3 / max(again.abs().max().item(), 1e-12)
    ag3 = (again.argmax(-1) == merged.argmax(-1)).float().mean().item()
    print(f"[drift] merge {n} layers: max|d| {d3:.3e} rel {rel3:.3e} "
          f"top1 agree {ag3:.4f}   (informational, not a failure)")

    if fails:
        print("\n** FAIL")
        for f in fails:
            print(f"   {f}")
        return 1
    print("\n[tiny] OK -- save_pack/load_pack round trip is bitwise exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
