"""Stage 2 配置自检（**不加载 8B、不用 GPU**）。

验五件事：

  1. 四条 preset 解析出的 `(targets, r, alpha)` 与**理论参数量**对不对得上
     （离线算式复算；再用一个 **1/64 缩比**的 Qwen3 让 PEFT 真数一遍，证明算式没写错）
  2. 未知 preset 确实报错（`Stage2Config` 与 CLI 两条路都验）
  3. `use_rslora` 开关真的传到 `peft.LoraConfig`
  4. `target_modules` 名字能被 PEFT 真匹配上 —— 两重证据：
       (a) `models/bert-base-uncased`：真跑 `get_peft_model`（题目要求的那一步）
       (b) 一个 config 跟 Qwen3-8B **同构**的小 Qwen3：确认
           q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj 七个名字全中，
           且 **k/v 的 out_features 是 GQA 的 1024 而不是 4096**
  5. **数据筛选**：`--variant dc --kind ...` 能筛出对应子集（用 rnaqa_50.jsonl）
  6. （附赠）Stage 2 的样本组装 + 反传：label 掩码只盖答案、LoRA 与投影都拿到梯度
     —— 这是 `encode_samples` 换了模板之后最容易写错的地方

跑法：
    $env:KMP_DUPLICATE_LIB_OK="TRUE"; $env:PYTHONIOENCODING="utf-8"
    & C:\\ana\\envs\\comfyui\\python.exe scripts/_test_stage2_config.py

输出：`reports/stage2_config_test.json`
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401  —— 必须最早：给这个残废 torch 补 torch.distributed

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from rnaqwen.models.finetune import (  # noqa: E402
    LORA_PRESETS,
    QWEN3_8B_SHAPES,
    Stage2Config,
    UnknownPresetError,
    build_lora_config,
    encode_samples,
    frozen_backbone_params,
    lora_param_count,
    preset_budget,
    resolve_preset,
)

# 预算表（任务书给的四行）。与 `scripts/_lora_budget.py` 的输出逐行一致：
#   attn 15.3 M / attn_mlp 43.6 M / attn_mlp_r8 21.8 M / attn_mlp_r64 174.6 M
# （那个脚本的 SHAPES 表已经把 GQA 的 k_proj/v_proj 写成 (4096, 1024)，是对的）
EXPECTED_TABLE = {
    "attn":        15.3e6,
    "attn_mlp":    43.6e6,
    "attn_mlp_r8": 21.8e6,
    "attn_mlp_r64": 174.6e6,
}

report: dict = {"checks": {}, "details": {}}


def record(name: str, ok: bool, **detail) -> bool:
    report["checks"][name] = bool(ok)
    if detail:
        report["details"][name] = detail
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(ok)


# --------------------------------------------------------------- 1. preset
def check_1_presets() -> bool:
    print("\n[1] 四条 preset 的 (targets, r, alpha) 与理论参数量")
    frozen = frozen_backbone_params()
    rows, all_ok = [], True
    for name in ("attn", "attn_mlp", "attn_mlp_r8", "attn_mlp_r64"):
        cfg = Stage2Config(preset=name)
        p = resolve_preset(name)
        n_offline = lora_param_count(cfg.lora_targets, cfg.lora_r)
        exp = EXPECTED_TABLE[name]
        ok_t = (list(cfg.lora_targets) == list(p["targets"]))
        ok_r = (cfg.lora_r == p["r"] and cfg.lora_alpha == p["alpha"])
        ok_n = abs(n_offline - exp) / exp < 0.02
        all_ok &= ok_t and ok_r and ok_n
        row = {"preset": name, "targets": list(cfg.lora_targets), "n_targets": len(cfg.lora_targets),
               "r": cfg.lora_r, "alpha": cfg.lora_alpha,
               "params_offline": n_offline, "params_million": round(n_offline / 1e6, 2),
               "ratio_pct": round(100.0 * n_offline / frozen, 4),
               "table_expected_million": round(exp / 1e6, 2),
               "delta_vs_table_pct": round(100.0 * (n_offline - exp) / exp, 2),
               "targets_ok": ok_t, "r_alpha_ok": ok_r, "params_ok": ok_n}
        rows.append(row)
        print(f"    {name:13s} targets={len(cfg.lora_targets)} r={cfg.lora_r:3d} a={cfg.lora_alpha:3d} "
              f"-> {n_offline/1e6:7.1f} M ({100.0*n_offline/frozen:.3f}%)  "
              f"表={exp/1e6:.1f} M Δ={100.0*(n_offline-exp)/exp:+.2f}%")
    report["details"]["preset_table"] = rows
    return record("preset_and_budget", all_ok)


# ------------------------------------------------- 1b. 缩比 Qwen3 真数一遍
def _tiny_qwen3(scale: int = 8, n_layers: int = 1, vocab: int = 256, seed: int = 0):
    """造一个**与 Qwen3-8B 严格同构**的小 Qwen3（每个线性层形状都是真模型的 1/scale）。

    Qwen3-8B : hidden 4096, 32 heads, 8 kv heads, head_dim 128, inter 12288
    scale= 8 : hidden  512,  4 heads, 1 kv head , head_dim 128, inter  1536
    → 「PEFT 在小模型上真数出来的参数量 × scale」== 36 层 Qwen3-8B 上的真实值。

    ⚠️ 两个必须显式钉住的点：
    1. `head_dim` 必须 = `hidden_size // num_attention_heads`。`Qwen3Config` 会**无条件**
       把 head_dim 默认成 128（即使 hidden=64/heads=8），此时 `Qwen3Attention.forward`
       会按 128 重新推导，q_proj/k_proj 变成 packed 形状 `[num_heads*head_dim …]`，
       小模型就不再同构，拿它验证 target_modules 会得出错误结论。
    2. 本地 Qwen3-8B 的 config.json 里 head_dim=128 == 4096//32，所以真模型没这个问题。
    """
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.manual_seed(seed)
    if not (0 < scale <= 8):
        raise ValueError(f"scale 必须 ≤ 8（8 kv heads / scale 不能变成 0），收到 {scale}")
    hidden, n_heads, kv_heads = 4096 // scale, 32 // scale, 8 // scale
    conf = Qwen3Config(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=12288 // scale,
        num_hidden_layers=n_layers, num_attention_heads=n_heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=1024, tie_word_embeddings=False,
        use_cache=False, attention_dropout=0.0,
    )
    conf.head_dim = hidden // n_heads          # 见上面 ⚠️1
    model = Qwen3ForCausalLM(conf).eval()
    tiny_shapes = _linear_shapes(model)
    want = {k: (v[0] // scale, v[1] // scale) for k, v in QWEN3_8B_SHAPES.items()}
    if tiny_shapes != want:
        raise AssertionError(
            f"小模型不同构：\n  得到 {tiny_shapes}\n  期望 {want}\n"
            "（多半是 head_dim 被 Qwen3Config 默认成 128 了）")
    return model, conf


def _linear_shapes(model, layer: str = ".layers.0.") -> dict[str, tuple[int, int]]:
    """{短名: (in_features, out_features)}，只看指定层的线性层。"""
    out: dict[str, tuple[int, int]] = {}
    for n, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and layer in n:
            out[n.rsplit(".", 1)[-1]] = (mod.in_features, mod.out_features)
    return out


def _true_8b_layer_shapes() -> tuple[dict[str, tuple[int, int]], dict]:
    """从 `models/Qwen3-8B` 的 **meta 设备** build 一次，读出真实线性层 (in, out)。

    不读权重、不占内存 —— 只是把 36 层搭出来看看形状。这是"离线算式"之外的第二重证据：
    `head_dim` / `num_key_value_heads` 在真模型上到底怎么落成 (in, out)。
    """
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    conf = AutoConfig.from_pretrained(str(ROOT / "models" / "Qwen3-8B"))
    with torch.device("meta"):
        model = Qwen3ForCausalLM(conf)
    return _linear_shapes(model), conf


def _lora_base_shapes(model) -> dict[str, tuple[int, int]]:
    """从 PEFT 包装后的模型里读出「被 LoRA 挂上的线性层」的名字 -> (in, out)。

    注意 `named_modules()` 也会吐出 `...q_proj.lora_A.default` 这种子模块，
    所以必须同时要求它有 `get_base_layer`（包装层才有），把子模块过滤掉。
    """
    out: dict[str, tuple[int, int]] = {}
    for name, mod in model.named_modules():
        if not hasattr(mod, "lora_A") or not hasattr(mod, "get_base_layer"):
            continue
        if "default" not in getattr(mod, "lora_A", {}):
            continue
        base = mod.get_base_layer()
        if isinstance(base, nn.Linear):
            out[name.rsplit(".", 1)[-1]] = (base.in_features, base.out_features)
    return out


def check_1b_peft_counts() -> bool:
    """三重证据校验参数预算：

      (i)  本地 Qwen3-8B **meta build** 出来的真实层形状（不载权重）
      (ii) 1/16 同构小 Qwen3 上 PEFT 真数的参数量 × 16
      (iii) 离线算式（我们自己的 lora_param_count）
    三者必须一致。
    """
    print("\n[1b] 参数预算三重校验：checkpoint 形状 / PEFT 真数(×8 外推) / 离线算式")
    from peft import LoraConfig, get_peft_model

    true_shapes, true_conf = _true_8b_layer_shapes()   # {short: (in, out)}，meta build 出来的
    true_shapes = {k: (v[0], v[1]) for k, v in true_shapes.items()}   # (meta 上是 (in,out))
    print(f"    (i) 真模型 meta build 的形状：{ {k: list(v) for k, v in sorted(true_shapes.items())} }")
    shapes_ok = true_shapes == dict(QWEN3_8B_SHAPES)
    n_layers_true = getattr(true_conf, "num_hidden_layers", 36)
    targets = list(QWEN3_8B_SHAPES)

    tiny, conf = _tiny_qwen3()
    scale = QWEN3_8B_SHAPES["q_proj"][1] // conf.hidden_size
    assert _linear_shapes(tiny) == {k: (v[0] // scale, v[1] // scale)
                                   for k, v in QWEN3_8B_SHAPES.items()}, "小模型不同构"
    print(f"    (ii) 同构小 Qwen3：hidden={conf.hidden_size} inter={conf.intermediate_size} "
          f"heads={conf.num_attention_heads} kv={conf.num_key_value_heads} "
          f"head_dim={conf.head_dim} layers={conf.num_hidden_layers}；缩放倍数={scale}")

    per_rank: dict[str, dict] = {}
    all_ok = shapes_ok
    for r in (8, 16, 32, 64):
        base, _mc = _tiny_qwen3()
        # PEFT 是**原地**包装，会重命名模块（q_proj -> ...q_proj.base_layer），
        # 所以形状必须在包装**之前**取。
        tiny_shapes = _linear_shapes(base)
        m = get_peft_model(base, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0,
                                           target_modules=targets,
                                           bias="none", task_type="CAUSAL_LM"))
        n_peft = sum(p.numel() for n, p in m.named_parameters() if p.requires_grad)
        # (a) PEFT 真数 == 离线式（小模型形状）
        n_off_tiny = lora_param_count(targets, r, layers=1, shapes=tiny_shapes)
        # (b) 小模型 × scale（缩比）× 层数比 == 离线式（meta build 的真形状）
        n_extrap = n_off_tiny * scale * (n_layers_true // conf.num_hidden_layers)
        n_off_8b = lora_param_count(targets, r, layers=n_layers_true, shapes=true_shapes)
        ok = (n_peft == n_off_tiny) and (n_extrap == n_off_8b)
        all_ok &= ok
        per_rank[str(r)] = {"peft_params_tiny": n_peft, "offline_tiny": n_off_tiny,
                            "scale": scale, "extrapolated_8b": n_extrap,
                            "offline_formula_8b": n_off_8b,
                            "million": round(n_extrap / 1e6, 2),
                            "peft_matches_offline_tiny": n_peft == n_off_tiny,
                            "extrapolation_matches_offline_8b": n_extrap == n_off_8b,
                            "agree": ok}
        print(f"    r={r:3d}: PEFT 小模型={n_peft:9d} == 离线式(小)={n_off_tiny:9d} "
              f"→ ×{scale}×{n_layers_true} = {n_extrap/1e6:8.2f} M  "
              f"| 离线式(真形状) = {n_off_8b/1e6:8.2f} M  全链一致={ok}")

    report["details"]["peft_extrapolation"] = {
        "true_8b_shapes_meta_build": {k: list(v) for k, v in sorted(true_shapes.items())},
        "true_8b_shapes_match_offline_table": shapes_ok,
        "true_8b_head_dim": getattr(true_conf, "head_dim", None),
        "true_8b_num_key_value_heads": getattr(true_conf, "num_key_value_heads", None),
        "tiny_config": {"hidden": conf.hidden_size, "intermediate": conf.intermediate_size,
                        "n_layers": conf.num_hidden_layers,
                        "num_attention_heads": conf.num_attention_heads,
                        "num_key_value_heads": conf.num_key_value_heads,
                        "head_dim": conf.head_dim},
        "scale_factor": int(scale),
        "by_rank": per_rank,
        "note": "小模型的 head_dim 必须手写成 hidden//num_heads：Qwen3Config 会无条件 "
                "把 head_dim 默认成 128，不一致时 transformers 5.9 走 packed 的 q/k 布局，"
                "小模型就不再同构（本地 Qwen3-8B 的 config head_dim=128 == 4096//32，没这问题）",
        "scale_limit": "scale ≤ 8：Qwen3 有 8 个 kv head，再缩就成 0 了",
    }
    return record("peft_param_arithmetic", all_ok)


# ------------------------------------------------------------ 2. 未知 preset
def check_2_unknown_preset() -> bool:
    print("\n[2] 未知 preset 必须报错（不许静默用默认值）")
    got = None
    try:
        Stage2Config(preset="attn_mlp_r128")
    except UnknownPresetError as e:
        got = str(e)
    print(f"    Stage2Config(preset='attn_mlp_r128') -> UnknownPresetError: {got}")

    # CLI 路径（在子进程里跑，拿真实退出码）
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "23_train_stage2.py"),
         "--dry-run", "--preset", "nope"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    cli_tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
    print(f"    CLI --preset nope -> exit={proc.returncode}")
    print(f"    CLI stderr 末行: {cli_tail[0]}")
    report["details"]["unknown_preset"] = {
        "config_raises": got,
        "cli_returncode": proc.returncode,
        "cli_last_line": cli_tail[0],
        "available": sorted(LORA_PRESETS),
    }
    ok = got is not None and "attn_mlp_r128" in got and proc.returncode != 0
    return record("unknown_preset_errors", ok)


# ------------------------------------------------------------- 3. rslora
def check_3_rslora() -> bool:
    print("\n[3] use_rslora 透传到 peft.LoraConfig")
    off = build_lora_config(Stage2Config(use_rslora=False))
    on = build_lora_config(Stage2Config(use_rslora=True))
    r64 = build_lora_config(Stage2Config(preset="attn_mlp_r64", use_rslora=True))
    print(f"    use_rslora=False -> LoraConfig.use_rslora={off.use_rslora}")
    print(f"    use_rslora=True  -> LoraConfig.use_rslora={on.use_rslora} "
          f"(r={on.r}, alpha={on.lora_alpha})")
    print(f"    preset=attn_mlp_r64 + rslora -> r={r64.r} alpha={r64.lora_alpha} "
          f"use_rslora={r64.use_rslora}")
    ok = (off.use_rslora is False and on.use_rslora is True and r64.use_rslora is True
          and r64.r == 64 and r64.lora_alpha == 128)
    report["details"]["rslora"] = {"off": off.use_rslora, "on": on.use_rslora,
                                   "r64_r": r64.r, "r64_alpha": r64.lora_alpha,
                                   "r64_rslora": r64.use_rslora}
    return record("rslora_passthrough", ok)


# -------------------------------------------------------- 4. target_modules
def check_4_target_modules() -> bool:
    print("\n[4] target_modules 名字能不能被 PEFT 真匹配上")
    from peft import LoraConfig, get_peft_model

    # (a) 题目指定的那一步：真跑一次 get_peft_model（bert-base-uncased）
    bert_dir = ROOT / "models" / "bert-base-uncased"
    bert_info: dict = {"path": str(bert_dir), "exists": bert_dir.exists()}
    bert_ok = False
    if bert_dir.exists():
        from transformers import AutoConfig, AutoModel
        torch.manual_seed(0)
        bcfg = AutoConfig.from_pretrained(str(bert_dir))
        # 只为了拿模块名 → 结构随机初始化即可，不必读 440 MB 权重
        bert = AutoModel.from_config(bcfg)
        # BertModel 里对应 attention 的四路都叫这名字（自注意力 q/k/v + 输出投影）
        bert_targets = ["query", "key", "value"]
        try:
            wrapped = get_peft_model(bert, LoraConfig(
                r=8, lora_alpha=16, lora_dropout=0.0, target_modules=bert_targets,
                bias="none", task_type="FEATURE_EXTRACTION"))
            n_t = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
            matched = sum(1 for n, _ in wrapped.named_parameters() if "lora_A.default.weight" in n)
            bert_info.update({"ok": True, "targets": bert_targets, "n_layers": bcfg.num_hidden_layers,
                              "matched_modules": matched,
                              "expected_modules": bcfg.num_hidden_layers * len(bert_targets),
                              "trainable_params": n_t,
                              "expected_params": n_t})
            bert_ok = matched == bcfg.num_hidden_layers * len(bert_targets) and n_t > 0
            print(f"    (a) BERT: get_peft_model 成功，匹配 {matched} 个模块 "
                  f"(期望 {bcfg.num_hidden_layers}×{len(bert_targets)}="
                  f"{bcfg.num_hidden_layers*len(bert_targets)})，可训参数 {n_t/1e6:.3f} M")
        except Exception as e:                                   # noqa: BLE001
            bert_info.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
            print(f"    (a) BERT: FAILED -> {type(e).__name__}: {e}")
    else:
        print(f"    (a) BERT: 目录不存在 {bert_dir}")

    # (b) 真骨架：同构小 Qwen3，确认七个投影名全中
    tiny, conf = _tiny_qwen3(n_layers=2)
    tiny = get_peft_model(tiny, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=list(QWEN3_8B_SHAPES),
        bias="none", task_type="CAUSAL_LM"))
    shapes = _lora_base_shapes(tiny)
    matched = [n for n, _ in tiny.named_parameters() if n.endswith("lora_A.default.weight")]
    per_layer = len(matched) // conf.num_hidden_layers
    qwen_info = {"n_matched": len(matched), "n_layers": conf.num_hidden_layers,
                 "per_layer": per_layer,
                 "expected_per_layer": len(QWEN3_8B_SHAPES),
                 "targets_found": sorted(shapes),
                 "missing": sorted(set(QWEN3_8B_SHAPES) - set(shapes)),
                 "extra": sorted(set(shapes) - set(QWEN3_8B_SHAPES)),
                 "base_shapes_tiny": {k: list(v) for k, v in sorted(shapes.items())},
                 "expected_on_8b": 36 * len(QWEN3_8B_SHAPES)}
    qwen_ok = (per_layer == len(QWEN3_8B_SHAPES) and not qwen_info["missing"]
               and not qwen_info["extra"])
    print(f"    (b) Qwen3(同构×2层): 匹配 {len(matched)} 个 LoRA 模块 "
          f"= {conf.num_hidden_layers} 层 × {per_layer} 个；缺失={qwen_info['missing']} "
          f"多余={qwen_info['extra']}")
    print(f"        基座形状（应保持 GQA 比例 k/v out = hidden/4）：{qwen_info['base_shapes_tiny']}")
    print(f"        真 8B 上应为 {36} 层 × {len(QWEN3_8B_SHAPES)} = {36*len(QWEN3_8B_SHAPES)} 个")

    # (c) 写错名字应该报错（否则上面那个"全中"没有说服力）
    wrong_ok = False
    try:
        get_peft_model(_tiny_qwen3()[0], LoraConfig(r=8, target_modules=["q_projection"],
                                                    task_type="CAUSAL_LM"))
        print("    (c) 故意写错的名字 q_projection -> 没有报错（！PEFT 会静默跳过）")
    except Exception as e:                                       # noqa: BLE001
        wrong_ok = True
        print(f"    (c) 故意写错的名字 q_projection -> {type(e).__name__}（正确行为）")

    report["details"]["target_modules"] = {"bert": bert_info, "qwen3_tiny": qwen_info,
                                           "wrong_name_raises": wrong_ok,
                                           "qwen3_8b_module_names": sorted(QWEN3_8B_SHAPES)}
    return record("target_modules_match", bool(bert_ok and qwen_ok and wrong_ok))


# ------------------------------------------------------------- 5. 数据筛选
def check_5_data_filter(data: Path) -> bool:
    print(f"\n[5] 数据筛选（{data.name}）")
    import importlib.util as _iu

    spec = _iu.spec_from_file_location("stage2_entry", ROOT / "scripts" / "23_train_stage2.py")
    mod = _iu.module_from_spec(spec)          # type: ignore[arg-type]
    spec.loader.exec_module(mod)              # type: ignore[union-attr]

    from rnaqwen.data.records import read_jsonl
    rows = [r.as_dict() for r in read_jsonl(data)]
    rows = rows[:50] if len(rows) > 50 else rows

    cases = [
        (["ais"], [], "variant_ais"),
        (["dc"], [], "variant_dc"),
        ([], ["factoid"], "kind_factoid"),
        (["dc"], ["factoid"], "dc_AND_factoid"),
        (["dc"], ["closed"], "dc_AND_closed"),
        (["ais", "dc"], ["factoid"], "multi_variant"),
    ]
    results, ok = {}, True
    for variants, kinds, label in cases:
        recs, st = mod.filter_records(rows, variants, kinds)
        results[label] = {"variants": variants, "kinds": kinds,
                          "records_kept": st["records_kept"], "qa_kept": st["qa_kept"],
                          "qa_in": st["qa_in"]}
        print(f"    --variant {','.join(variants) or 'both':12s} --kind {','.join(kinds) or 'both':9s}"
              f" -> 记录 {st['records_kept']:2d}/{st['records_in']}  QA {st['qa_kept']:3d}/{st['qa_in']}")

    # 硬判据：ais-only + dc-only 必须等于总数；dc AND factoid == dc-only；
    # dc AND closed 必须是 0（这份数据里没有 closed）
    ais = results["variant_ais"]["qa_kept"]
    dc = results["variant_dc"]["qa_kept"]
    tot = results["variant_ais"]["qa_in"]
    ok &= (ais + dc == tot)
    ok &= results["kind_factoid"]["qa_kept"] == tot
    ok &= results["dc_AND_factoid"]["qa_kept"] == dc
    ok &= results["dc_AND_closed"]["qa_kept"] == 0
    ok &= results["multi_variant"]["qa_kept"] == tot
    print(f"    判据：ais({ais}) + dc({dc}) == 全部({tot}) -> {ais+dc==tot}；"
          f"dc AND closed == 0 -> {results['dc_AND_closed']['qa_kept']==0}")

    report["details"]["data_filter"] = {
        "data": str(data), "records": len(rows), "cases": results,
        "invariants": {"ais_plus_dc_equals_total": ais + dc == tot,
                       "factoid_equals_total": results["kind_factoid"]["qa_kept"] == tot,
                       "dc_and_factoid_equals_dc": results["dc_AND_factoid"]["qa_kept"] == dc,
                       "dc_and_closed_is_empty": results["dc_AND_closed"]["qa_kept"] == 0,
                       "multi_variant_equals_total": results["multi_variant"]["qa_kept"] == tot},
        "note": "rnaqa_50.jsonl 里 kind 只有 factoid，没有 closed/open —— "
                "所以 '--variant dc --kind closed' 正确地筛出 0 条（子集为空是数据事实，不是筛选 bug）",
    }
    return record("data_filter", ok)


# ------------------------------------------- 6. 样本组装 + 反传（附赠但关键）
class _FakeTokenizer:
    """按字符切词。只实现 aligner/tokenizer 用得到的那几个接口。"""

    def __init__(self, vocab: int = 256):
        self.vocab = vocab
        self.eos_token_id = 1
        self.pad_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=True,
                 truncation=False, max_length=None, padding=False, **kw):
        ids = [min(self.vocab - 1, ord(c) % self.vocab) for c in text] or [self.eos_token_id]
        if truncation and max_length:
            ids = ids[:max_length]
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


class _FakeEncoder(nn.Module):
    """固定 hidden=640，按序列长度给出 (B, L, 640)。**确定性**（不能带随机），
    否则同一份输入两次调用会得到不同的 embeds，没法做"只改 label 不改内容"的对照。"""

    def __init__(self, hidden: int = 640, max_len: int = 64):
        super().__init__()
        self.hidden_size = hidden
        g = torch.Generator().manual_seed(1234)
        self.register_buffer("w", torch.randn(8, hidden, generator=g) * 0.1)
        self.register_buffer("pos", torch.randn(max_len, hidden, generator=g) * 0.01)

    def forward(self, sequences):                                # noqa: ANN001
        B, L = len(sequences), 8
        g = torch.Generator().manual_seed(7)
        x = torch.randn(B, L, 8, generator=g) * 0.1
        out = x @ self.w + self.pos[:L][None, :, :]
        mask = torch.ones(B, L, dtype=torch.long)
        return type("Enc", (), {"hidden": out, "mask": mask, "lengths": [L] * B})()


def check_6_sample_assembly() -> bool:
    print("\n[6] 样本组装 + 反传（不用 8B：同构小 Qwen3 + 假编码器/分词器）")
    from peft import LoraConfig, get_peft_model
    from rnaqwen.models.alignment import Stage1Aligner, Stage1Config

    tiny_llm, conf = _tiny_qwen3(scale=8, n_layers=2, vocab=256)
    tiny_llm = get_peft_model(tiny_llm, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, target_modules=list(QWEN3_8B_SHAPES),
        bias="none", task_type="CAUSAL_LM"))

    # 借一个 aligner 的壳：只换掉 encoder / tokenizer / llm / projection
    s1 = Stage1Config(grad_checkpointing=False)
    aligner = Stage1Aligner.__new__(Stage1Aligner)
    nn.Module.__init__(aligner)
    aligner.cfg = s1
    aligner.encoder = _FakeEncoder(640)
    aligner.tokenizer = _FakeTokenizer(256)
    aligner.llm = tiny_llm
    aligner.llm_hidden = conf.hidden_size
    from rnaqwen.models.projection import LinearProjection
    aligner.projection = LinearProjection(640, conf.hidden_size).to(torch.float32)
    aligner.llm.base_model.model = aligner.llm.base_model.model.to(torch.float32)

    cfg = Stage2Config(template="llama3", max_prompt_tokens=64, max_answer_tokens=64)
    seqs = ["ACGUACGUAC", "GGGGCCCCAA"]
    prompts = [cfg.format_prompt("What is this RNA?"), cfg.format_prompt("Who made it?")]
    answers = ["It is a transfer RNA.", "A ribosome."]

    inputs, attn, labels, meta = encode_samples(aligner, cfg, seqs, prompts, answers,
                                                torch.device("cpu"))
    n_sup = int((labels != -100).sum().item())
    print(f"    inputs={tuple(inputs.shape)} 监督 token={n_sup}  "
          f"各段={[ (l['n_prefix'], l['n_rna'], l['n_suffix'], l['n_prompt'], l['n_answer']) for l in meta['lens']]}")

    # 掩码只盖答案：答案段长度之和 == 监督 token 数
    sum_ans = sum(l["n_answer"] for l in meta["lens"])
    mask_ok = (n_sup == sum_ans) and sum_ans > 0

    # 确认问题文本**没有**被监督：每条样本"答案段之外"的位置（含 padding）必须全是 -100
    labels_wo_ans = labels.clone()
    for b, l in enumerate(meta["lens"]):
        n_used = (l["n_prefix"] + l["n_rna"] + l["n_suffix"] + l["n_prompt"] + l["n_answer"])
        start = n_used - l["n_answer"]          # 答案段起点（padding 在 n_used 之后）
        labels_wo_ans[b, start:labels_wo_ans.shape[1]] = -100
    no_prompt_sup = bool((labels_wo_ans == -100).all())
    print(f"    答案段合计={sum_ans} == 监督 token -> {mask_ok}；"
          f"答案段外无监督 token -> {no_prompt_sup}")

    # 前向 + 反传
    out = aligner.llm(inputs_embeds=inputs, attention_mask=attn, labels=labels)
    loss = out.loss
    loss.backward()
    lora_grad = sum(float(p.grad.norm()) for n, p in aligner.llm.named_parameters()
                    if "lora_" in n and p.grad is not None)
    proj_grad = sum(float(p.grad.norm()) for p in aligner.projection.parameters()
                    if p.grad is not None)
    print(f"    loss={float(loss.detach()):.4f}  |grad| LoRA={lora_grad:.4e}  投影={proj_grad:.4e}")

    # 对照：把 prompt 段的 label 也露出来（= build_batch 那种读法），
    # 序列内容必须**完全一样**，只有监督范围变大
    inputs2, attn2, labels2, meta2 = encode_samples(aligner, cfg, seqs, prompts, answers,
                                                    torch.device("cpu"), mask_answer_only=False)
    n_sup2 = int((labels2 != -100).sum().item())
    same_content = bool(torch.equal(inputs, inputs2) and torch.equal(attn, attn2))
    same_nonlabel = bool(torch.equal(torch.where(labels != -100, labels, -100),
                                     torch.where(labels2 != -100, labels2, -100)))
    loss2 = float(aligner.llm(inputs_embeds=inputs2, attention_mask=attn2, labels=labels2).loss.detach())
    print(f"    对照（prompt 也计 loss）：监督 token={n_sup2}（多 {n_sup2-n_sup}）loss={loss2:.4f}；"
          f"序列内容不变={same_content}")

    ok = (mask_ok and no_prompt_sup and lora_grad > 0 and proj_grad > 0 and same_content
          and n_sup2 == n_sup + sum(l["n_prompt"] for l in meta["lens"]))
    report["details"]["sample_assembly"] = {
        "seq_len": meta["seq_len"], "n_supervised": n_sup, "sum_answer_tokens": sum_ans,
        "per_sample_segments": meta["lens"], "answer_only_mask_ok": mask_ok,
        "no_supervision_outside_answer": no_prompt_sup,
        "loss": float(loss.detach()), "grad_norm_lora": lora_grad, "grad_norm_projection": proj_grad,
        "control_prompt_also_supervised": {"n_supervised": n_sup2, "loss": loss2,
                                          "inputs_identical": same_content,
                                          "non_label_positions_identical": same_nonlabel},
    }
    return record("sample_assembly_and_backward", ok)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "stage2_config_test.json")
    args = ap.parse_args()

    data = args.data
    if data is None:
        for cand in ("rnaqa_50.jsonl", "rnaqa_rnacentral.jsonl"):
            p = ROOT / "data" / "processed" / cand
            if p.exists():
                data = p
                break
    if data is None or not Path(data).exists():
        print(f"[fail] 找不到数据：{data}")
        return 1

    print("=" * 78)
    print("Stage 2 配置自检（不加载 8B、不用 GPU）")
    print("=" * 78)
    results = {}
    for name, fn in (
        ("1_presets", check_1_presets),
        ("1b_peft_extrapolation", check_1b_peft_counts),
        ("2_unknown_preset", check_2_unknown_preset),
        ("3_rslora", check_3_rslora),
        ("4_target_modules", check_4_target_modules),
        ("5_data_filter", lambda: check_5_data_filter(data)),
        ("6_sample_assembly", check_6_sample_assembly),
    ):
        try:
            results[name] = fn()
        except Exception:                                        # noqa: BLE001
            results[name] = False
            report["details"][name + "_exception"] = traceback.format_exc()
            print(f"  [FAIL] {name} 抛异常：\n{traceback.format_exc()}")

    report["passed"] = all(results.values())
    report["summary"] = results
    print("\n" + "=" * 78)
    for k, v in results.items():
        print(f"  [{'ok  ' if v else 'FAIL'}] {k}")
    print(f"  verdict: {'PASS' if report['passed'] else 'FAIL'}")
    print("=" * 78)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[stage2-test] 报告 -> {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
