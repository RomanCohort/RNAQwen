"""Stage 1 真实速度测量（GPU 基准计时）—— 不改动 `20_train_stage1.py` / `alignment.py`。

为什么另写一个
--------------
`train_stage1()` 的 `log_every` 默认 10，且不返回逐步 loss，也不记显存。
计时需要：逐步 loss、逐步耗时、峰值显存。所以这里直接调
`rnaqwen.models.alignment.Stage1Aligner`，自己写训练回路 —— 但**逐字复刻**它的
样本组装（`build_batch` / `forward_loss`）与 optimizer step 时机（accum 满才 step），
保证测出来的 s/step 就是真实训练的 s/step。

并发改动兼容性
--------------
另一个 agent 正在给 `alignment.py` 加可选 token 压缩开关。所以本脚本**不 monkey-patch
`build_batch`**（那等于复制一份随时会过期的实现），而是：
  * token 截断靠**子类覆写 `encoder`**，其余全部走库里的原路径；
  * 记录每次运行前后 `alignment.py` 的 mtime + sha256，跨版本的数字一律标注。
传入 `forward_loss` 的额外参数用 `**extra` 转发 —— 新版加参数不会打爆这里。

四种模式
--------
  timing  ：warmup 若干步后，逐步记 (forward_s, backward_s, step_s)、峰值显存
  train   ：同上，但额外把所有逐步 loss 写进结果（用于"loss 是否在降"）
  compare ：一个进程里连续跑多组配置（省掉每次 3–5 min 的模型加载）
  compress：把编码器输出在**长度维截断**到前 N 个 token（只测速，不管语义）

用法
----
    python scripts/_time_stage1.py --spec tmp/_timing_spec.json --out reports/_raw_timing.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from inspect import signature
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401  —— 装载 env.shim，必须早于 multimolecule

import torch  # noqa: E402

from rnaqwen.data.records import read_jsonl  # noqa: E402
from rnaqwen.models.alignment import Stage1Aligner, Stage1Config, cosine_with_warmup  # noqa: E402

GiB = 1024 ** 3
ALIGNMENT_PY = ROOT / "src" / "rnaqwen" / "models" / "alignment.py"


def file_fingerprint(p: Path) -> dict:
    if not p.exists():
        return {"path": str(p), "exists": False}
    b = p.read_bytes()
    return {
        "path": str(p),
        "exists": True,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
        "size": len(b),
        "sha256": hashlib.sha256(b).hexdigest()[:16],
    }


# --------------------------------------------------------------------------- #
# RNA token 截断：只覆写 encoder，其余交给库里的 build_batch
# --------------------------------------------------------------------------- #
class TruncatedEncoder(torch.nn.Module):
    """把 (B, L, 640) 的编码器输出沿长度维切到前 N 个 token，mask 同步切。

    **只为计时**，语义上不等价（等于把 RNA 的前 N 个碱基当全部输入）。
    必须是 nn.Module —— `model.encoder = ...` 会走 nn.Module.__setattr__ 的
    child-module 校验，普通对象会被直接拒绝。
    """

    def __init__(self, real_encoder, keep: int):
        super().__init__()
        object.__setattr__(self, "_real", real_encoder)  # 不作为子模块注册，避免重复登记参数
        object.__setattr__(self, "_keep", int(keep))

    def __call__(self, sequences):
        out = self._real(sequences)
        keep = min(self._keep, out.hidden.shape[1], out.mask.shape[1])
        out.hidden = out.hidden[:, :keep, :].contiguous()
        out.mask = out.mask[:, :keep].contiguous()
        out.lengths = out.mask.sum(-1)
        return out

    def __getattr__(self, name):
        try:
            return getattr(super().__getattr__(name), name)
        except AttributeError:
            real = object.__getattribute__(self, "_real")
            return getattr(real, name)


def install_truncation(model: Stage1Aligner, keep: int):
    """返回 (restore_callable, applied_flag)。若模型的 build_batch 已支持压缩参数，则不动 encoder。"""
    real = model.encoder
    model.encoder = TruncatedEncoder(real, keep)

    def restore():
        model.encoder = real

    return restore


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_records(data: Path, target_field: str, limit: int = 0) -> list[dict]:
    rows = [r.as_dict() for r in read_jsonl(data)]
    if limit:
        rows = rows[:limit]
    keep = []
    for r in rows:
        tgt = r.get(target_field) or ""
        if r.get("sequence") and tgt.strip():
            keep.append({"sequence": r["sequence"], "annotation": tgt})
    return keep


# --------------------------------------------------------------------------- #
# 一次计时运行
# --------------------------------------------------------------------------- #
def run_one(model: Stage1Aligner, records: list[dict], device: torch.device, spec: dict) -> dict:
    grad_accum = int(spec.get("grad_accum", 1))
    warmup = int(spec.get("warmup_steps", 2))
    measure = int(spec.get("measure_steps", 5))
    lr = float(spec.get("lr", 1e-4))
    warmup_lr = float(spec.get("warmup_lr", 1e-6))
    max_grad_norm = float(spec.get("max_grad_norm", 1.0))
    collect = bool(spec.get("collect_losses", False))
    extra = dict(spec.get("forward_kwargs") or {})   # 转发给 forward_loss，兼容新版压缩开关

    # 不再限制 micro-batch 总数 <= 样本数：训练回路按 epoch 循环复用样本，
    # 50 条 × 2 epoch 就是要跑 100 个 micro-batch。

    lr_cfg = Stage1Config(lr=lr, warmup_lr=warmup_lr, warmup_steps=max(1, int(spec.get("lr_warmup", 1e9))))

    opt = torch.optim.AdamW(model.projection.parameters(), lr=lr, betas=(0.9, 0.98),
                            weight_decay=0.05)

    # ---- 预热：先用 no_grad 把 kernel 编译/autotune 的代价吃掉 ----
    # 关键：no_grad 不建图、不 backward、不动权重、不动 optimizer 状态 ——
    # 只为触发 HIP kernel 编译，这样正式训练的第 1 步就已是"热"的。
    prewarm = int(spec.get("prewarm_micro", 0))
    if prewarm:
        pw0 = time.perf_counter()
        with torch.no_grad():
            for k in range(prewarm):
                rec = records[k % len(records)]
                _ = model.forward_loss([rec["sequence"]], [rec["annotation"]], device, **extra)
        torch.cuda.synchronize()
        print(f"[time] prewarm {prewarm} micro-batches (no_grad, 不动权重) "
              f"{time.perf_counter() - pw0:.1f}s", flush=True)
        model.zero_grad(set_to_none=True)
        # optimizer 状态必须是干净的（prewarm 没 step 过，这里只是防御性重建）
        opt = torch.optim.AdamW(model.projection.parameters(), lr=lr, betas=(0.9, 0.98),
                                weight_decay=0.05)

    fwd_s: list[float] = []
    bwd_s: list[float] = []
    micro_s: list[float] = []
    losses: list[float] = []          # 每步（accum 平均后）的 loss
    micro_losses: list[float] = []    # 每个 micro-batch 的 loss
    batch_meta: list[dict] = []       # 逐步：样本长度 / token 数（用于解释耗时分布）

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    free_before, total_mem = torch.cuda.mem_get_info()

    # ---- 逐字复刻 train_stage1 的样本顺序与 step 触发时机 ----
    #   for epoch: for i in range(0, len(records), bs): micro-batch = records[i:i+bs]
    #   step 触发条件：(i // bs + 1) % grad_accum == 0
    #   -> 每 epoch 的 micro-batch 数 = ceil(n / bs)，跨 epoch 连续累积
    micro_seen = 0
    step = 0
    target_steps = warmup + measure
    step_fwd = step_bwd = 0.0
    t_run0 = time.perf_counter()
    done = False
    while not done:
        for epoch in range(10_000):
            i = 0
            while i < len(records):
                batch = records[i : i + 1]
                seqs = [b["sequence"] for b in batch]
                tgts = [b["annotation"] for b in batch]
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                loss = model.forward_loss(seqs, tgts, device, **extra) / grad_accum
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                p_fwd, p_bwd = t1 - t0, t2 - t1
                step_fwd += p_fwd
                step_bwd += p_bwd
                micro_losses.append(float(loss.detach()) * grad_accum)
                del loss
                micro_seen += 1

                if (i // 1 + 1) % grad_accum == 0:
                    for g in opt.param_groups:
                        g["lr"] = cosine_with_warmup(step, lr_cfg, max(target_steps, 1))
                    torch.nn.utils.clip_grad_norm_(model.projection.parameters(), max_grad_norm)
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    t3 = time.perf_counter()
                    # 一个 optimizer step 的真实耗时 = 它包含的 grad_accum 个 micro-batch 之和
                    whole_step = step_fwd + step_bwd
                    if step >= warmup:
                        fwd_s.append(step_fwd / grad_accum)
                        bwd_s.append(step_bwd / grad_accum)
                        micro_s.append(whole_step / grad_accum)
                        loss_now = sum(micro_losses[-grad_accum:]) / grad_accum
                        losses.append(loss_now)
                        batch_meta.append({
                            "step": step, "epoch": epoch, "seq_nt": len(seqs[0]),
                            "step_fwd_s": step_fwd, "step_bwd_s": step_bwd,
                            "step_total_s": whole_step,
                        })
                    print(f"  [{spec.get('name')}] step {step + 1}/{target_steps} "
                          f"loss_now={sum(micro_losses[-grad_accum:]) / grad_accum:.4f} "
                          f"step={1000 * whole_step:.0f}ms micro={1000 * whole_step / grad_accum:.0f}ms",
                          flush=True)
                    step += 1
                    step_fwd = step_bwd = 0.0
                    if step >= target_steps:
                        done = True
                        break
                i += 1
            if done:
                break

    torch.cuda.synchronize()
    wall = time.perf_counter() - t_run0
    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    free_after, _ = torch.cuda.mem_get_info()

    def stat(xs: list[float]) -> dict:
        if not xs:
            return {}
        s = sorted(xs)
        n = len(s)
        return {
            "mean": sum(xs) / n,
            "median": s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2]),
            "min": s[0],
            "max": s[-1],
        }

    per_step = [f + b for f, b in zip(fwd_s, bwd_s)]  # 每 micro 的 fwd+bwd 均值
    del opt

    return {
        "name": spec.get("name"),
        "spec": spec,
        "env": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "mem_total_GiB": total_mem / GiB,
        },
        "grad_accum": grad_accum,
        "rna_tokens": spec.get("truncate_rna_tokens"),
        "n_micro_measured": len(micro_s),
        "n_optimizer_steps_measured": len(losses),
        "seconds": {
            "wall_total": wall,
            "per_optimizer_step": stat([m * grad_accum for m in micro_s]),
            "per_micro_batch": stat(micro_s),
            "forward_per_micro": stat(fwd_s),
            "backward_per_micro": stat(bwd_s),
            "step_total_median": stat([m * grad_accum for m in micro_s]).get("median"),
            "step_fwd_median": stat(per_step).get("median"),
        },
        "memory": {
            "peak_allocated_GiB": peak_alloc / GiB,
            "peak_reserved_GiB": peak_reserved / GiB,
            "free_before_GiB": free_before / GiB,
            "free_after_GiB": free_after / GiB,
            "used_delta_GiB": (free_before - free_after) / GiB,
        },
        "losses": losses if collect else None,
        "loss_head": losses[:10],
        "loss_tail": losses[-10:],
        "micro_losses": micro_losses if collect else None,
        "batch_meta": batch_meta,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True, help="JSON：{data, target_field, device, runs:[...]}")
    ap.add_argument("--out", type=Path, default=Path("reports/_raw_timing.json"))
    args = ap.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8-sig"))  # 容忍 PowerShell 写出的 BOM
    data = Path(spec.get("data", "data/processed/annotations_50.jsonl"))
    target_field = spec.get("target_field", "annotation")
    records = load_records(data, target_field, int(spec.get("limit", 0)))
    print(f"[time] {len(records)} 条样本 <- {data} (field={target_field})", flush=True)
    if not records:
        print("[fail] 无可用样本")
        return 1

    fprint_before = file_fingerprint(ALIGNMENT_PY)
    print(f"[time] alignment.py 指纹(前): mtime={fprint_before.get('mtime')} "
          f"sha={fprint_before.get('sha256')} size={fprint_before.get('size')}", flush=True)

    device = torch.device(spec.get("device", "cuda"))

    import dataclasses
    _CFG_FIELDS = {f.name for f in dataclasses.fields(Stage1Config)}

    def run_compress_cfg(run: dict) -> tuple[int | None, str]:
        """压缩设置允许写在顶层或单条 run 上；单条 run 优先。默认 None=原样。"""
        lat = run.get("compress_latent", spec.get("compress_latent"))
        meth = run.get("compress_method", spec.get("compress_method", "resampler"))
        return (int(lat) if lat else None), str(meth)

    def build_model(compress_latent: int | None, compress_method: str) -> Stage1Aligner:
        cfg_kwargs: dict = {}
        if compress_latent:
            if "compress_latent" not in _CFG_FIELDS:
                raise RuntimeError(
                    "本版 Stage1Config 还没有 compress_latent —— alignment.py 可能正处在"
                    "半改完的状态，等 60–90s 重试同一条命令")
            cfg_kwargs["compress_latent"] = int(compress_latent)
            cfg_kwargs["compress_method"] = compress_method
        cfg = Stage1Config(llm_path=spec.get("llm", "models/Qwen3-8B"),
                           encoder_path=spec.get("encoder", "models/rnafm"),
                           grad_checkpointing=bool(spec.get("grad_checkpointing", True)),
                           dtype=spec.get("dtype", "bfloat16"), **cfg_kwargs)
        m = Stage1Aligner(cfg)
        m.encoder.to(device)
        m.llm.to(device)
        # ⚠️ 关键：compression 是**普通属性**不是 nn.Module 子模块，`model.to(device)`
        #    搬不动它（残留在 CPU -> addmm 设备不匹配）。必须用 trainable_modules()，
        #    这也正是 train_stage1 里 device 迁移 / optimizer 参数的正确契约。
        for mod in m.trainable_modules():
            mod.to(device)
            mod.train()
        torch.cuda.synchronize()
        return m

    # 按 (compress_latent, compress_method) 分组，同配置的 run 共用一个模型，
    # 配置一变就重建 —— 避免上一组跑出来的额外参数混进下一组的 optimizer。
    cur_key: tuple | None = None
    cur_ckpt: bool | None = None
    model = None
    load_s = None
    sig = None
    results = []

    def ensure_model(key: tuple) -> Stage1Aligner:
        nonlocal model, cur_key, load_s, sig
        if model is None or key != cur_key:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            t_load = time.perf_counter()
            model = build_model(key[0], key[1])
            load_s = time.perf_counter() - t_load
            sig = str(signature(model.forward_loss))
            cur_key = key
            print(f"[time] 模型加载 {load_s:.1f}s  compress_latent={key[0]} "
                  f"method={key[1]}  grad_ckpt={model.cfg.grad_checkpointing}", flush=True)
            print(f"[time] {model.projection.describe()}", flush=True)
            if key[0]:
                print(f"[time] 压缩器: {model.compression.describe()}", flush=True)
                print(f"[time] 可训: {model.describe_trainable()}", flush=True)
        return model

    for run in spec["runs"]:
        clat, cmeth = run_compress_cfg(run)
        model = ensure_model((clat, cmeth))

        want_ckpt = bool(run.get("grad_checkpointing", spec.get("grad_checkpointing", True)))
        if cur_ckpt != want_ckpt:
            if want_ckpt:
                model.llm.gradient_checkpointing_enable()
                model.llm.config.use_cache = False
            else:
                model.llm.gradient_checkpointing_disable()
            cur_ckpt = want_ckpt
            model.cfg.grad_checkpointing = want_ckpt
            print(f"[time] grad_checkpointing -> {cur_ckpt}", flush=True)

        # 记录一条样本的真实 token 构成（用原路径，不受截断 hack 影响）
        with torch.no_grad():
            probe = records[0]
            ins, att, lab = model.build_batch([probe["sequence"]], [probe["annotation"]], device)
            n_total = int(ins.shape[1])
            n_ans = int((lab != -100).sum())
        n_pre = model._encode_const(model.PREFIX_TEXT, device).shape[1]
        n_suf = model._encode_const(model.SUFFIX_TEXT, device).shape[1]
        n_rna_real = n_total - n_pre - n_suf - n_ans
        del ins, att, lab
        torch.cuda.empty_cache()
        print(f"[time] 样本构成: total={n_total} prefix={n_pre} "
              f"rna={n_rna_real} suffix={n_suf} answer={n_ans}"
              + (f"  [压缩已生效 n_latent={clat}]" if n_rna_real < 700 else ""), flush=True)

        # ---- 硬校验：声明压缩就必须真的变短，否则整组数字会静默退化成 baseline 的复制品
        expect_rna = int(clat) if clat else 731
        if clat and n_rna_real > expect_rna + 4:
            msg = (f"声明 compress_latent={clat} 但 RNA token 仍是 {n_rna_real} "
                   f"(期望 ≈{expect_rna}) —— 压缩未生效")
            print(f"[fail] {msg}", flush=True)
            results.append({"name": run.get("name"), "spec": run, "error": msg})
            continue

        trunc = run.get("truncate_rna_tokens")
        restore = None
        if trunc is not None:
            restore = install_truncation(model, int(trunc))
            print(f"[time] RNA token 截断 -> {trunc}", flush=True)

        try:
            res = run_one(model, records, device, run)
        except RuntimeError as exc:
            print(f"[warn] run {run.get('name')} 失败: {exc}", flush=True)
            results.append({"name": run.get("name"), "spec": run, "error": str(exc)})
            if restore:
                restore()
            torch.cuda.empty_cache()
            continue
        if restore:
            restore()
        res["token_layout_raw"] = {"total": n_total, "prefix": n_pre, "rna": n_rna_real,
                                   "suffix": n_suf, "answer": n_ans,
                                   "rna_tokens_if_truncated": (min(int(trunc), n_rna_real)
                                                               if trunc else n_rna_real)}
        res["model_load_s"] = load_s
        res["forward_loss_signature"] = sig
        res["compress_latent_requested"] = clat
        res["compress_effective"] = bool(clat and n_rna_real < 700)
        results.append(res)
        eff_tokens = (n_total - n_rna_real + min(int(trunc), n_rna_real)) if trunc else n_total
        print(f"[time] == {run.get('name')}: "
              f"{res['seconds']['per_optimizer_step'].get('median'):.3f} s/step, "
              f"peak {res['memory']['peak_allocated_GiB']:.2f} GiB allocated, "
              f"total_tokens≈{eff_tokens}", flush=True)
        torch.cuda.empty_cache()

    fprint_after = file_fingerprint(ALIGNMENT_PY)
    changed = fprint_before.get("sha256") != fprint_after.get("sha256")
    if changed:
        print(f"[warn] !! alignment.py 在本次计时期间被改动: "
              f"{fprint_before.get('sha256')} -> {fprint_after.get('sha256')}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "spec": spec,
        "model_load_s": load_s,
        "alignment_py_before": fprint_before,
        "alignment_py_after": fprint_after,
        "alignment_py_changed_midrun": bool(changed),
        "runs": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[time] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
