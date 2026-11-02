"""Qwen3.6-35B-A3B 的专家量化装载 —— 换底座的主工程量（PLAN §14.3）。

要解决什么
----------
MoE 的专家是**裸的 3D nn.Parameter**（modeling_qwen3_5_moe.py:744）

    experts.gate_up_proj  [256, 1024, 2048]
    experts.down_proj     [256, 2048, 512]
    41 层合计 33.01 B —— 占 text 的 96.7%

bitsandbytes 的 replace_with_bnb_linear 只替换 nn.Linear，**碰不到它**。
所以 bf16 全量 67 GiB 装不进 63.6 GB，而"4bit 就装得下"那笔账也不成立。

做法（两条同时用）
----------------
1. **Linear 化**：把每个专家拆成独立的 nn.Linear
       gate_up_proj[e]  ->  experts.gate_up.{e}.weight   [1024, 2048]
       down_proj[e]     ->  experts.down.{e}.weight      [2048, 512]
   形状天然对齐（Linear(in=2048, out=1024) 的 weight 就是 [1024, 2048]）。
   共 256 x 41 x 2 = 20,992 个 Linear。forward 按命中的专家派发，
   和官方 Qwen3_5MoeExperts.forward 逐句等价。

2. **流式装载**：模型先在 meta 上建（不占内存），然后**逐层**
   读盘 -> load_state_dict(assign=True) -> 就地 bnb 量化 -> 释放 bf16 副本。
   峰值内存 = 一层的 bf16 副本（约 1.6 GiB），而不是整个模型的 67 GiB。

   这一步能成立的关键是 `load_state_dict(..., assign=True)`：
   它**替换** Parameter 对象而不是 copy_，所以能写进 meta 参数。

其他必须记住的点（都在 PLAN §14.2/§14.7）
-----------------------------------------
· checkpoint 的权重名是 `model.language_model.*`，而文本-only 类期望 `model.*`
  直接比对交集只有 1。**必须走 VLM 类**（只有它认识那个前缀）。
· `mtp.*` 和 `model.visual.*` 一律丢弃。
· `flash-linear-attention` / `causal-conv1d` 都没装，线性注意力走 torch 回退实现。

用法
----
    python scripts/84_qwen36_quant.py --smoke 2      # 只装 2 层，验机制
    python scripts/84_qwen36_quant.py --smoke 0      # 全量
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: F401,E402  必须先于 transformers

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MDL = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_EXPERTS = 256
EXPERT_CHUNK = 32          # 每次处理多少专家的切片（控制瞬时内存）


def fix_meta_buffers(model, tcfg) -> list[str]:
    """修掉建在 meta 上、且不在 checkpoint 里的 buffer。

    `rotary_emb.inv_freq` / `original_inv_freq` 是 __init__ 里算出来的，
    不在权重文件里，所以 strict=False 的 load_state_dict 不会碰它 ——
    它会一直是 meta，最后 .to(DEV) 直接报 "Cannot copy out of meta tensor"。
    这类错误不会在装载阶段暴露，只在第一次前向或 .to() 时炸，很难查。

    修法：整个模块在真实设备上重建（它只有 buffer，没有参数）。
    """
    fixed = []
    stale = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    if not stale:
        return fixed
    text = model.model.language_model
    R = type(text.rotary_emb)
    text.rotary_emb = R(tcfg, device=DEV)
    fixed.append("model.language_model.rotary_emb")
    remain = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    for n in remain:
        if n.startswith("model.visual."):
            continue                      # 视觉塔待会儿整个删掉
        print(f"  ** 仍有 meta buffer 没修：{n}")
    return fixed


def free_gb() -> float:
    """APU 上 mem_get_info() 会撒谎，只有 Win32_OperatingSystem 可信。"""
    import subprocess
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory"],
        capture_output=True, text=True).stdout.strip()
    try:
        return float(out) / 1024 / 1024
    except ValueError:
        return -1.0


# ---------------------------------------------------------------- Linear 化

def make_quant_experts_cls(base_cls):
    """生成一个 drop-in 替换 Qwen3_5MoeExperts 的类。

    保留类名，这样 transformers 里任何 isinstance/名字判断都不受影响。
    """
    from transformers.activations import ACT2FN

    class Qwen3_5MoeExperts(nn.Module):
        """和官方实现逐句等价，只是专家是 per-expert nn.Linear（bnb 能量化）。"""

        def __init__(self, config):
            super().__init__()
            self.num_experts = config.num_experts
            self.hidden_dim = config.hidden_size
            self.intermediate_dim = config.moe_intermediate_size
            self.gate_up = nn.ModuleList([
                nn.Linear(self.hidden_dim, 2 * self.intermediate_dim, bias=False)
                for _ in range(self.num_experts)])
            self.down = nn.ModuleList([
                nn.Linear(self.intermediate_dim, self.hidden_dim, bias=False)
                for _ in range(self.num_experts)])
            self.act_fn = ACT2FN[config.hidden_act]

        def forward(self, hidden_states, top_k_index, top_k_weights):
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index,
                                        num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(
                    expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                e = int(expert_idx[0])
                if e == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[e])
                current_state = hidden_states[token_idx]
                gate, up = self.gate_up[e](current_state).chunk(2, dim=-1)
                cur = self.act_fn(gate) * up
                cur = self.down[e](cur)
                cur = cur * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(
                    0, token_idx, cur.to(final_hidden_states.dtype))
            return final_hidden_states

    Qwen3_5MoeExperts.__name__ = base_cls.__name__
    Qwen3_5MoeExperts.__qualname__ = base_cls.__qualname__
    return Qwen3_5MoeExperts


# ---------------------------------------------------------------- 索引

def build_index() -> tuple[dict[str, str], list[str]]:
    idx = json.loads((MDL / "model.safetensors.index.json").read_text(encoding="utf-8"))
    wmap = idx["weight_map"]
    shards = sorted(set(wmap.values()))
    return wmap, shards


def remap(k: str):
    """checkpoint key -> 我们模型的 key。返回 None 表示丢弃。

    **注意：这里不做任何前缀改写。**
    那个 `model.language_model.` -> `model.` 的重映射，只在装载到
    **文本-only 类**（Qwen3_5MoeForCausalLM）时才需要 —— 这就是 §14.2 里
    "直接比对交集只有 1" 的来源。

    我们装的是 **VLM 类**（Qwen3_5MoeForConditionalGeneration），
    它的 state_dict key 本来就是 model.language_model.layers.*，
    和 checkpoint 逐字相同。多加一步重映射会把对的改错
    （实测：不改是 693/693 命中，改了反而全部 unexpected）。
    """
    if k.startswith("mtp.") or k.startswith("model.visual."):
        return None                        # PLAN §14.2：一律丢弃
    return k                                # 其余原样，包括 lm_head.weight


def layer_fragment(layer: int, reader) -> dict[str, torch.Tensor]:
    """读出一层的全部张量，专家按需拆分。

    key 一律保持 checkpoint 原样（VLM 命名），因为装的就是 VLM 类。
    """
    frag: dict[str, torch.Tensor] = {}
    pre = f"model.language_model.layers.{layer}."
    for k in reader.keys(prefix=pre):
        nk = remap(k)
        if nk is None:
            continue
        short = k[len(pre):]
        t = reader.get(k)
        if short == "mlp.experts.gate_up_proj":
            for e in range(t.shape[0]):
                frag[f"{pre}mlp.experts.gate_up.{e}.weight"] = t[e]
            del t
        elif short == "mlp.experts.down_proj":
            for e in range(t.shape[0]):
                frag[f"{pre}mlp.experts.down.{e}.weight"] = t[e]
            del t
        else:
            frag[nk] = t
    return frag


class ShardReader:
    """按前缀枚举 + 按名取张量，底层用 safetensors 的惰性句柄。"""

    def __init__(self, wmap: dict[str, str], shards: list[str]):
        from safetensors import safe_open
        self.wmap = wmap
        self._h = {}
        self._open = safe_open
        self.shards = shards
        self.names = sorted(wmap)

    def _handle(self, shard: str):
        if shard not in self._h:
            self._h[shard] = self._open(str(MDL / shard), framework="pt",
                                        device="cpu")
        return self._h[shard]

    def keys(self, prefix: str):
        import bisect
        lo = bisect.bisect_left(self.names, prefix)
        out = []
        i = lo
        while i < len(self.names) and self.names[i].startswith(prefix):
            out.append(self.names[i])
            i += 1
        return out

    def get(self, name: str) -> torch.Tensor:
        return self._handle(self.wmap[name]).get_tensor(name)


# ---------------------------------------------------------------- 量化

def to_bnb_linear(lin: nn.Linear, mode: str) -> nn.Module:
    """nn.Linear -> bnb 的量化 Linear。

    直接用 bitsandbytes，不走 transformers.integrations：
    transformers 5.9 的 BitsAndBytesConfig 只剩 quant_method 一个 dataclass 字段，
    replace_with_bnb_linear 的签名也变了（(model, modules_to_not_convert, ...)，
    不再返回替换后的模块）。这层抽象在这个版本上不稳，绕开它。

    **必须让权重在 CPU 上交给 bnb，再 .to(dev) 触发量化。**
    bnb 的 `Int8Params.to()` 里判断是：
        if not is_quantized and device is not None and self.data.device.type == "cpu":
            return self._quantize(device)
    —— **只有「数据还在 CPU 上」时才量化。**
    如果先 `module.to(DEV)` 再换 Linear，权重到 bnb 手上时已经在显卡上，
    这个条件不成立，量化**静默不发生**，权重原样以 bf16 留着。
    实测：int8 走这条路 → `字节/权重 2.000`（就是 bf16），
    而 int4 的 Params4bit 没这个限制，所以侥幸成了 —— 不能靠它。
    """
    dev = lin.weight.device
    w = lin.weight.data.detach().to("cpu").contiguous()   # 先拉回 CPU
    if mode == "int8":
        from bitsandbytes.nn import Int8Params, Linear8bitLt
        new = Linear8bitLt(lin.in_features, lin.out_features,
                           bias=lin.bias is not None,
                           has_fp16_weights=False, threshold=6.0)
        new.weight = Int8Params(w, has_fp16_weights=False, requires_grad=False)
        if lin.bias is not None:
            new.bias = nn.Parameter(lin.bias.data.detach().to("cpu").clone())
    elif mode == "int4":
        from bitsandbytes.nn import Linear4bit, Params4bit
        new = Linear4bit(lin.in_features, lin.out_features,
                         bias=lin.bias is not None,
                         compute_dtype=torch.bfloat16, compress_statistics=True,
                         quant_type="nf4")
        new.weight = Params4bit(w, requires_grad=False, compress_statistics=True,
                                quant_type="nf4")
        if lin.bias is not None:
            new.bias = nn.Parameter(lin.bias.data.detach().to("cpu").clone())
    else:
        return lin

    # 这一步才是真正量化发生的地方（CPU -> cuda 会走 _quantize）
    new = new.to(dev)
    return new


def swap_linears(mod: nn.Module, mode: str, prefix: str,
                 experts_only: bool) -> int:
    """递归把 Linear 换成 bnb 版本。experts_only 时只碰 mlp.experts 下面那些。"""
    n = 0
    for name, child in list(mod.named_children()):
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            is_expert = ".mlp.experts." in full or full.endswith(".mlp.experts")
            if experts_only and not is_expert:
                continue
            if full.endswith("lm_head"):
                continue
            setattr(mod, name, to_bnb_linear(child, mode))
            n += 1
        else:
            n += swap_linears(child, mode, full, experts_only)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=-1,
                    help="前 N 层（默认 -1 = 全量 40 层）。>=0 时只装前 N 层")
    ap.add_argument("--mode", choices=["int8", "int4", "none"], default="int8")
    ap.add_argument("--quant-rest", action="store_true",
                    help="连非专家的 Linear 一起量化（默认只量化专家）")
    ap.add_argument("--forward-test", action="store_true",
                    help="装完后跑一次真实前向（含 inputs_embeds 路径，即 route C 的注入方式）")
    ap.add_argument("--out", default="", help="装完后 save_pretrained 到该目录")
    ap.add_argument("--min-free", type=float, default=4.0,
                    help="空闲物理内存低于这个数就主动中止（不等它崩）")
    args = ap.parse_args()

    print("=" * 78)
    print("Qwen3.6-35B-A3B 流式装载 + 专家量化")
    print("=" * 78)
    print(f"  空闲内存 {free_gb():.1f} GiB")

    from transformers import AutoConfig, BitsAndBytesConfig
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as M

    cfg_vlm = AutoConfig.from_pretrained(str(MDL))
    tcfg = cfg_vlm.text_config
    print(f"  text: hidden {tcfg.hidden_size}  层 {tcfg.num_hidden_layers}  "
          f"专家 {tcfg.num_experts}  top-{tcfg.num_experts_per_tok}  "
          f"vocab {tcfg.vocab_size}")

    # ---- 1. 替换专家类 ----
    orig = M.Qwen3_5MoeExperts
    M.Qwen3_5MoeExperts = make_quant_experts_cls(orig)
    print(f"  [patch] Qwen3_5MoeExperts -> per-expert nn.Linear")

    # ---- 2. 在 meta 上建模型 ----
    t0 = time.time()
    with torch.device("meta"):
        model = M.Qwen3_5MoeForConditionalGeneration(cfg_vlm)
    n_meta = sum(p.numel() for p in model.parameters())
    print(f"  [build] meta 模型 {n_meta/1e9:.3f} B  用时 {time.time()-t0:.0f}s")
    text = model.model.language_model
    nl = len(text.layers)
    print(f"  [build] text.layers {nl}   lm_head {tuple(model.lm_head.weight.shape)}")

    # ---- 3. 量化配置 ----
    print(f"  [quant] mode={args.mode}  只量化专家={not args.quant_rest}")

    # ---- 4. 流式装载 ----
    wmap, shards = build_index()
    reader = ShardReader(wmap, shards)
    print(f"  [index] {len(wmap):,} 张量  {len(shards)} 分片")

    # meta 缓冲的诊断：rotary_emb 的 inv_freq 之类不在 checkpoint 里，
    # 建在 meta 上就永远是 meta，最后 .to(DEV) 会报 "Cannot copy out of meta tensor"。
    meta_bufs = [n for n, b in model.named_buffers() if b.device.type == "meta"]
    print(f"  [meta] 建完后 meta buffer {len(meta_bufs)} 个: {meta_bufs[:6]}")
    fx = fix_meta_buffers(model, tcfg)
    if fx:
        print(f"  [meta] 已重建 {fx}")
    still = [n for n, b in model.named_buffers()
             if b.device.type == "meta" and not n.startswith("model.visual.")]
    if still:
        print(f"  ** 还有 {len(still)} 个 meta buffer 没修，后面 .to() 会炸: {still[:5]}")

    n_layers = nl if args.smoke < 0 else min(args.smoke, nl)
    total_lin = 0
    peak = 0.0

    print(f"  [load] 逐层装载 {n_layers}/{nl} 层")
    for i in range(n_layers):
        ts = time.time()
        frag = layer_fragment(i, reader)
        nkeys = len(frag)
        res = model.load_state_dict(frag, strict=False, assign=True)
        del frag
        gc.collect()
        # 诊断：这一层还有哪些参数是 meta（没被 assign 到）
        mm = [n for n, p in text.layers[i].named_parameters()
              if p.device.type == "meta"]
        mb = [n for n, b in text.layers[i].named_buffers()
              if b.device.type == "meta"]
        if mm:
            print(f"    ** layer {i} 仍有 meta 参数 {len(mm)} 个: {mm[:6]}")
        if mb:
            print(f"    ** layer {i} 仍有 meta buffer: {mb[:6]}")
        unexpected = [k for k in res.unexpected_keys
                      if f"layers.{i}." in k][:4]
        if unexpected:
            print(f"    ** load 报 unexpected: {unexpected}")
        # 量化**先于**搬到显卡：bnb 只在权重还在 CPU 上时才量化（见 to_bnb_linear）
        nq = swap_linears(text.layers[i], args.mode,
                          f"model.language_model.layers.{i}",
                          experts_only=not args.quant_rest)
        text.layers[i].to(DEV)
        total_lin += nq
        gc.collect()
        if i == 0:
            for n, m in text.layers[0].named_modules():
                wgt = getattr(m, "weight", None)
                if wgt is not None and wgt.dtype in (torch.uint8, torch.int8):
                    print(f"    [check] {n}.weight dtype={wgt.dtype} "
                          f"shape={tuple(wgt.shape)}  numel={wgt.numel()}")
                    break
            else:
                print("    ** [check] 第 0 层里找不到 uint8/int8 权重 —— 量化没发生 **")
        # 用 torch 的显存计数器量真实模型占用；系统"空闲内存"含 page cache，会被骗
        gpu_gb = (torch.cuda.memory_allocated() / 2 ** 30
                  if torch.cuda.is_available() else 0.0)
        p = free_gb()
        peak = max(peak, p)
        print(f"    layer {i:>2}  {nkeys:>4} 张量  量化 {nq:>5} 个 Linear  "
              f"{time.time()-ts:>5.1f}s  显存 {gpu_gb:>5.2f} GiB  "
              f"系统空闲 {p:>5.1f} GiB", flush=True)
        if p >= 0 and p < args.min_free:
            print(f"  ** 系统空闲 {p:.1f} GiB < 阈值 {args.min_free} GiB，中止")
            return 4

    # ---- 5. 非层内张量 ----
    for k in ("model.language_model.embed_tokens.weight", "lm_head.weight"):
        if k in wmap:
            nk = remap(k)
            model.load_state_dict({nk: reader.get(k)}, strict=False, assign=True)
            print(f"  [load] {nk}  {tuple(reader.get(k).shape)}")

    # ---- 6. 丢掉视觉塔和 mtp ----
    for attr in ("visual",):
        if hasattr(model.model, attr):
            delattr(model.model, attr)
            print(f"  [drop] model.{attr}")
    if hasattr(model, "mtp"):
        del model.mtp

    print()
    print("=" * 78)
    n_real = sum(p.numel() for p in model.parameters() if p.device.type != "meta")
    n_unset = sum(p.numel() for p in model.parameters() if p.device.type == "meta")
    print(f"  装载 {n_real/1e9:.3f} B   仍为 meta {n_unset/1e9:.3f} B")
    print(f"  量化 Linear {total_lin:,} 个")
    print(f"  最低空闲内存 {peak:.1f} GiB")
    print(f"  用时 {time.time()-t0:.0f}s")

    if n_unset > 0:
        print(f"  ** 还有 {n_unset/1e9:.3f} B 在 meta 上 —— 前向会炸")
        return 3

    # ---- 7. 真实前向 ----
    if args.forward_test:
        print()
        print("=" * 78)
        print("真实前向验证")
        print("=" * 78)
        model.eval()
        text.to(DEV)
        model.lm_head.to(DEV)
        dev_ok = all(p.device.type == DEV.split(":")[0]
                     for p in model.parameters())
        print(f"  全部参数在 {DEV}: {dev_ok}")
        with torch.no_grad():
            # (a) 普通 input_ids 路径
            ids = torch.tensor([[tk_id for tk_id in (248045, 8678, 198, 248046,
                                                     248045, 846, 198)]], device=DEV)
            t1 = time.time()
            o1 = model(input_ids=ids)
            l1 = o1.logits.float()
            print(f"  input_ids    logits {tuple(l1.shape)}  "
                  f"值域 [{l1.min():.2f}, {l1.max():.2f}]  "
                  f"有限 {bool(torch.isfinite(l1).all())}  {time.time()-t1:.1f}s")
            # (b) inputs_embeds 路径 —— route C 就是这么注入的
            emb = model.model.language_model.embed_tokens(ids)
            emb = emb + torch.randn_like(emb) * 0.05      # 假装是码向量
            t2 = time.time()
            o2 = model(inputs_embeds=emb)
            l2 = o2.logits.float()
            print(f"  inputs_embeds logits {tuple(l2.shape)}  "
                  f"值域 [{l2.min():.2f}, {l2.max():.2f}]  "
                  f"有限 {bool(torch.isfinite(l2).all())}  {time.time()-t2:.1f}s")
            # (c) 反向能不能通到输入（训练必须成立）
            emb2 = model.model.language_model.embed_tokens(ids).detach()
            emb2.requires_grad_(True)
            o3 = model(inputs_embeds=emb2)
            o3.logits.float().sum().backward()
            g = emb2.grad
            print(f"  inputs_embeds 反向  grad 有限 {bool(torch.isfinite(g).all())}  "
                  f"|grad|max {g.abs().max():.3e}  非零 {bool(g.abs().sum() > 0)}")
        model.train()

    if args.out:
        model.save_pretrained(args.out)
        print(f"  [save] {args.out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
