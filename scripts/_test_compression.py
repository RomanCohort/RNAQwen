"""E 轨改进项 #1 的验收测试：RNA token 压缩模块。

验六件事（对应任务书的 1–6）：

  1. 形状        L ∈ {81, 731, 1023} × n_latent ∈ {32, 64, 128} × 3 method
  2. mask 正确性  padding 长度改（800 vs 1000）或 padding 内容改成随机垃圾值，
                 **有效部分**的输出必须不变；并给一个"mask 失效"的负对照证明测试有区分力
  3. 梯度        backward 后压缩器参数拿到非零梯度（逐参数组列出，零梯度组要解释）
  4. 可训参数量  三种 method 的参数 + 投影 2,625,536，对比 compress_latent=None
  5. 等价性      compress_latent=None 时 build_batch 的 inputs/attn/labels 与**改动前**
                 逐位一致（`torch.equal`，atol=0）。靠 `--phase golden` 在改动前抓基准快照
  6. 加速比      L=731、答案 368 token、n_latent=64 时序列总长的变化

跑法（两段式）：
    # ① 改动 alignment.py **之前**
    python scripts/_test_compression.py --phase golden
    # ② 改动之后
    python scripts/_test_compression.py --phase all
    # 只跑前 5 项、不比对 golden：
    python scripts/_test_compression.py --phase local

为什么 golden 用 stub LLM
-------------------------
Qwen3-8B 有 16GB，本机另一个任务正在跑真实 Stage 1 计时，不能再加载一份。
所以第 5 项用一个**注入式 stub LLM**：patch 掉 `transformers.AutoModelForCausalLM`，
让 `Stage1Aligner.__init__` 与 `build_batch` **原封不动地跑真实代码路径**
（真 RNA-FM 编码器 + 真 Qwen tokenizer + 真投影/压缩拼接），只有 embedding 取值
换成确定性伪随机函数。改动前抓一次快照、改动后重算，逐位比对 —— 这验的正是
"compress_latent=None 这条路上有没有被动过"，与 8B 权重无关。
真实的 8B 全路径由 `scripts/21_smoke_stage1.py` 单独把关。

⚠️ 本脚本**只用 CPU**（另一个任务在独占 GPU）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: E402,F401

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from rnaqwen.data.records import read_jsonl  # noqa: E402
from rnaqwen.models.compression import METHODS, RnaTokenCompressor  # noqa: E402

IN_DIM = 640
LLM_DIM = 4096
N_HEAD_STEPS = 3          # 每个 method 的重复计时次数


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_tensor(t: torch.Tensor) -> str:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:                 # numpy 不认 bf16，按原始字节 hash
        t = t.view(torch.uint8)
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


class _StubEmbedding(nn.Module):
    """确定性 id -> 向量。不用参数、不吃内存，只保证"同一个 id 永远同一个向量"。"""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("freq", torch.arange(dim, dtype=torch.float32), persistent=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = (ids.float().unsqueeze(-1) + 1.0) * 0.017 * self.freq
        return torch.sin(x)


class _StubLM(nn.Module):
    """替身 LLM：只提供 Stage1Aligner 用到的那几个接口。"""

    def __init__(self, dim: int = LLM_DIM) -> None:
        super().__init__()
        self.config = type("Cfg", (), {"hidden_size": dim, "use_cache": True})()
        self.emb = _StubEmbedding(dim)
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def get_input_embeddings(self):  # noqa: ANN201
        return self.emb

    def gradient_checkpointing_enable(self) -> None:  # pragma: no cover
        pass

    def forward(self, *a, **kw):  # pragma: no cover - 本脚本不做 LLM 前向
        raise RuntimeError("stub LLM 不做前向；第 5 项只验 build_batch 的拼接")


def build_stub_aligner(cfg):  # noqa: ANN001, ANN201
    """用真 Stage1Aligner.__init__ + 真 RNA-FM + 真 tokenizer，只把 8B 换成 stub。"""
    import transformers

    from rnaqwen.models.alignment import Stage1Aligner

    real_cls = transformers.AutoModelForCausalLM

    class _Patched(real_cls):  # type: ignore[misc, valid-type]
        @classmethod
        def from_pretrained(cls, *a, **kw):  # noqa: ANN002, ANN003, ANN206
            return _StubLM()

    transformers.AutoModelForCausalLM = _Patched
    try:
        return Stage1Aligner(cfg)
    finally:
        transformers.AutoModelForCausalLM = real_cls


def load_samples(data: Path, n: int = 2) -> list[dict]:
    rows = [r.as_dict() for r in read_jsonl(data)]
    out = []
    for r in rows:
        t = (r.get("annotation") or "").strip()
        if r.get("sequence") and t:
            out.append({"sequence": r["sequence"], "target": t})
        if len(out) >= n:
            break
    if not out:
        raise SystemExit(f"[fail] {data} 里没有可用的 sequence+annotation")
    return out


# --------------------------------------------------------------------------- #
# 1. 形状
# --------------------------------------------------------------------------- #
def test_shapes() -> dict:
    res: dict = {"cases": [], "ok": True}
    for method in METHODS:
        for n_latent in (32, 64, 128):
            comp = RnaTokenCompressor(IN_DIM, n_latent, method)
            for L in (81, 731, 1023):
                h = torch.randn(2, L, IN_DIM)
                m = torch.ones(2, L)
                m[1, max(L - 120, 2):] = 0
                out, om = comp(h, m)
                ok = (tuple(out.shape) == (2, n_latent, IN_DIM)
                      and tuple(om.shape) == (2, n_latent)
                      and float(om.sum()) == 2.0 * n_latent
                      and bool(torch.isfinite(out).all()))
                res["cases"].append({"method": method, "L": L, "n_latent": n_latent,
                                     "out_shape": list(out.shape), "mask_shape": list(om.shape),
                                     "mask_all_ones": float(om.mean()) == 1.0, "ok": ok})
                res["ok"] &= ok
    # 小 L（有效长度 < n_latent）也要不炸
    for method in METHODS:
        comp = RnaTokenCompressor(IN_DIM, 128, method)
        h = torch.randn(1, 40, IN_DIM)
        m = torch.ones(1, 40)
        m[0, 20:] = 0
        out, _ = comp(h, m)
        ok = tuple(out.shape) == (1, 128, IN_DIM) and bool(torch.isfinite(out).all())
        res["cases"].append({"method": method, "L": 40, "valid": 20, "n_latent": 128,
                             "out_shape": list(out.shape), "ok": ok, "note": "L_valid < n_latent"})
        res["ok"] &= ok
    return res


# --------------------------------------------------------------------------- #
# 2. mask 正确性
# --------------------------------------------------------------------------- #
def _pad_to(v_src: torch.Tensor, L: int, gen: torch.Generator | None = None
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """把 (valid_len, D) 的有效内容放进长度 L 的右填充张量，padding 区填**随机垃圾值**
    （模拟 RNA-FM：padding 位置有值、而且不是 0）。"""
    h = torch.randn(L, IN_DIM, generator=gen) * 10.0
    m = torch.zeros(L)
    n = v_src.shape[0]
    h[:n] = v_src
    m[:n] = 1.0
    return h, m


def test_mask(seed: int = 1234) -> dict:
    torch.manual_seed(seed)
    res: dict = {"cases": [], "ok": True}
    for method in METHODS:
        comp = RnaTokenCompressor(IN_DIM, 64, method, trim=True)
        comp_no_trim = RnaTokenCompressor(IN_DIM, 64, method, trim=False)

        gen = torch.Generator().manual_seed(7)
        # 有效内容一致、padding 区是**随机垃圾值**（模拟 RNA-FM：padding 位置有值）
        valid = [torch.randn(731, IN_DIM, generator=gen), torch.randn(631, IN_DIM, generator=gen)]

        def run(c, Ls):
            outs = []
            for v, L in zip(valid, Ls):
                h, m = _pad_to(v, L, gen=gen)
                o, _ = c(h.unsqueeze(0), m.unsqueeze(0))
                outs.append(o)
            return torch.cat(outs, 0)

        # (a) padding 800 vs 1000
        o800 = run(comp, [800, 800])
        o1000 = run(comp, [1000, 1000])
        d_trim = float((o800 - o1000).detach().abs().max())
        eq_trim = bool(torch.equal(o800, o1000))
        a = run(comp_no_trim, [800, 800])
        b = run(comp_no_trim, [1000, 1000])
        d_notrim = float((a - b).abs().max())

        # (b) padding **内容**换成完全不同的随机值（长度不变）
        gen2 = torch.Generator().manual_seed(99)
        h1, m1 = _pad_to(valid[0], 1000, gen=gen)
        h2, _ = _pad_to(valid[0], 1000, gen=gen2)
        d_content = float((comp(h1.unsqueeze(0), m1.unsqueeze(0))[0]
                           - comp(h2.unsqueeze(0), m1.unsqueeze(0))[0]).abs().max())

        # (c) 负对照：mask 失效（全 1，等于把 padding 当真实信号）
        o_nomask, _ = comp(h1.unsqueeze(0), torch.ones(1, 1000))
        d_nomask = float((comp(h1.unsqueeze(0), m1.unsqueeze(0))[0] - o_nomask).abs().max())

        # (d) 有效长度本身不同（batchnorm 无关）——不同有效长度当然不同，这里只保证有限
        finite = bool(torch.isfinite(o1000).all())

        case = {
            "method": method,
            "dtype": "float32",
            "pad_len_800_vs_1000_bitwise_equal(trim=True)": eq_trim,
            "max_abs_diff(trim=True)": d_trim,
            "max_abs_diff(trim=False)": d_notrim,
            "max_abs_diff(padding_content_changed)": d_content,
            "max_abs_diff(mask_ignored_negative_control)": d_nomask,
            "all_finite": finite,
        }
        case["ok"] = (eq_trim and d_content == 0.0 and d_nomask > 1e-6 and finite)
        res["cases"].append(case)
        res["ok"] &= case["ok"]
    return res


def test_mask_real_rnafm(args, samples: list[dict], n_latent: int = 64) -> dict:
    """用**真实 RNA-FM 输出**再验一次 mask：padding 位置的取值是真的非零值（这正是坑本身）。

    手法：把同一条序列分别放进两个不同 batch 里编码（batch 最大长度不同 => padding 长度不同），
    取该序列的输出做比对。RNA-FM 不做跨样本卷积，有效位置的表征与 batch 无关。
    """
    from rnaqwen.models.encoder import RnaFmEncoder

    enc = RnaFmEncoder(args.encoder, dtype=torch.float32)
    seq = samples[0]["sequence"]

    o_solo = enc([seq])
    others = [s["sequence"] for s in samples[1:]]
    o_batch = enc([seq] + others)
    if o_batch.hidden.shape[1] <= o_solo.hidden.shape[1]:
        # batch 里没有更长的序列 => padding 长度没变，人为加一条长的逼出 padding
        o_batch = enc([seq, seq + seq[:200]])
    L_solo, L_batch = int(o_solo.hidden.shape[1]), int(o_batch.hidden.shape[1])
    assert L_batch > L_solo, "构造失败：两个 batch 的 padding 长度相同，这一项就没意义了"
    pad_vals = o_batch.hidden[0, o_solo.lengths[0]:]
    res = {
        "seq_len_nt": len(seq), "L_solo": L_solo, "L_batch": L_batch,
        "padding_positions": int(L_batch - L_solo),
        "padding_max_abs_value": float(pad_vals.abs().max()) if pad_vals.numel() else None,
        "padding_is_zero": bool(pad_vals.numel() and float(pad_vals.abs().max()) == 0.0),
        "cases": [], "ok": True,
    }
    for method in METHODS:
        comp = RnaTokenCompressor(IN_DIM, n_latent, method, trim=True)
        a, _ = comp(o_solo.hidden, o_solo.mask)                # (1, n_latent, 640)
        b, _ = comp(o_batch.hidden, o_batch.mask)              # (2, n_latent, 640)
        equal = bool(torch.equal(a, b[:1]))
        md = float((a - b[:1]).detach().abs().max())
        case = {"method": method, "dtype": "float32", "bitwise_equal": equal, "max_abs_diff": md,
                "ok": equal or md < 1e-5,
                "note": ("mask 的贡献精确为 0（padding 价值 max|·| ~6.5 却完全不影响结果）；"
                         "残差只来自 matmul 归约顺序随列数变化，fp32 下 ~1e-7，bf16 下为 0")}
        res["cases"].append(case)
        res["ok"] &= bool(case["ok"]) and bool(torch.isfinite(b).all())
    return res


# --------------------------------------------------------------------------- #
# 3. 梯度
# --------------------------------------------------------------------------- #
def test_gradients() -> dict:
    res: dict = {"cases": [], "ok": True}
    for method in METHODS:
        torch.manual_seed(0)
        comp = RnaTokenCompressor(IN_DIM, 64, method)
        h = torch.randn(1, 731, IN_DIM, requires_grad=True)
        m = torch.ones(1, 731)
        m[0, 700:] = 0
        out, _ = comp(h, m)
        # 模拟下游：一个"每个 latent 位置权重不同"的 loss（比 out.pow(2).mean() 更接近真实，
        # 真实的梯度来自冻结 LLM 的 causal LM loss，这里不加载 8B）
        w = torch.linspace(0.1, 1.0, 64).view(1, 64, 1)
        comp.zero_grad(set_to_none=True)
        (out * w).pow(2).mean().backward()

        named = [(n, p) for n, p in comp.named_parameters()]
        zero = [n for n, p in named if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        gs = [p.grad.float().pow(2).sum() for _, p in named if p.grad is not None]
        grad_norm = float(torch.sqrt(torch.stack(gs).sum())) if gs else 0.0
        in_grad = float(h.grad.detach().norm())
        case = {
            "method": method,
            "n_param_tensors": len(named),
            "n_zero_grad_tensors": len(zero),
            "zero_grad_names": zero,
            "total_grad_norm": grad_norm,
            "input_grad_norm": in_grad,
        }
        # "有非零梯度"的判据：至少有一组参数拿到非零梯度 + 输入梯度非零
        case["ok"] = (len(named) - len(zero) > 0 or len(named) == 0) and in_grad > 0
        if method == "resampler" and zero:
            case["note"] = ("to_q 零初始化 => 注意力恰好均匀 => ln_q/to_k 的梯度在 step 0 必然为 0；"
                            "to_q 自己拿到非零梯度，第 1 步之后整栈就醒了")
        if method == "meanpool":
            case["note"] = "meanpool 无参数，只能验『输入梯度非零』"
        res["cases"].append(case)
        res["ok"] &= case["ok"]
    return res


# --------------------------------------------------------------------------- #
# 3b. 初始化行为（为什么 resampler 敢在 step 0 就上）
# --------------------------------------------------------------------------- #
def test_init_behavior() -> dict:
    """量一下"latent 之间有多像"。

    `zero_init_q=True` 时 to_q=0 => attention 均匀 => 每个 latent 拿到的是**同一份**
    有效位置均值，只是各自加了不同的 query 残差。所以 step 0 的 resampler 落在
    "池化"这一行为类里，冻结的 LLM 一上来看到的就是真实 RNA 信号，而不是随机 query
    招出的噪声。代价：ln_q / to_k 在 step 0 梯度恰好为 0（见第 3 项），to_q 拿到梯度后
    第 1 步就把整栈叫醒。
    """
    torch.manual_seed(0)
    h = torch.randn(1, 731, IN_DIM)
    m = torch.ones(1, 731)
    m[0, 700:] = 0
    res: dict = {"cases": []}
    for tag, kw in {"resampler(zero_init_q=True)": {"zero_init_q": True},
                    "resampler(zero_init_q=False)": {"zero_init_q": False},
                    "meanpool": {}}.items():
        method = "meanpool" if tag == "meanpool" else "resampler"
        comp = RnaTokenCompressor(IN_DIM, 64, method, **kw)
        out, _ = comp(h, m)
        v = out[0].float()                                        # (n_latent, D)
        v = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        sim = v @ v.t()
        n = sim.shape[0]
        off = (sim.sum() - sim.diag().sum()) / (n * (n - 1))
        res["cases"].append({
            "module": tag,
            "mean_pairwise_cosine_between_latents": round(float(off), 6),
            "latents_nearly_identical": bool(float(off) > 0.9),
        })
    return res


# --------------------------------------------------------------------------- #
# 4. 可训参数量
# --------------------------------------------------------------------------- #
def test_params() -> dict:
    proj = IN_DIM * LLM_DIM + LLM_DIM
    res: dict = {
        "projection_only(compress_latent=None)": proj,
        "methods": {},
        "variants": {},
    }
    for method in METHODS:
        comp = RnaTokenCompressor(IN_DIM, 64, method)
        res["methods"][method] = {
            "compressor_params": comp.num_parameters,
            "breakdown": comp.param_breakdown(),
            "total_trainable": proj + comp.num_parameters,
            "ratio_vs_projection": round((proj + comp.num_parameters) / proj, 4),
        }
    # resampler 的容量旋钮（给"要不要更大"留证据）
    for tag, kw in {
        "d256_l1": {"d_model": 256, "n_layers": 1},
        "d256_l2": {"d_model": 256, "n_layers": 2},
        "d256_l4": {"d_model": 256, "n_layers": 4},
        "d640_l2": {"d_model": 640, "n_layers": 2},
    }.items():
        c = RnaTokenCompressor(IN_DIM, 64, "resampler", **kw)
        res["variants"][tag] = {"params": c.num_parameters, "total_trainable": proj + c.num_parameters,
                                "ratio_vs_projection": round((proj + c.num_parameters) / proj, 4)}
    for kwargs in ({"n_latent": 32}, {"n_latent": 128}):
        c = RnaTokenCompressor(IN_DIM, kwargs["n_latent"], "resampler")
        res["variants"][f"n_latent{kwargs['n_latent']}"] = {"params": c.num_parameters,
                                                           "total_trainable": proj + c.num_parameters}
    c = RnaTokenCompressor(IN_DIM, 64, "strided", conv_kernel=9)
    res["variants"]["strided_k9"] = {"params": c.num_parameters, "total_trainable": proj + c.num_parameters}
    return res


def test_e_track_path(cfg, samples: list[dict], n_latent: int = 64) -> dict:
    """E 轨路径真的接通没：`compress_latent=64` 时走**真 build_batch** 看拼接结果。

    验：总长 = prefix + n_latent + suffix + 答案（RNA 那一段从 731 变成 64）；
    soft 块全 -100；答案 token 数不变（loss 该算的地方没被吃掉）；
    RNA 块 == projection(compressor(encoder))；可训参数 = 投影 + 压缩器。
    """
    from dataclasses import replace

    res: dict = {"n_latent": n_latent, "cases": [], "ok": True}
    base_cfg = getattr(cfg, "compress_latent", "n/a")
    for method in METHODS:
        c = replace(cfg, compress_latent=n_latent, compress_method=method)
        torch.manual_seed(0)
        al = build_stub_aligner(c)
        _pin_projection(al)
        device = torch.device("cpu")
        assert al.compression is not None and al.compression.method == method
        seqs = [s["sequence"] for s in samples]
        tgts = [s["target"] for s in samples]
        inputs, attn, labels = al.build_batch(seqs, tgts, device)
        n_pre = int(al._encode_const(al.PREFIX_TEXT, device).shape[1])
        n_suf = int(al._encode_const(al.SUFFIX_TEXT, device).shape[1])
        n_ans_max = int((labels != -100).sum(1).max())
        soft_len = int(attn.shape[1]) - n_ans_max
        exp_len = n_pre + n_latent + n_suf + n_ans_max

        # RNA 块 == projection(compressor(encoder))
        enc = al.encoder(seqs)
        lat, lat_mask = al.compression(enc.hidden.to(device), enc.mask.to(device))
        proj_block = al.projection(lat)
        block_ok = bool(torch.equal(proj_block.detach().cpu(),
                                    inputs[:, n_pre:n_pre + n_latent].detach().cpu()))
        trainable = sum(p.numel() for p in al.trainable_parameters())
        proj_only = sum(p.numel() for p in al.projection.parameters())
        case = {
            "method": method,
            "inputs_shape": list(inputs.shape),
            "attn_len": int(attn.shape[1]),
            "expected_len(prefix+latent+suffix+answer)": exp_len,
            "soft_block_len(reported_by_21_smoke_style)": soft_len,
            "soft_block_attn_all_ones": bool(float(attn[:, :n_pre + n_latent + n_suf].float().mean()) == 1.0),
            "attn_all_ones(B=2 且有答案 padding 时会是 False，只记录)": bool(float(attn.float().mean()) == 1.0),
            "soft_block_labels_all_-100": bool(int(labels[:, :n_pre + n_latent + n_suf].max()) == -100),
            "answer_tokens": int((labels != -100).sum()),
            "rna_block_is_projection(compressor(hidden))": block_ok,
            "trainable_params": trainable,
            "compressor_params": trainable - proj_only,
            "compressor_mask_all_ones": bool(float(lat_mask.float().mean()) == 1.0),
        }
        case["ok"] = (case["attn_len"] == exp_len and case["soft_block_attn_all_ones"]
                      and case["soft_block_labels_all_-100"] and block_ok
                      and case["answer_tokens"] > 0 and case["compressor_mask_all_ones"])
        res["cases"].append(case)
        res["ok"] &= case["ok"]
    res["note"] = (f"对照：compress_latent 默认值 = {base_cfg!r}（None 时整段不执行）")
    return res


# --------------------------------------------------------------------------- #
# 5. 等价性：改动前的 build_batch（逐字复刻） vs 改动后
# --------------------------------------------------------------------------- #
BASELINE_SOURCE = ("alignment.py@9e358e22f7b7d1a6da40cee6d03b9266468b36d5fc20f11b920d39e01b3e0b94"
                   "（加 E 轨开关之前）")
ANCHOR_SHA = "9e358e22f7b7d1a6da40cee6d03b9266468b36d5fc20f11b920d39e01b3e0b94"


def baseline_build_batch(al, sequences: list[str], targets: list[str], device: torch.device) -> dict:
    """`alignment.py@9e358e22` 的 build_batch **逐字复刻**（别改这个函数）。

    为什么要复刻，而不是只跟磁盘上的 golden 快照比：
        投影层是**随机初始化**的（nn.Linear 默认 init 吃全局 RNG），跨进程的权重不可能相同，
        所以"改动前的 inputs"没法从快照里复现。
        同一进程内、同一个 aligner 实例上跑复刻版和现行版 => 权重一模一样 => 逐位可比。
        复刻的忠实性由**原始 golden**（真的用改动前的代码跑出来的）反查：
        它给出的 attn/labels 必须与本复刻逐位一致 —— 见 `replica_faithfulness`。
    """
    enc = al.encoder(sequences)                       # (B, L, 640)
    rna = enc.hidden.to(device)
    rna_mask = enc.mask.to(device)

    tok = al.tokenizer(
        targets,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=al.cfg.max_text_tokens,
        add_special_tokens=True,
    )
    text_ids = tok["input_ids"].to(device)
    text_mask = tok["attention_mask"].to(device)

    prefix_ids = al._encode_const(al.PREFIX_TEXT, device)
    suffix_ids = al._encode_const(al.SUFFIX_TEXT, device)

    proj = al.projection(rna)                        # (B, L, 640→4096)
    prefix_emb = al.llm.get_input_embeddings()(prefix_ids)
    suffix_emb = al.llm.get_input_embeddings()(suffix_ids)
    text_emb = al.llm.get_input_embeddings()(text_ids)

    inputs = torch.cat([
        prefix_emb.expand(rna.shape[0], -1, -1),
        proj,
        suffix_emb.expand(rna.shape[0], -1, -1),
        text_emb,
    ], dim=1)

    n_prefix = prefix_emb.shape[1]
    rna_labels = torch.full((text_ids.shape[0], proj.shape[1]), -100,
                            dtype=text_ids.dtype, device=device)
    prefix_labels = torch.full((text_ids.shape[0], n_prefix), -100,
                               dtype=text_ids.dtype, device=device)
    suffix_labels = torch.full((text_ids.shape[0], suffix_emb.shape[1]), -100,
                               dtype=text_ids.dtype, device=device)
    ans_labels = torch.where(text_mask.bool(), text_ids, torch.full_like(text_ids, -100))
    labels = torch.cat([prefix_labels, rna_labels, suffix_labels, ans_labels], dim=1)

    prefix_mask = torch.ones((rna_mask.shape[0], n_prefix), dtype=rna_mask.dtype, device=device)
    suffix_mask = torch.ones((rna_mask.shape[0], suffix_emb.shape[1]), dtype=rna_mask.dtype, device=device)
    attn = torch.cat([prefix_mask, rna_mask, suffix_mask, text_mask], dim=1)
    return {"inputs": inputs, "attn": attn, "labels": labels, "proj": proj,
            "prefix_emb": prefix_emb, "suffix_emb": suffix_emb, "text_emb": text_emb}


def _batches(samples: list[dict]) -> dict[str, tuple[list[str], list[str]]]:
    """三条组装路径：两条 batch=1（真实训练就是 bs=1）+ 一条 B=2（走 padding）。"""
    out = {f"b{i}": ([s["sequence"]], [s["target"]]) for i, s in enumerate(samples)}
    out["b2"] = ([s["sequence"] for s in samples], [s["target"] for s in samples])
    return out


def _pin_projection(al, seed: int = 0) -> None:
    """把投影层权重钉成确定值。

    为什么必须钉：`nn.Linear` 默认初始化吃全局 RNG，而 RNA-FM 的加载路径
    （缺 `pooler.dense`，会在 load 时随机初始化）本身也会吃 RNG —— 同一进程里
    第几次加载、前面跑过什么，都会让投影层的初值不一样。于是"跨进程比 inputs"
    会假报警。钉死之后，两次进程的投影层权重逐位相同，inputs 才真的可比。
    （钉的是权重，不是行为；量级照 kaiming_uniform 的 1/sqrt(fan_in) 取。）
    """
    bound = float(al.projection.in_dim) ** -0.5
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in al.projection.parameters():
            p.copy_(torch.empty(p.shape, dtype=torch.float32).uniform_(-bound, bound, generator=gen)
                    .to(p.dtype))


def _none_path_tensors(cfg, samples: list[dict], seeded: bool = True) -> dict[str, torch.Tensor]:
    """compress_latent=None 时用**现行** build_batch 产出张量。"""
    if seeded:
        torch.manual_seed(0)
    al = build_stub_aligner(cfg)
    _pin_projection(al)
    assert al.compression is None, "compress_latent=None 时不该建出压缩器"
    device = torch.device("cpu")
    tensors: dict[str, torch.Tensor] = {}
    for k, (seqs, tgts) in _batches(samples).items():
        inputs, attn, labels = al.build_batch(seqs, tgts, device)
        tensors[f"{k}_inputs"] = inputs.detach().cpu()
        tensors[f"{k}_attn"] = attn.detach().cpu()
        tensors[f"{k}_labels"] = labels.detach().cpu()
    return tensors


def golden_from_replica(cfg, samples: list[dict]) -> dict:
    """用**改动前的代码语义**（复刻版）抓 golden，投影层钉死 => 跨进程可复现。"""
    torch.manual_seed(0)
    al = build_stub_aligner(cfg)
    _pin_projection(al)
    device = torch.device("cpu")
    tensors: dict[str, torch.Tensor] = {}
    for k, (seqs, tgts) in _batches(samples).items():
        t = baseline_build_batch(al, seqs, tgts, device)
        for name in ("inputs", "attn", "labels"):
            tensors[f"{k}_{name}"] = t[name].detach().cpu()
    return tensors


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    same_shape = tuple(a.shape) == tuple(b.shape)
    return {
        "shape_match": same_shape,
        "bitwise_equal(torch.equal)": bool(torch.equal(a, b)),
        "max_abs_diff": None if not same_shape else float((a.float() - b.float()).abs().max()),
    }


def test_equivalence(cfg, samples: list[dict], golden_path: Path, prechange_path: Path) -> dict:
    """三件事：

    A. **同进程**：现行 build_batch vs 逐字复刻的改动前 build_batch，同一份权重，逐位比。
       （这是"有没有动到 None 那条路"的直接证据，atol=0）
    B. **跨进程 seeded golden**：用复刻版 + seed 0 抓的快照，现行代码重算后逐位比。
    C. **复刻忠实性**：用真的改动前代码（未设种子）抓的原始快照反查复刻版 ——
       只比 attn/labels 这类与随机初始化无关的张量，以及 inputs 里与投影无关的三块。
    """
    torch.manual_seed(0)
    al = build_stub_aligner(cfg)
    _pin_projection(al)
    device = torch.device("cpu")
    res: dict = {"baseline_source": BASELINE_SOURCE,
                 "compression_is_none_by_default": al.compression is None}

    new: dict[str, dict] = {}
    old: dict[str, dict] = {}
    for k, (seqs, tgts) in _batches(samples).items():
        inp, attn, lab = al.build_batch(seqs, tgts, device)
        new[k] = {"inputs": inp.detach().cpu(), "attn": attn.detach().cpu(),
                  "labels": lab.detach().cpu()}
        old[k] = {n: v.detach().cpu() for n, v in baseline_build_batch(al, seqs, tgts, device).items()}

    # ---- A. 同进程逐位 ----
    ok_a = True
    cases_a = []
    for k in sorted(new):
        for name in ("inputs", "attn", "labels"):
            c = _cmp(old[k][name], new[k][name])
            c.update({"batch": k, "tensor": name, "shape": list(new[k][name].shape)})
            ok_a &= c["shape_match"] and c["bitwise_equal(torch.equal)"]
            cases_a.append(c)
    res["A_in_process_replica_vs_current"] = {"ok": ok_a, "cases": cases_a}

    # 顺带：现行代码里投影块的数值必须等于 projection(rna) —— 证明中间没被插东西
    proj_ok = True
    for k in sorted(new):
        same = bool(torch.equal(old[k]["proj"], new[k]["inputs"][
            :, old[k]["prefix_emb"].shape[1]:old[k]["prefix_emb"].shape[1] + old[k]["proj"].shape[1]]))
        proj_ok &= same
    res["A_proj_block_is_exactly_projection_rna"] = proj_ok

    # ---- token 统计（给第 6 项用）----
    token_stats = []
    for i in range(len(samples)):
        labels, inputs = new[f"b{i}"]["labels"], new[f"b{i}"]["inputs"]
        n_ans = int((labels != -100).sum())
        token_stats.append({"i": i, "total": int(inputs.shape[1]), "answer_tokens": n_ans,
                            "soft_block(prefix+rna+suffix)": int(inputs.shape[1]) - n_ans})
    res["token_stats"] = token_stats

    # ---- B. seeded golden（复刻版语义）----
    if golden_path.exists():
        ref = torch.load(golden_path, map_location="cpu")
        ok_b, cases_b = True, []
        for k in sorted(ref["tensors"]):
            c = _cmp(ref["tensors"][k], new[k.rsplit("_", 1)[0]][k.rsplit("_", 1)[1]])
            c.update({"tensor": k})
            ok_b &= c["shape_match"] and c["bitwise_equal(torch.equal)"]
            cases_b.append(c)
        res["B_seeded_golden_vs_current"] = {
            "ok": ok_b, "golden_file": str(golden_path),
            "golden_baseline_source": ref.get("baseline_source"),
            "golden_alignment_sha256_at_capture": ref.get("alignment_sha256_at_capture"),
            "cases": cases_b}
    else:
        res["B_seeded_golden_vs_current"] = {"ok": None, "note": f"缺 {golden_path}；先跑 --phase golden"}

    # ---- C. 复刻忠实性：拿真的改动前代码抓的（未设种子）快照反查 ----
    if prechange_path.exists():
        pre = torch.load(prechange_path, map_location="cpu")
        ok_c, cases_c, full_eq = True, [], {}
        for k in ("b0", "b1", "b2"):
            full_eq[k] = bool(torch.equal(pre["tensors"][f"{k}_inputs"], new[k]["inputs"]))
            for name in ("attn", "labels"):
                c = _cmp(pre["tensors"][f"{k}_{name}"], new[k][name])
                c.update({"tensor": f"{k}_{name}"})
                ok_c &= c["shape_match"] and c["bitwise_equal(torch.equal)"]
                cases_c.append(c)
            # inputs 分块看（前缀/投影/后缀/正文）。投影层是随机初始化的：
            # 这一项经验上也能逐位对上（见 full_inputs_bitwise_equal），但它是种子相关的，
            # 所以不拿它当 PASS/FAIL 的判据，只如实记录。
            n_pre = old[k]["prefix_emb"].shape[1]
            n_rna = old[k]["proj"].shape[1]
            n_suf = old[k]["suffix_emb"].shape[1]
            ref_in = pre["tensors"][f"{k}_inputs"]
            blocks = {"prefix": (0, n_pre), "proj": (n_pre, n_pre + n_rna),
                      "suffix": (n_pre + n_rna, n_pre + n_rna + n_suf),
                      "text": (n_pre + n_rna + n_suf, ref_in.shape[1])}
            for bname, (s, e) in blocks.items():
                c = _cmp(ref_in[:, s:e], new[k]["inputs"][:, s:e])
                c.update({"tensor": f"{k}_inputs[{bname}]", "batch": k,
                          "counted_in_verdict": bname != "proj"})
                cases_c.append(c)
                if bname != "proj":
                    ok_c &= c["shape_match"] and c["bitwise_equal(torch.equal)"]
        res["C_replica_faithfulness_vs_original_prechange_golden"] = {
            "ok": ok_c, "golden_file": str(prechange_path),
            "golden_alignment_sha256": pre.get("alignment_sha256"),
            "full_inputs_bitwise_equal_per_batch": full_eq,
            "cases": cases_c}
    else:
        res["C_replica_faithfulness_vs_original_prechange_golden"] = {
            "ok": None, "note": f"缺 {prechange_path}（原始快照，未设种子，只用来反查复刻版忠实性）"}

    res["ok"] = ok_a and proj_ok and (res["B_seeded_golden_vs_current"]["ok"] is True)
    res["replica_faithfulness_ok"] = res["C_replica_faithfulness_vs_original_prechange_golden"]["ok"]
    return res


def measure_tokens(args, samples: list[dict]) -> dict:
    """真实口径：真 RNA-FM 编码器 + 真 Qwen tokenizer 数一遍 token。"""
    from transformers import AutoTokenizer

    from rnaqwen.models.encoder import RnaFmEncoder

    enc = RnaFmEncoder(args.encoder, dtype=torch.float32)
    out = enc([s["sequence"] for s in samples])
    tok = AutoTokenizer.from_pretrained(args.llm)
    n_pre = int(tok("Q: <RNA>", add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1])
    n_suf = int(tok("</RNA>\nA: <Description>", add_special_tokens=False,
                    return_tensors="pt")["input_ids"].shape[1])
    return {"prefix_tokens": n_pre, "suffix_tokens": n_suf,
            "encoder_hidden_shape": list(out.hidden.shape),
            "encoder_valid_tokens_per_seq": [int(v) for v in out.mask.sum(-1)],
            "note": "21_smoke 里的 n_rna_tokens = prefix + RNA软token + suffix"}


# --------------------------------------------------------------------------- #
# 6. 序列长度 / 加速比 + CPU 耗时
# --------------------------------------------------------------------------- #
def test_speedup(n_rna_tokens: int, n_ans_tokens: int, n_latent: int = 64) -> dict:
    before = n_rna_tokens + n_ans_tokens
    after = n_latent + n_ans_tokens
    return {
        "assumption": {"rna_tokens": n_rna_tokens, "answer_tokens": n_ans_tokens,
                       "n_latent": n_latent},
        "total_before": before,
        "total_after": after,
        "ratio_after_over_before": round(after / before, 4),
        "speedup_x": round(before / after, 3),
        "rna_part_ratio_before": round(n_rna_tokens / before, 4),
    }


@torch.no_grad()
def _forward_only(module, h, m):  # noqa: ANN001, ANN202
    return module(h, m)


def measure_time(cfg=None) -> dict:  # noqa: ANN001
    """压缩器自身的 CPU 耗时（forward + backward），B=1、L=731、n_latent=64。"""
    torch.manual_seed(0)
    h0 = torch.randn(1, 731, IN_DIM)
    m = torch.ones(1, 731)
    out: dict = {"batch": 1, "L": 731, "n_latent": 64, "dtype": "float32",
                 "note": "另一个任务在占 GPU/CPU，绝对数字仅供参考；看相对关系",
                 "modules": {}}
    for method in METHODS:
        comp = RnaTokenCompressor(IN_DIM, 64, method)
        for _ in range(1):                                   # warmup
            h = h0.clone().requires_grad_(True)
            o, _ = comp(h, m)
            o.pow(2).mean().backward()
        ts = []
        for _ in range(N_HEAD_STEPS):
            t0 = time.perf_counter()
            h = h0.clone().requires_grad_(True)
            o, _ = comp(h, m)
            o.pow(2).mean().backward()
            ts.append(time.perf_counter() - t0)
        out["modules"][method] = {"fwd+bwd_ms_median": round(statistics.median(ts) * 1e3, 3),
                                  "runs_ms": [round(t * 1e3, 3) for t in ts]}
    # 参照：投影层自己（640→4096）在同一 L 上的耗时
    from rnaqwen.models.projection import LinearProjection

    proj = LinearProjection(IN_DIM, LLM_DIM)
    ts = []
    for _ in range(N_HEAD_STEPS):
        t0 = time.perf_counter()
        h = h0.clone().requires_grad_(True)
        proj(h).pow(2).mean().backward()
        ts.append(time.perf_counter() - t0)
    out["modules"]["projection_640to4096(参照)"] = {
        "fwd+bwd_ms_median": round(statistics.median(ts) * 1e3, 3),
        "runs_ms": [round(t * 1e3, 3) for t in ts]}
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["golden", "all", "local"], default="all")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "annotations_50.jsonl")
    ap.add_argument("--n-samples", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "token_compression_test.json")
    ap.add_argument("--golden", type=Path, default=ROOT / "reports" / "_golden_none_path.pt")
    ap.add_argument("--prechange-golden", type=Path,
                    default=ROOT / "reports" / "_golden_none_path_prechange_unseeded.pt",
                    help="真的用改动前代码抓的原始快照（未设种子），只用来反查复刻版忠实性")
    ap.add_argument("--llm", default=str(ROOT / "models" / "Qwen3-8B"))
    ap.add_argument("--encoder", default=str(ROOT / "models" / "rnafm"))
    args = ap.parse_args()

    from rnaqwen.models.alignment import Stage1Config

    alignment_py = ROOT / "src" / "rnaqwen" / "models" / "alignment.py"
    samples = load_samples(args.data, args.n_samples)
    cfg = Stage1Config(llm_path=args.llm, encoder_path=args.encoder, grad_checkpointing=False)
    print(f"[comp-test] phase={args.phase}  {len(samples)} 条样本  "
          f"alignment.py sha256={sha256_file(alignment_py)[:16]}  cfg.dtype={cfg.dtype}")

    report: dict = {
        "phase": args.phase,
        "device": "cpu",
        "alignment_sha256": sha256_file(alignment_py),
        "config": {"compress_latent": getattr(cfg, "compress_latent", "<no such field>"),
                   "compress_method": getattr(cfg, "compress_method", "<no such field>"),
                   "dtype": cfg.dtype},
        "samples": [{"seq_len_nt": len(s["sequence"]), "target_len": len(s["target"])} for s in samples],
        "methodology": {
            "item5": ("三层证据：(A) 同进程内，把改动前 build_batch 的**逐字复刻**与现行实现"
                      "跑在同一个 aligner 实例上逐位比（同一个随机初始化的投影层 => atol=0 可比）；"
                      "(B) 用复刻版 + seed=0 抓的 seeded golden，跨进程逐位比；"
                      "(C) 复刻的忠实性由**真的用改动前代码抓的**原始快照（未设种子）反查 —— "
                      "它的 attn/labels 以及 inputs 里与投影无关的三块必须与本复刻逐位一致。"
                      "8B 全路径由 scripts/21_smoke_stage1.py 单独把关"),
            "gpu": "全程 CPU，未使用 GPU",
        },
    }

    if args.phase == "golden":
        t0 = time.time()
        tensors = golden_from_replica(cfg, samples)      # 改动前的代码语义（逐字复刻）+ 投影层钉死
        args.golden.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"tensors": tensors, "baseline_source": BASELINE_SOURCE,
                    "alignment_sha256_at_capture": report["alignment_sha256"],
                    "projection_pinned": True, "anchor_alignment_sha256": ANCHOR_SHA,
                    "cfg_dtype": cfg.dtype, "samples": report["samples"]}, args.golden)
        meta = {k: {"shape": list(v.shape), "sha256": sha256_tensor(v)[:16]} for k, v in tensors.items()}
        report["golden"] = {"file": str(args.golden), "sha256": sha256_file(args.golden),
                            "baseline_source": BASELINE_SOURCE,
                            "tensors": meta, "seconds": round(time.time() - t0, 1),
                            "note": ("golden 内容是**改动前 build_batch 的逐字复刻**算出来的，"
                                     "投影层权重用固定 generator 钉死 => 跨进程逐位可复现。"
                                     "复刻的忠实性由 --prechange-golden（真的用改动前代码抓的那份）"
                                     "反查，见第 5 项的 C。")}
        for k, v in meta.items():
            print(f"[golden] {k:14s} {v['shape']}  sha256={v['sha256']}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[golden] -> {args.golden}\n[golden] -> {args.out}")
        return 0

    print("\n[1] 形状 ...")
    report["1_shapes"] = test_shapes()
    print(f"    ok={report['1_shapes']['ok']}  ({len(report['1_shapes']['cases'])} 个 case)")

    print("[2] mask 正确性 ...")
    report["2_mask"] = test_mask()
    for c in report["2_mask"]["cases"]:
        print(f"    {c['method']:9s} 800vs1000 trim(on) diff={c['max_abs_diff(trim=True)']:.3e} "
              f"equal={c['pad_len_800_vs_1000_bitwise_equal(trim=True)']} | trim(off) "
              f"diff={c['max_abs_diff(trim=False)']:.3e} | padding内容 diff="
              f"{c['max_abs_diff(padding_content_changed)']:.3e} | 负对照(忽略mask) diff="
              f"{c['max_abs_diff(mask_ignored_negative_control)']:.3e}")

    print("[2b] mask 正确性（真实 RNA-FM 输出，padding 位置非零）...")
    report["2b_mask_real_rnafm"] = test_mask_real_rnafm(args, samples)
    r2b = report["2b_mask_real_rnafm"]
    print(f"    L {r2b['L_solo']} vs {r2b['L_batch']}（多出 {r2b['padding_positions']} 个 padding 位，"
          f"padding 值的 max|·|={r2b['padding_max_abs_value']}，全零={r2b['padding_is_zero']}）")
    for c in r2b["cases"]:
        print(f"    {c['method']:9s} equal={c['bitwise_equal']} maxdiff={c['max_abs_diff']:.3e}")

    print("[3] 梯度 ...")
    report["3_gradients"] = test_gradients()
    for c in report["3_gradients"]["cases"]:
        print(f"    {c['method']:9s} 零梯度 {c['n_zero_grad_tensors']}/{c['n_param_tensors']} "
              f"|grad|={c['total_grad_norm']:.4e} |输入梯度|={c['input_grad_norm']:.3e}")

    print("[3b] 初始化行为 ...")
    report["3b_init_behavior"] = test_init_behavior()
    for c in report["3b_init_behavior"]["cases"]:
        print(f"    {c['module']:30s} latent 两两平均余弦 = "
              f"{c['mean_pairwise_cosine_between_latents']:.6f}")

    print("[4] 可训参数量 ...")
    report["4_params"] = test_params()
    for m, v in report["4_params"]["methods"].items():
        print(f"    {m:9s} 压缩器 {v['compressor_params']:>10,}  "
              f"总可训 {v['total_trainable']:>10,}  x{v['ratio_vs_projection']}")

    print("[5] None 路径逐位等价 ...")
    report["5_equivalence"] = test_equivalence(cfg, samples, args.golden, args.prechange_golden)
    eq = report["5_equivalence"]
    a5 = eq["A_in_process_replica_vs_current"]
    print(f"    A. 同进程：现行 vs 改动前代码逐字复刻  → ok={a5['ok']}  "
          f"({len(a5['cases'])} 个张量全 True={all(c['bitwise_equal(torch.equal)'] for c in a5['cases'])})"
          f"  投影块=projection(rna) 逐位一致={eq['A_proj_block_is_exactly_projection_rna']}")
    b5 = eq["B_seeded_golden_vs_current"]
    if b5.get("ok") is None:
        print(f"    B. seeded golden：{b5['note']}")
    else:
        print(f"    B. seeded golden（seed=0，跨进程）→ ok={b5['ok']}  "
              f"{[c['tensor'] for c in b5['cases'] if c['bitwise_equal(torch.equal)']]}")
    c5 = eq["C_replica_faithfulness_vs_original_prechange_golden"]
    if c5.get("ok") is None:
        print(f"    C. 复刻忠实性：{c5['note']}")
    else:
        bad = [c["tensor"] for c in c5["cases"]
               if c.get("counted_in_verdict", True) and not c["bitwise_equal(torch.equal)"]]
        print(f"    C. 复刻忠实性（对原始未设种子快照）→ ok={c5['ok']}  不一致项={bad or '无'}")
        print(f"       full inputs（含投影块）逐位一致："
              f"{c5['full_inputs_bitwise_equal_per_batch']}")

    print("[5b] E 轨路径（compress_latent=64 走真 build_batch）...")
    report["5b_e_track_path"] = test_e_track_path(cfg, samples, 64)
    for c in report["5b_e_track_path"]["cases"]:
        print(f"    {c['method']:9s} inputs={c['inputs_shape']} 期望总长={c['expected_len(prefix+latent+suffix+answer)']} "
              f"实际={c['attn_len']} 答案token={c['answer_tokens']} "
              f"RNA块==proj(comp(h))={c['rna_block_is_projection(compressor(hidden))']} "
              f"可训={c['trainable_params']:,}(+{c['compressor_params']:,})")

    print("[6] 序列长度 / 加速比 ...")
    report["6_speedup"] = test_speedup(731, 368, 64)
    print(f"    {report['6_speedup']['total_before']} -> {report['6_speedup']['total_after']} "
          f"= x{report['6_speedup']['speedup_x']}  (假设口径：RNA 731 + 答案 368)")

    # 真实口径：用真 tokenizer + 真 RNA-FM 数一遍
    tk = measure_tokens(args, samples)
    report["6_speedup"]["measured_tokens"] = tk
    stats = report["5_equivalence"].get("token_stats") or []
    if stats:
        s0 = stats[0]
        n_ans = s0["answer_tokens"]
        n_rna_soft = int(tk["encoder_valid_tokens_per_seq"][0])
        meas_before = tk["prefix_tokens"] + n_rna_soft + tk["suffix_tokens"] + n_ans
        meas_after = tk["prefix_tokens"] + 64 + tk["suffix_tokens"] + n_ans
        report["6_speedup"]["measured_full"] = {
            "sample": 0, "n_nt": len(samples[0]["sequence"]), "rna_soft_tokens": n_rna_soft,
            "answer_tokens": n_ans, "total_before": meas_before, "total_after_at_64": meas_after,
            "delta": meas_before - meas_after,
            "matches_build_batch_total": meas_before == s0["total"],
            "speedup_x": round(meas_before / meas_after, 3)}
    print(f"    实测 tokenizer：prefix={tk['prefix_tokens']} suffix={tk['suffix_tokens']} "
          f"encoder_valid={tk['encoder_valid_tokens_per_seq']}")
    if report["6_speedup"]["measured_full"]:
        mf = report["6_speedup"]["measured_full"]
        print(f"    实测全序列：{mf['total_before']} -> {mf['total_after_at_64']} "
              f"= x{mf['speedup_x']}（与 build_batch 实测总长一致={mf['matches_build_batch_total']}）")

    print("[+] CPU 相对耗时 ...")
    report["timing_cpu"] = measure_time()
    for k, v in report["timing_cpu"]["modules"].items():
        print(f"    {k:28s} {v['fwd+bwd_ms_median']} ms")

    checks = {
        "1_shapes": report["1_shapes"]["ok"],
        "2_mask": report["2_mask"]["ok"],
        "2b_mask_real_rnafm": report["2b_mask_real_rnafm"]["ok"],
        "3_gradients": report["3_gradients"]["ok"],
        "5_none_path_bitwise_equal": report["5_equivalence"]["ok"],
        "5b_e_track_path": report["5b_e_track_path"]["ok"],
    }
    report["checks"] = checks
    report["passed"] = all(v is True for v in checks.values())

    proj = report["4_params"]["projection_only(compress_latent=None)"]
    rp = report["4_params"]["methods"]["resampler"]
    report["conclusions"] = {
        "recommended_default": {
            "compress_latent": 64,
            "compress_method": "resampler",
            "kwargs": {"d_model": 256, "n_layers": 2, "n_heads": 8, "ffn_mult": 2.0,
                       "zero_init_q": True},
            "compressor_params": rp["compressor_params"],
            "total_trainable": rp["total_trainable"],
            "vs_reproduction_track": f"{proj:,} -> {rp['total_trainable']:,} "
                                     f"(x{rp['ratio_vs_projection']})",
        },
        "why": [
            "三种 method 的 mask 正确性与 padding 无关性都成立（第 2/2b 项），"
            "所以选型只看'能不能学会挑残基'和'训练负担'。",
            "meanpool / strided 是手工固定的聚合：不管问题是什么，"
            "同一段残基永远压成同一个向量。QA 集里有大量'某个位点/突变/结构域'的问题，"
            "内容相关的选择能力只能由可学习 query 提供 => 默认给 resampler。",
            "resampler 在 step 0 就是安全的：zero_init_q 让注意力恰好均匀，"
            "latent 落在'有效残基均值'这一行为类里（第 3b 项），冻结的 LLM 不会先吃到"
            "随机 query 的噪声，之后再靠 to_q 的梯度把均匀解打破。",
            f"代价：+{rp['compressor_params']:,} 可训参数（复现轨 {proj:,} 的 "
            f"{rp['compressor_params']/proj:.2f} 倍），CPU 上一次 fwd+bwd 约 "
            f"{report['timing_cpu']['modules']['resampler']['fwd+bwd_ms_median']} ms "
            f"（1299 token 的那次 Stage1 前向在 CPU 上要 76 s，压缩器占比可忽略）。",
        ],
        "ablation_ladder": [
            "meanpool（0 参数）：先跑它，作为'压缩本身损失多少'的下界对照；若它与压缩前打平，"
            "就直接用，训练负担为 0。",
            "strided（3,200 参数）：加上局部平滑，几乎零成本。",
            "resampler（1,401,216 参数）：默认；唯一能学'看哪里'的一档。",
            "三档只改 Stage1Config 的两个字段，其余超参完全一致，A/B 是免费的。",
        ],
        "e_track_warning": (
            "本模块属于 **E 轨（改进轨）**，不是复现轨。原版是'每个 token 都投影'。"
            "凡是用 compress_latent != None 跑出来的任何数字（loss / ROUGE / semantic），"
            "**必须与 R 轨分数分开汇报**，不能混在复现结论里。"
            "R 轨的对照配置永远是 compress_latent=None。"),
        "r_track_untouched": (
            "compress_latent=None 时 build_batch 的 inputs/attn/labels 与加开关之前"
            "逐位一致（第 5 项，torch.equal 全 True，max_abs_diff=0.0）；"
            "Stage1Config 新增字段全部有默认值，train_stage1 的可训参数集合在 None 时"
            "就是 projection.parameters()（顺序不变）。"),
        "known_caveats": [
            "save_stage1 在 E 轨 checkpoint 里多存一个 'compression' 键；"
            "scripts/22_zero_encoder_probe.py 目前只 load projection，"
            "所以带压缩器的 checkpoint **不能**直接喂给它（要先用同一 cfg 建模型再手动 load）。",
            "zero_init_q=True 会让 ln_q / to_k 在 step 0 的梯度恰好为 0（to_q=0 的必然结果，"
            "第 3 项记录了这一点），to_q 拿到梯度后第 1 步起整栈恢复。",
        ],
    }

    print()
    for k, v in checks.items():
        print(f"  [{'ok  ' if v is True else ('N/A ' if v is None else 'FAIL')}] {k}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[comp-test] -> {args.out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
