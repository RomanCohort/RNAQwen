"""路线 C 短实验：把 RNA 换成可训练的离散码 embedding,模型到底会不会用?

路线 C 的前提：RNA 位置给一串**离散码**,码的 embedding 作为普通 token 参与训练。
这条路能不能成,判据不是 loss —— loss 降下来也可能只是模型学会了忽略 RNA 直接编答案。

判据用 MAPO 式的**逐 token 消融**（与 33_token_modality_relevance.py 同一个仪器）：
    对每个答案 token t:   Δlogp(t) = logp_real(t) - logp_ablated(t)
    ablated = 把码 embedding 全部置零（模型在 RNA 位置看到"什么都没有"）
    **Δlogp < 0 的比例** = RNA 表示在帮倒忙的比例
臂 A（软提示）的基线是 **58.4%**。只有这个数掉下来,才叫模态对齐。

单变量：底座冻结 + 只挂 LoRA(attn) + 一张可训练的码 embedding 表。
没有投影层,没有压缩器。动作全在"码怎么变成向量"这一件事上。

两个对照,防止自欺（方法论规则 4）：
  · 随机码表对照 —— 把训好的表换成随机初始化的表。若 Δ 分布不变,说明模型没从码里学到东西
  · 常数码对照   —— 所有 RNA 位置用同一个码。若 Δ 分布不变,说明码的内容根本不重要

用法:
    python scripts/47_routeC_short_sft.py --smoke     # 冒烟:60 条 RNA / 30 步
    python scripts/47_routeC_short_sft.py             # 正式短实验:1500 条 / 500 步
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rnaqwen  # noqa: F401,E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NT = 1024
LATENT, LEVELS = 12, 2                    # vocab 4096（44_/46_ 验证过的那一档）
VOCAB = LEVELS ** LATENT
ERNIE = ROOT / "models/ernierna"
QWEN = ROOT / "models/Qwen3-8B"
DATA = ROOT / "data/processed/rnaqa_n6_merged.jsonl"
CACHE = ROOT / "data/processed/_routeC_codes.pt"

SYS = ("You are an RNA expert. Given an RNA sequence and a question, "
       "answer accurately and concisely.")


# ---------------------------------------------------------------- FSQ

def fsq(z, levels):
    t = (torch.tanh(z) + 1) / 2
    s = t * (levels - 1)
    k = torch.round(s)
    return (k / (levels - 1) * 2 - 1) + (s - s.detach()), k.long()


class FSQTokenizer(nn.Module):
    """ERNIE-RNA(768) -> 线性降到 latent 维 -> FSQ 量化 -> 取出离散码。"""

    def __init__(self, dim, latent, levels, hidden=256):
        super().__init__()
        self.enc = nn.Linear(dim, latent)
        self.levels = levels
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x):
        q, k = fsq(self.enc(x), self.levels)
        c = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for d in range(k.shape[1]):
            c = c * self.levels + k[:, d]
        return self.dec(q), c


# ---------------------------------------------------------------- data

def load_records(limit_rna: int):
    recs = []
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            s = r.get("sequence") or ""
            if not s or len(s) > MAX_NT:      # 与 48_ load_all 同过滤
                continue
            recs.append(r)
    # 按 uid 的哈希确定性打乱：直接取文件前 N 条可能全是同一类 RNA
    import hashlib
    recs.sort(key=lambda r: hashlib.md5(r["uid"].encode()).hexdigest())
    return recs[:limit_rna]


@torch.no_grad()
def encode_codes(recs, tok: FSQTokenizer, batch: int = 16):
    """返回 {uid: 码的 list}，长度 = 序列长度（丢掉 [CLS]/[SEP]）。"""
    from rnaqwen.models.encoder import RnaEncoder
    enc = RnaEncoder(str(ERNIE), dtype=torch.float32).to(DEV).eval()
    out = {}
    for i in range(0, len(recs), batch):
        chunk = recs[i:i + batch]
        o = enc([r["sequence"] for r in chunk])
        h = o.hidden.float()
        mk = o.mask.squeeze(-1) > 0
        if mk.shape[1] != h.shape[1]:
            mk = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        for j, r in enumerate(chunk):
            L = len(r["sequence"])
            sel = mk[j]
            hh = h[j][sel]
            # ERNIE-RNA 加了 [CLS] 和 [SEP]：位置数 = L + 2，取中间 L 个
            if hh.shape[0] == L + 2:
                hh = hh[1:-1]
            elif hh.shape[0] != L:
                continue
            _, c = tok(hh)
            out[r["uid"]] = c.cpu()
        del o, h, mk
        if (i // batch) % 20 == 0:
            print(f"    [enc] {i + len(chunk)}/{len(recs)}", flush=True)
    del enc
    import gc
    gc.collect(); torch.cuda.empty_cache()
    return out


def train_fsq(recs, steps: int, batch: int = 1024):
    """在训练序列的位置上训 FSQ。位置多、模型小，很快。"""
    from rnaqwen.models.encoder import RnaEncoder
    enc = RnaEncoder(str(ERNIE), dtype=torch.float32).to(DEV).eval()
    xs = []
    with torch.no_grad():
        for i in range(0, len(recs), 16):
            chunk = [r["sequence"] for r in recs[i:i + 16]]
            o = enc(chunk)
            h = o.hidden.float()
            mk = o.mask.squeeze(-1) > 0
            if mk.shape[1] != h.shape[1]:
                mk = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
            for j, s in enumerate(chunk):
                hh = h[j][mk[j]]
                if hh.shape[0] == len(s) + 2:
                    hh = hh[1:-1]
                if hh.shape[0] == len(s):
                    xs.append(hh.cpu())
            del o, h, mk
    del enc
    import gc
    gc.collect(); torch.cuda.empty_cache()

    x = torch.cat(xs).to(DEV)
    print(f"  [fsq] 训练位置 {x.shape[0]}  dim {x.shape[1]}")
    torch.manual_seed(0)
    tok = FSQTokenizer(x.shape[1], LATENT, LEVELS).to(DEV)
    opt = torch.optim.AdamW(tok.parameters(), lr=3e-3)
    for it in range(steps):
        p = torch.randint(0, x.shape[0], (batch,), device=DEV)
        xb = x[p]
        rec, _ = tok(xb)
        loss = F.mse_loss(rec, xb)
        loss.backward(); opt.step(); opt.zero_grad()
        if it % max(1, steps // 8) == 0:
            print(f"    step {it:>5}  mse {float(loss):.4f}", flush=True)
    tok.eval()
    return tok, x


# ---------------------------------------------------------------- model

class CodeQwen(nn.Module):
    """底座冻结 + LoRA；RNA 位置由一张可训练的码 embedding 表提供向量。"""

    def __init__(self, lora_r=8):
        super().__init__()
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        base = AutoModelForCausalLM.from_pretrained(
            str(QWEN), torch_dtype=torch.bfloat16, device_map={"": DEV})
        for p in base.parameters():
            p.requires_grad_(False)
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                         task_type="CAUSAL_LM")
        self.llm = get_peft_model(base, cfg)
        self.base_emb = base.get_input_embeddings()
        dev = self.base_emb.weight.device              # 必须跟着底座走
        d = self.base_emb.weight.shape[1]
        self.rna_emb = nn.Embedding(VOCAB, d).to(device=dev, dtype=torch.float32)
        nn.init.normal_(self.rna_emb.weight, std=0.02)
        self.base_vocab = self.base_emb.weight.shape[0]
        print(f"  [dev] 底座 {dev}  码表 {self.rna_emb.weight.device}")

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        is_rna = ids >= self.base_vocab
        safe = ids.clamp(max=self.base_vocab - 1)
        e = self.base_emb(safe)
        if is_rna.any():
            e = e.clone() if not e.requires_grad else e
            e[is_rna] = self.rna_emb(ids[is_rna] - self.base_vocab).to(e.dtype)
        return e


# ---------------------------------------------------------------- batch

def build_batch(tk, examples, max_len=2048):
    """每条样本：
    <|im_start|>system...<|im_end|><|im_start|>user
    Q: <RNA>{codes}</RNA>{question}<|im_end|><|im_start|>assistant
    A: {answer}<|im_end|>
    只对 assistant 段落算 loss。
    """
    seqs, labs = [], []
    for r, qa, codes in examples:
        a = f" A: {qa['answer']}{tk.eos}"
        ids = (tk.sys_ids + tk.user_ids + tk.encode_plain("Q: <RNA>")
               + [tk.base_vocab + int(c) for c in codes]
               + tk.encode_plain(f"</RNA>{qa['question']}")
               + tk.assistant_ids + tk.encode_plain(a))
        n_head = len(ids) - len(tk.assistant_ids) - len(tk.encode_plain(a))
        lab = [-100] * n_head + ids[n_head:]
        if len(ids) > max_len:
            # 截断会切掉答案 —— 明确报出来，不要静默
            raise RuntimeError(
                f"序列 {len(ids)} > max_len {max_len}（码长 {len(codes)}）。"
                f"评测前必须保证答案完整。")
        seqs.append(ids); labs.append(lab)

    mx = max(len(s) for s in seqs)
    ii = torch.full((len(seqs), mx), tk.pad_id, dtype=torch.long, device=DEV)
    ll = torch.full((len(seqs), mx), -100, dtype=torch.long, device=DEV)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        ii[i, :len(s)] = torch.tensor(s, device=DEV)
        ll[i, :len(l)] = torch.tensor(l, device=DEV)
    return ii, ll


class Tok:
    """把 tokenizer 里需要的片段预先算好，避免每步重算。"""

    def __init__(self, tk):
        self._tk = tk
        self.base_vocab = tk.vocab_size
        self.pad_id = tk.pad_token_id if tk.pad_token_id is not None else tk.eos_token_id
        self.eos = tk.eos_token
        self.im_start = tk("<|im_start|>", add_special_tokens=False).input_ids
        self.im_end = tk("<|im_end|>", add_special_tokens=False).input_ids
        self.sys_ids = self.im_start + tk("system").input_ids + tk("\n").input_ids
        self.user_ids = self.im_end + self.im_start + tk("user").input_ids + tk("\n").input_ids
        self.assistant_ids = self.im_end + self.im_start + tk("assistant").input_ids + tk("\n").input_ids

    def encode_plain(self, s):
        return self._tk(s, add_special_tokens=False).input_ids


# ---------------------------------------------------------------- eval

@torch.no_grad()
def per_token_ablation(model, tk, examples, mode="trained", max_n=40):
    """Δlogp = logp_real - logp_ablated，逐答案 token。返回负值比例等。"""
    model.eval()
    real = model.rna_emb.weight.detach().clone()
    if mode == "zero":
        model.rna_emb.weight.zero_()
    elif mode == "random":
        g = torch.Generator(device="cpu").manual_seed(123)
        model.rna_emb.weight.copy_(torch.randn_like(real.cpu(), generator=g).to(real.device) * 0.02)
    elif mode == "constant":
        model.rna_emb.weight.copy_(real[0].expand_as(real).clone())

    out = []
    for r, qa, codes in examples[:max_n]:
        ids, lab = build_batch(tk, [(r, qa, codes)])
        emb = model.embed(ids)
        lo = model.llm(inputs_embeds=emb).logits.float()[:, :-1]
        tgt = lab[:, 1:]
        m = tgt >= 0
        if m.sum() == 0:
            continue
        lp = F.log_softmax(lo, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        for v in lp[m].tolist():
            out.append(v)

    model.rna_emb.weight.copy_(real)
    return torch.tensor(out)


@torch.no_grad()
def run_ablation(model, tk, ev_ex, max_n=25, seed=123):
    """四路消融一次跑完，返回 {mode: 负值比例 / 均值}。"""
    real = per_token_ablation(model, tk, ev_ex, "trained", max_n=max_n)
    out = {}
    for mode in ("zero", "random", "constant"):
        a = per_token_ablation(model, tk, ev_ex, mode, max_n=max_n)
        n = min(len(real), len(a))
        d = real[:n] - a[:n]
        out[mode] = dict(n=n, mean=float(d.mean()), median=float(d.median()),
                         frac_neg=float((d < 0).float().mean()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="60 条 RNA / 30 步")
    ap.add_argument("--rna", type=int, default=1500)
    ap.add_argument("--eval-rna", type=int, default=60)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--fsq-steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--codes", default="", help="用 48_ 的全量码表 codes.pt，跳过 FSQ 与编码")

    ap.add_argument("--eval-every", type=int, default=0,
                    help="每 N 步跑一次三路消融；0=只在末尾跑。"
                         "看对照差距随码表漂移拉开还是一动不动 —— 那才是判据")
    ap.add_argument("--eval-n", type=int, default=25)
    args = ap.parse_args()
    if args.smoke:
        args.rna, args.eval_rna, args.steps, args.fsq_steps = 60, 12, 30, 300

    tag = args.tag or ("smoke" if args.smoke else f"r{args.rna}_s{args.steps}")
    outdir = ROOT / "runs" / f"routeC_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {outdir}")

    t0 = time.time()
    recs = load_records(args.rna + args.eval_rna)
    train_recs, eval_recs = recs[:-args.eval_rna], recs[-args.eval_rna:]
    print(f"[data] 训练 {len(train_recs)} 条 RNA  评测 {len(eval_recs)} 条")
    print(f"       训练 QA {sum(len(r['qa']) for r in train_recs)}  "
          f"评测 QA {sum(len(r['qa']) for r in eval_recs)}")

    cmap = None
    if args.codes:
        cd = torch.load(args.codes, map_location="cpu", weights_only=False)
        cmap = {k: v.tolist() for k, v in cd["codes"].items()}
        if cd["vocab"] != VOCAB:
            raise SystemExit(f"码表 vocab {cd['vocab']} != 本脚本 VOCAB {VOCAB}")
        print(f"[codes] 载入 {args.codes}  {len(cmap)} 条")

    # --- FSQ ---
    tok = None
    if cmap is None:
        print("\n=== 1. 训练 FSQ 分词器 ===")
        tok, _ = train_fsq(train_recs, args.fsq_steps)
    else:
        print("\n=== 1./2. 跳过：直接用全量码表 ===")

    # --- 编码 ---
    codes_tr = codes_ev = None
    if cmap is not None:
        codes_tr = {r["uid"]: cmap[r["uid"]] for r in train_recs if r["uid"] in cmap}
        codes_ev = {r["uid"]: cmap[r["uid"]] for r in eval_recs if r["uid"] in cmap}
        miss = [r["uid"] for r in train_recs + eval_recs if r["uid"] not in cmap]
        if miss:
            print(f"[warn] {len(miss)} 条 uid 不在码表里，已跳过（例：{miss[:3]}）")
    elif CACHE.exists():
        c = torch.load(CACHE, map_location="cpu", weights_only=False)
        if (c.get("vocab") == VOCAB and len(c.get("train", {})) >= len(train_recs)
                and len(c.get("eval", {})) >= len(eval_recs)):
            codes_tr, codes_ev = c["train"], c["eval"]
            print(f"[cache] {CACHE.name}")
    if codes_tr is None:
        print("\n=== 2. 编码 ===")
        codes_tr = encode_codes(train_recs, tok)
        codes_ev = encode_codes(eval_recs, tok)
        torch.save(dict(vocab=VOCAB, latent=LATENT, levels=LEVELS,
                        train=codes_tr, eval=codes_ev), CACHE)

    def usable(recs, cmap):
        e = []
        for r in recs:
            c = cmap.get(r["uid"])
            if c is None:
                continue
            for qa in (r.get("qa") or []):
                if qa.get("question") and qa.get("answer"):
                    e.append((r, qa, c))
        return e

    tr_ex = usable(train_recs, codes_tr)
    ev_ex = usable(eval_recs, codes_ev)
    ln = [len(c) for c in codes_tr.values()]
    print(f"[codes] 训练样本 {len(tr_ex)}  评测样本 {len(ev_ex)}  "
          f"码长 中位 {sorted(ln)[len(ln)//2]}  max {max(ln)}  用到码字 "
          f"{len(set(int(v) for c in codes_tr.values() for v in c))}/{VOCAB}")
    if not tr_ex:
        raise SystemExit("没有可用训练样本")

    # --- 模型 ---
    print("\n=== 3. 建模型（底座冻结 + LoRA + 可训练码表）===")
    from transformers import AutoTokenizer
    tk = Tok(AutoTokenizer.from_pretrained(str(QWEN)))
    model = CodeQwen(args.lora_r)
    n_lora = sum(p.numel() for n, p in model.llm.named_parameters() if p.requires_grad)
    print(f"  LoRA 可训练 {n_lora:,}   码表 {VOCAB}x{model.rna_emb.weight.shape[1]} "
          f"= {model.rna_emb.weight.numel():,}")

    params = [p for p in model.llm.parameters() if p.requires_grad] + [model.rna_emb.weight]
    opt = torch.optim.AdamW(params, lr=args.lr)
    w_init = model.rna_emb.weight.detach().clone()
    print(f"  码表初始行范数 {float(w_init.norm(dim=1).mean()):.4f}  "
          f"（Qwen 词表行范数 {float(model.base_emb.weight.float().norm(dim=1).mean()):.4f}）")

    # --- 训练 ---
    print(f"\n=== 4. 短 SFT {args.steps} 步 (batch {args.batch}) ===")
    model.train()
    log = []
    for step in range(1, args.steps + 1):
        ex = [tr_ex[torch.randint(0, len(tr_ex), (1,)).item()] for _ in range(args.batch)]
        ids, lab = build_batch(tk, ex)
        emb = model.embed(ids)
        out = model.llm(inputs_embeds=emb, labels=lab)
        out.loss.backward()
        # 必须在 step/zero_grad 之前读梯度
        g = model.rna_emb.weight.grad
        gn = float(g.norm()) if g is not None else -1.0
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            drift = float((model.rna_emb.weight.detach() - w_init).norm()
                          / w_init.norm().clamp(min=1e-9))
            print(f"  step {step:>5}  loss {float(out.loss):.4f}  "
                  f"|grad_code| {gn:.4f}  码表漂移 {100*drift:.2f}%  "
                  f"{time.time()-t0:.0f}s", flush=True)
            log.append(dict(step=step, loss=float(out.loss), grad=gn, drift=drift))
        if args.eval_every and step % args.eval_every == 0:
            model.eval()
            ab = run_ablation(model, tk, ev_ex, max_n=args.eval_n)
            model.train()
            dr = float((model.rna_emb.weight.detach() - w_init).norm()
                       / w_init.norm().clamp(min=1e-9))
            print(f"  [eval@{step}] 漂移 {100*dr:.1f}%  "
                  f"vs零表 {100*ab['zero']['frac_neg']:.1f}%  "
                  f"vs随机 {100*ab['random']['frac_neg']:.1f}%  "
                  f"vs常数码 {100*ab['constant']['frac_neg']:.1f}%", flush=True)
            log.append(dict(step=step, drift=dr, ablation=ab))

    # --- 判据 ---
    print("\n=== 5. 逐 token 消融：RNA 表示到底有没有在帮忙 ===")
    res = {}
    real = per_token_ablation(model, tk, ev_ex, "trained", max_n=40)
    zero = per_token_ablation(model, tk, ev_ex, "zero", max_n=40)
    n = min(len(real), len(zero))
    d = (real[:n] - zero[:n])
    frac_neg = float((d < 0).float().mean())
    res["trained_vs_zero"] = dict(n=n, mean=float(d.mean()), median=float(d.median()),
                                  frac_neg=frac_neg)
    print(f"  答案 token 数 {n}")
    print(f"  Δlogp  均值 {float(d.mean()):+.4f}  中位 {float(d.median()):+.4f}")
    print(f"  **负值比例 {100*frac_neg:.1f}%**   (臂 A 软提示基线 58.4%)")

    rnd = per_token_ablation(model, tk, ev_ex, "random", max_n=40)
    n2 = min(len(real), len(rnd))
    d2 = (real[:n2] - rnd[:n2])
    res["trained_vs_random"] = dict(n=n2, mean=float(d2.mean()),
                                    frac_neg=float((d2 < 0).float().mean()))
    print(f"  对照·随机码表: Δlogp 均值 {float(d2.mean()):+.4f}  "
          f"负值 {100*float((d2<0).float().mean()):.1f}%")

    cst = per_token_ablation(model, tk, ev_ex, "constant", max_n=40)
    n3 = min(len(real), len(cst))
    d3 = (real[:n3] - cst[:n3])
    res["trained_vs_constant"] = dict(n=n3, mean=float(d3.mean()),
                                      frac_neg=float((d3 < 0).float().mean()))
    print(f"  对照·常数码  : Δlogp 均值 {float(d3.mean()):+.4f}  "
          f"负值 {100*float((d3<0).float().mean()):.1f}%")

    print("\n" + "=" * 72)
    print("  怎么读这三个数（关键在对照，不在第一个数）:")
    print("    vs 零表   —— 低 => RNA 位置上「有东西」在起作用")
    print("    vs 随机表 —— 低 => 起作用的是**码的内容**，不是随便一个向量")
    print("    vs 常数码 —— 低 => 起作用的是**码的区分度**，不是一个共享向量")
    print("  后两个 ≈ 50% 的话，模型只是学会了「RNA 位置有个可训练向量」，")
    print("  它根本没在读码 —— 那就是「Qwen 眼里的 RNA 是乱码」。")
    print()
    t = res["trained_vs_zero"]
    r = res["trained_vs_random"]
    c = res["trained_vs_constant"]
    if t["frac_neg"] < 0.45 and r["frac_neg"] < 0.45 and c["frac_neg"] < 0.45:
        print("  ✅ **路线 C 成立。** 三个对照全部低于 50%，码的内容和区分度都在被使用。")
        print("     上全量、加步数。")
    elif t["frac_neg"] < 0.45 and (r["frac_neg"] > 0.47 or c["frac_neg"] > 0.47):
        print("  ❌ **码的内容没被使用。** 零表消融有害，但换成随机表/常数码几乎无差别。")
        print("     模型学到的是「RNA 位置有一个可训练向量」，不是「这个码代表什么」。")
        print("     —— 这和软提示臂是**同一个病**，只是换了个形式。")
        print("     下一步不是加步数，是加复用压力（commitment / 熵正则）再试。")
    elif t["frac_neg"] > 0.55:
        print("  ❌ **连「有东西」都没起作用。** 零表消融都不疼。")
        print("     先查这一步：码表梯度是不是 0、训练步数是不是太少。")
    else:
        print(f"  ⚠️ **不确定。** 零表 {100*t['frac_neg']:.1f}% / 随机表 {100*r['frac_neg']:.1f}% / "
              f"常数码 {100*c['frac_neg']:.1f}%。")
        print("     步数或数据量不够，加大再跑一次，不要据此下结论。")
    print("=" * 72)

    (outdir / "summary.json").write_text(json.dumps(
        dict(args=vars(args), log=log, result=res,
             n_train=len(tr_ex), n_eval=len(ev_ex)), ensure_ascii=False, indent=2),
        encoding="utf-8")
    torch.save(model.rna_emb.weight.detach().cpu(), outdir / "code_emb.pt")
    print(f"\n[save] {outdir/'summary.json'}  {outdir/'code_emb.pt'}   "
          f"用时 {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
