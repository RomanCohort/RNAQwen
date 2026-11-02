"""route C 的推理入口 —— 把这套东西从「一堆 checkpoint」变成「能问的东西」。

为什么需要它
------------
在这之前，全部资产只有训练脚本和评测脚本：
    runs/B_seq8/artifacts/      B_seq8（8B，三臂验过 A 80.0%）
    runs/e2e_D_qwen36/ckpt/     35B（在训）
    checkpoints/fsq_full/       FSQ 分词器 + 内容初始化
但 **没有任何一个地方能回答"给一条序列和一个问题"**。
`scripts/30_infer.py` 是老 route B 的（stage-1 投影 + stage-2 LoRA），
和 route C 不是一套东西。

推理链（和训练时完全一致，一环都不能少）
----------------------------------------
    序列
      -> ERNIE-RNA（冻结，768 维/核苷酸）
      -> 4x 压缩器 + FSQ              (fsq.pt)
      -> 离散码 id (0..4095)
      -> 码表查表                      (code_emb.pt)
      -> 填进 <RNA> 占位符的位置
      -> 底座 + LoRA 生成              (lora_adapter/)

用法
----
    # 单条
    python scripts/91_predict.py --artifacts runs/B_seq8/artifacts \
        --seq AUAAAGGAAGUU --q "What are the first 4 nucleotides of this RNA?"

    # 批量（jsonl，每行 {"sequence":..., "question":...}）
    python scripts/91_predict.py --artifacts runs/B_seq8/artifacts --file in.jsonl --out out.jsonl

    # 三臂对照（验这条链是不是真在读信道）
    python scripts/91_predict.py --artifacts runs/B_seq8/artifacts \
        --seq AUAAAGGAAGUU --q "What are the first 4 nucleotides of this RNA?" --arms
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

import rnaqwen  # noqa: F401,E402  必须先于 transformers/peft
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

DEV = rc.DEV
QWEN8 = ROOT / "models/Qwen3-8B"
QWEN36 = Path(r"C:\qwen-api\models\Qwen3.6-35B-A3B")
ERNIE = ROOT / "models/ernierna"


class RouteC:
    """加载一套 route C 权重，提供 ask(sequence, question) -> answer。

    `artifacts` 和 `ckpt` 二选一：
      artifacts  训练产出的成品目录（lora_adapter/ + code_emb.pt + fsq.pt + meta.json）
      ckpt       训练中的 step_*.pt（含 fsq / rna_emb / lora / args）
    """

    def __init__(self, artifacts: str | Path | None = None, ckpt: str | Path | None = None,
                 base: str = "qwen3-8b", base_bits: str = "bf16",
                 qwen36_mode: str = "int4", lora_preset: str = "C",
                 log=print):
        if not (artifacts or ckpt):
            raise SystemExit("必须给 --artifacts 或 --ckpt 之一")
        self.log = log

        # ---- 1. 配置：LATENT/LEVELS 的单一真相来源是 tokenizer.pt ----
        tkc = torch.load(rc.FSQ_CKPT, map_location="cpu", weights_only=False)
        self.LAT, self.LEV = int(tkc["latent"]), int(tkc["levels"])
        rc.LATENT, rc.LEVELS, rc.VOCAB = self.LAT, self.LEV, self.LEV ** self.LAT
        self.POOL = 4
        self.TAU = self.LAT / 2
        self.VOCAB = rc.VOCAB

        # ---- 2. 从 artifacts 或 ckpt 取权重和参数 ----
        self.base = base
        self.base_bits = base_bits
        self.qwen36_mode = qwen36_mode
        self.lora_preset = lora_preset
        lora_r = 8
        fsq_sd = rna_emb = lora_sd = None
        lora_dir = None
        if artifacts:
            art = Path(artifacts)
            if not art.is_absolute():
                art = ROOT / art
            meta = json.loads((art / "meta.json").read_text(encoding="utf-8"))
            fsq_sd = torch.load(art / "fsq.pt", map_location="cpu",
                                weights_only=False)["state"]
            rna_emb = torch.load(art / "code_emb.pt", map_location="cpu",
                                 weights_only=False)
            lora_dir = art / "lora_adapter"
            log(f"[load] artifacts {art.name}  vocab {meta.get('vocab')}  "
                f"latent {meta.get('latent')}")
        else:
            ck = torch.load(ROOT / ckpt if not Path(ckpt).is_absolute() else ckpt,
                            map_location="cpu", weights_only=False)
            cargs = ck.get("args", {})
            fsq_sd = ck["fsq"]
            rna_emb = ck["rna_emb"]
            lora_sd = ck["lora"]
            lora_r = int(cargs.get("lora_r", 8))
            self.POOL = int(cargs.get("pool", 4))
            log(f"[load] ckpt step {ck.get('step')}  lora_r {lora_r}  pool {self.POOL}")

        # ---- 3. tokenizer（必须跟底座走，写死会在换底座时静默出错）----
        from transformers import AutoTokenizer
        base_dir = QWEN36 if base == "qwen36-35b" else QWEN8
        tk0 = AutoTokenizer.from_pretrained(str(base_dir))
        self.tk = rc.Tok(tk0)
        log(f"[tok] {base_dir.name}  vocab {tk0.vocab_size}  "
            f"no_think {len(self.tk.no_think)} token")

        # ---- 4. 建模型并载入权重 ----
        log("[model] 建底座 + LoRA + 码表 …")
        self.model = rc.E2EModel(lora_r, "table", latent=self.LAT, levels=self.LEV,
                                 pool=self.POOL, base=base, base_mode=qwen36_mode,
                                 lora_preset=lora_preset, base_bits=base_bits)
        self.tk.base_vocab = self.model.base_vocab
        self.model.fsq.load_state_dict(fsq_sd, strict=False)
        if rna_emb is not None:
            self.model.rna_emb.weight.data.copy_(
                rna_emb.to(self.model.rna_emb.weight.device,
                           self.model.rna_emb.weight.dtype))
        if lora_dir is not None:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            sd = load_file(str(lora_dir / "adapter_model.safetensors"))
            r = set_peft_model_state_dict(self.model.llm, sd)
            bad = list(getattr(r, "unexpected_keys", None) or [])
            if bad:
                raise RuntimeError(f"LoRA 没挂上，未匹配 {bad[:4]}")
            log(f"[lora] {len(sd)} 个张量已载入")
        elif lora_sd:
            miss = self.model.llm.load_state_dict(lora_sd, strict=False)
            bad = list(getattr(miss, "unexpected_keys", None) or [])
            if bad:
                raise RuntimeError(f"LoRA 没挂上，未匹配 {bad[:4]}")
            log(f"[lora] {len(lora_sd)} 个张量已载入")
        self.model.eval()

        # ---- 5. 序列编码器（ERNIE-RNA）----
        from rnaqwen.models.encoder import RnaEncoder
        self.enc = RnaEncoder(str(ERNIE), dtype=torch.float32).to(DEV).eval()
        log("[ernie] 编码器就绪")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, sequence: str):
        """序列 -> (码向量 (n,d), 码 id (n,), 实际用到的位置数)。

        **注意尾部截断**：FSQCode 里 `x = x[:n*pool]`，位置数不是 pool 的倍数时
        末尾会被丢掉。所以这里返回的 n 是压缩后的码数，
        提示里也要写 n 个占位符，两边才对得上。
        """
        s = "".join(c for c in sequence.strip().upper() if c.isalpha())
        o = self.enc([s])
        h = o.hidden.float()
        mk = o.mask.squeeze(-1) > 0
        if mk.shape[1] != h.shape[1]:
            mk = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        hh = h[0][mk[0]]
        if hh.shape[0] == len(s) + 2:      # [CLS] ... [SEP]
            hh = hh[1:-1]
        if hh.shape[0] != len(s):
            raise RuntimeError(f"编码位置数 {hh.shape[0]} != 序列长度 {len(s)}")
        hh = hh.to(DEV, dtype=torch.float32)
        v, code, _, _ = self.model.rna_vectors(hh, tau=self.TAU)
        return v, code, hh.shape[0]

    def _prompt_ids(self, n: int, question: str):
        tk = self.tk
        return (tk.sys_ids + tk.user_ids + tk.enc("Q: <RNA>")
                + [tk.base_vocab] * n
                + tk.enc(f"</RNA>{question}")
                + tk.assistant_ids + tk.no_think)

    @torch.no_grad()
    def ask(self, sequence: str, question: str, max_new: int = 16,
            arm: str = "A", other: str | None = None) -> dict:
        """arm: A 真实码 / C 码向量置零 / B 换一条序列（other 给那条）"""
        src = other if (arm == "B" and other) else sequence
        v, code, npos = self.encode(src)
        if arm == "C":
            v = torch.zeros_like(v)
        ids = self._prompt_ids(v.shape[0], question)
        ii = torch.tensor([ids], device=DEV)
        emb = self.model.embed(ii, v)
        out = self.model.llm.generate(
            inputs_embeds=emb, attention_mask=torch.ones_like(ii),
            max_new_tokens=max_new, do_sample=False, pad_token_id=self.tk.pad_id)
        txt = self.tk._tk.decode(out[0], skip_special_tokens=True).strip()
        # 模型学到要输出 " A: " 前缀（训练目标的一部分），推理时必须剥掉
        if "A:" in txt:
            txt = txt.split("A:")[-1].strip()
        return dict(answer=txt, n_codes=int(v.shape[0]), positions=npos, arm=arm)


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--base", choices=["qwen3-8b", "qwen36-35b"], default="qwen3-8b")
    ap.add_argument("--base-bits", choices=["bf16", "int8", "int4"], default="bf16")
    ap.add_argument("--qwen36-mode", choices=["int4", "int8"], default="int4")
    ap.add_argument("--lora-preset", default="C")
    ap.add_argument("--seq", default="")
    ap.add_argument("--q", default="")
    ap.add_argument("--file", default="", help="jsonl，每行 {sequence, question}")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--arms", action="store_true", help="同时跑 A/B/C 三臂")
    args = ap.parse_args()

    p = RouteC(artifacts=args.artifacts or None, ckpt=args.ckpt or None,
               base=args.base, base_bits=args.base_bits,
               qwen36_mode=args.qwen36_mode, lora_preset=args.lora_preset)

    if args.seq and args.q:
        if args.arms:
            r = p.ask(args.seq, args.q, args.max_new, "A")
            c = p.ask(args.seq, args.q, args.max_new, "C")
            b = p.ask(args.seq, args.q, args.max_new, "B",
                      other="ACGU" * 20)
            print(f"\n序列 {args.seq[:40]}  码数 {r['n_codes']}  位置 {r['positions']}")
            print(f"问题 {args.q}")
            for tag, x in (("A 真实码", r), ("B 换序列", b), ("C 置零  ", c)):
                print(f"  {tag}  {x['answer']!r}")
            if r["answer"] == c["answer"]:
                print("  ** A 和 C 相同 —— 这条问题上信道没起作用 **")
        else:
            r = p.ask(args.seq, args.q, args.max_new)
            print(f"\n序列 {args.seq[:60]}")
            print(f"问题 {args.q}")
            print(f"答案 {r['answer']!r}   (码数 {r['n_codes']}，位置 {r['positions']})")
        return 0

    if args.file:
        rows = [json.loads(l) for l in open(args.file, encoding="utf-8") if l.strip()]
        fo = open(args.out, "w", encoding="utf-8") if args.out else None
        for i, r in enumerate(rows):
            got = p.ask(r["sequence"], r["question"], args.max_new)
            got["gold"] = r.get("answer", "")
            got["sequence"] = r["sequence"]
            got["question"] = r["question"]
            line = json.dumps(got, ensure_ascii=False)
            if fo:
                fo.write(line + "\n")
            else:
                print(line)
            if (i + 1) % 20 == 0:
                print(f"  {i+1}/{len(rows)}", file=sys.stderr, flush=True)
        if fo:
            fo.close()
            print(f"[out] {args.out}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
