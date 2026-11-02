"""Export RNAQwen-preview as a distributable GGUF (llama.cpp).

Two things a GGUF cannot hold on its own
----------------------------------------
1. **The SOS adapter.**  llama.cpp has no adapter slot inside a GGUF, so
   dw = sum_k s_k B_k A_k has to be folded into the 310 bf16 attention /
   shared_expert Linears.  That merge is measurable, not free: on
   D_qwen36_sos it moves the probe logits by max|dlogit| 2.32 (9.05e-2 of
   the max logit) and flips top-1 on 4 of 128 probe positions
   (runs/RNAQwen-preview/merge_drift.json, same probe).  The numbers travel
   next to the artifact instead of being quietly dropped.

2. **The RNA code channel.**  Training fed the LLM `clen` embeddings looked
   up from `rna_emb` (4096 x 2048) through E2EModel.embed(); the token id at
   those positions was a placeholder (== base_vocab == 248320), so the id
   carried no information -- the vector did.  A GGUF can only get a vector
   from the vocab, so every code gets its own token (id 248320 + c, text
   `<R{c}>`) and rna_emb[c] is baked into that token's input embedding row.
    Rounding it to bf16 is what training did anyway (E2EModel.embed casts with
    rna_vecs.to(e.dtype)), so feeding those tokens reproduces the training
    hidden states.  The code tokens' *output* rows are zeros.

   One wrinkle the converter forced: AutoTokenizer only *names* 248077 of the
   base's 248320 rows, and a token declared past the last named id gets
   renumbered down to max(id)+1.  That is how <R0> first landed on 248077
   while rna_emb[0] sat on 248320, decoupling the token text from the vector.
   Rows 248077..248319 are dead weight (nothing can name them), so they are
   zeroed and 243 <|pad...|> filler tokens hold the codes at 248320 + c --
   the same id the training code used as the placeholder.

   Measured the hard way: filling them with -100.0 does NOT suppress them.
   The logit of a constant row c is c * sum(h), so a "very negative" row is a
   very *large positive* logit whenever sum(h) < 0 -- the smoke test answered
   a plain text prompt with a run of <R...> tokens.  Zeros give logit exactly
   0, which only wins if every real logit is negative; the frontend keeps
   sampling inside the text range anyway.

Phases (separate processes on purpose: `patch` wants the 18 GB GPU model,
`convert` wants the CPU plus most of the free disk)
    patch     base + adapter -> merge -> work/gguf/patch.safetensors
    convert   work/gguf/src (hardlinked shards, patched tokenizer/config)
              -> <out>.gguf, with the patched tensors injected by overriding
              ModelBase.index_tensors
    quantize  llama-quantize.exe  Q8_0 -> Q4_K_M
    verify    read the GGUF back: code-token embeddings vs rna_emb, one merged
              attention tensor vs the patch, and a llama.cpp load + generate

Usage
    python scripts/98_export_gguf.py --phase patch
    python scripts/98_export_gguf.py --phase convert
    python scripts/98_export_gguf.py --phase quantize
    python scripts/98_export_gguf.py --phase verify
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import shutil
import subprocess
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

LLAMA = Path(r"C:\qwen-api\llama-cpp-src\vendor\llama.cpp")
LLAMA_BIN = Path(r"C:\qwen-api\llama.cpp-b10075")

BASE_VOCAB = 248320      # rows of the base embed_tokens / lm_head
SENTINEL = 0.0           # output-row value for code tokens (see the header)
EMBED_KEY = "model.language_model.embed_tokens.weight"
HEAD_KEY = "lm_head.weight"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _mem() -> str:
    try:
        from rnaqwen.qwen36 import commit_gb
        cu, cl = commit_gb()
        return f"commit {cu:.1f}/{cl:.1f} GB"
    except Exception as e:
        return f"(mem 读不到: {e})"


def _write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def _load_97():
    """Reuse 97_'s load path.

    The merge this script bakes into the GGUF has to be *the same* merge whose
    drift was measured -- a second copy of the load code would drift from it.
    """
    spec = importlib.util.spec_from_file_location(
        "rc97", ROOT / "scripts/97_export_preview.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["rc97"] = m
    spec.loader.exec_module(m)
    return m


def _token_vocab_size(base_dir: Path) -> int:
    """How many rows of the base embed matrix the base *tokenizer* can name.

    248320 rows exist, but AutoTokenizer only names 248077 of them.  That
    matters because the converter numbers tokens from AutoTokenizer.vocab and
    renumbers anything declared past the last named id down to max(id)+1.
    """
    from transformers import AutoTokenizer   # caller imports rnaqwen first
    return len(AutoTokenizer.from_pretrained(str(base_dir)).vocab)


# ------------------------------------------------------------------ patch
def phase_patch(a) -> int:
    import torch
    import bitsandbytes as bnb
    from safetensors.torch import save_file

    from rnaqwen.models.sos_lora import SOSLoRALayer, merge_sos_layers

    s97 = _load_97()
    rc = s97.rc

    t0 = time.time()
    ckpt_path = ROOT / a.ckpt
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cargs = ck["args"]
    base_dir, mk_kw = s97.base_spec_from_ckpt_args(cargs)
    lora_r = int(cargs.get("lora_r", 8))
    fsq_path = ROOT / a.fsq
    if not fsq_path.is_file():
        _log(f"  ** FSQ tokenizer 不存在：{fsq_path}")
        return 3
    tkc = torch.load(fsq_path, map_location="cpu", weights_only=False)
    lat = int(ck["fsq"]["enc.weight"].shape[0])
    lev = int(tkc["levels"])
    rc.LATENT, rc.LEVELS, rc.VOCAB = lat, lev, lev ** lat
    _log(f"[cfg] {ckpt_path.name} step={ck.get('step')} base={base_dir.name} "
         f"adapter={mk_kw['adapter']} r={lora_r} codes={lev ** lat}")
    if mk_kw["base"] != "qwen36-35b":
        _log("  ** 这不是 35B 那条路的 checkpoint（本脚本只支持 expert-nf4 那条）")
        return 3
    if not (base_dir / "model.safetensors.index.json").is_file():
        _log(f"  ** 底座目录里没有 safetensors 索引：{base_dir}")
        return 3

    _log("[model] 装载底座 + 适配器（与 97_ 同一条路，约 6-7 min）...")
    model = rc.E2EModel(lora_r, cargs.get("rna_emb", "table"),
                        latent=lat, levels=lev, **mk_kw)
    model.fsq.load_state_dict(ck["fsq"])
    if ck.get("rna_emb") is not None:
        model.rna_emb.weight.data.copy_(
            ck["rna_emb"].to(model.rna_emb.weight.device, model.rna_emb.weight.dtype))
    miss = model.llm.load_state_dict(ck["lora"], strict=False)
    bad = list(getattr(miss, "unexpected_keys", None) or [])
    want = {n for n, p in model.llm.named_parameters() if p.requires_grad}
    got = set(ck["lora"])
    if bad or (want - got) or (got - want):
        _log(f"  ** 适配器与底座对不上 unexpected={bad[:3]} "
             f"缺失={sorted(want - got)[:3]} 多余={sorted(got - want)[:3]}")
        return 3
    model.eval()
    _log(f"[model] 装载完成 {time.time() - t0:.1f}s  适配器 {len(got)} 张量  {_mem()}")

    sos = [(n, m) for n, m in model.llm.named_modules()
           if isinstance(m, SOSLoRALayer)]
    if not sos:
        _log("  ** 一个 SOSLoRALayer 都没有")
        return 3
    badq = [n for n, m in sos
            if isinstance(m.base, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt))]
    if badq:
        _log(f"  ** {len(badq)} 个 SOS 目标是量化层，合并会写坏 nf4 权重：{badq[:3]}")
        return 3
    _log(f"[sos] {len(sos)} 个目标，全是 nn.Linear / dtype "
         f"{sorted({str(m.base.weight.dtype) for _, m in sos})}")
    n_merged = merge_sos_layers(model.llm)
    if n_merged != len(sos):
        _log(f"  ** merge_sos_layers 报 {n_merged}，记录到 {len(sos)}，不一致")
        return 3

    patch: dict = {}
    nbytes = 0
    for name, _m in sos:
        w = model.llm.get_submodule(name).weight.detach().to("cpu", copy=True).contiguous()
        patch[name + ".weight"] = w
        nbytes += w.numel() * w.element_size()
    _log(f"[merge] 折进 {len(patch)} 个权重  {nbytes / 2 ** 30:.2f} GiB  "
         f"（合并代价见 97_ 的 merge_drift.json，同一探针）")

    emb = model.base_emb.weight.detach()
    rna = model.rna_emb.weight.detach()
    n_codes = int(rna.shape[0])
    if n_codes != lev ** lat:
        _log(f"  ** 码表 {n_codes} 行 != levels**latent {lev ** lat}")
        return 3
    if int(emb.shape[0]) != BASE_VOCAB:
        _log(f"  ** 底座 embed {emb.shape[0]} 行，脚本按 {BASE_VOCAB} 算")
        return 3
    if int(rna.shape[1]) != int(emb.shape[1]):
        _log("  ** 码表宽度 != hidden")
        return 3
    code_rows = rna.to(emb.dtype).cpu().contiguous()
    n_tok = _token_vocab_size(base_dir)
    n_fill = BASE_VOCAB - n_tok
    if n_fill < 0:
        _log(f"  ** 底座 tokenizer 能命名 {n_tok} 行 > BASE_VOCAB {BASE_VOCAB}")
        return 3
    patch[EMBED_KEY] = torch.cat(
        [emb[:n_tok].cpu().contiguous(),
         torch.zeros((n_fill, emb.shape[1]), dtype=emb.dtype),
         code_rows], dim=0)
    head = model.llm.lm_head.weight.detach()
    patch[HEAD_KEY] = torch.cat(
        [head[:n_tok].cpu().contiguous(),
         torch.full((n_fill + n_codes, head.shape[1]), SENTINEL,
                    dtype=head.dtype)], dim=0)
    _log(f"[token] 底座 tokenizer 只命名 {n_tok} 行（嵌入矩阵 {emb.shape[0]} 行）；"
         f"{n_tok}..{BASE_VOCAB - 1} 这 {n_fill} 行谁也命不了名，输入置 0、输出也置 0")
    _log(f"[vocab] embed {emb.shape[0]} 行 = 底座 {n_tok} + filler {n_fill} + "
         f"{n_codes} 码；码 token 的输出行 = {SENTINEL}（logit 恒 0；"
         f"-100 那种常数行反而会变成极大的正 logit，实测过）")

    a.work.mkdir(parents=True, exist_ok=True)
    p = a.work / "patch.safetensors"
    save_file(patch, str(p), metadata={"format": "pt",
                                       "producer": "scripts/98_export_gguf.py"})
    man = {
        "name": a.name,
        "ckpt": str(a.ckpt), "step": ck.get("step"),
        "base_dir": str(base_dir), "base_mode": mk_kw["base_mode"], "fsq": str(a.fsq),
        "adapter": mk_kw["adapter"], "r_tot": lora_r, "sos_k": int(mk_kw["sos_k"]),
        "base_vocab": BASE_VOCAB, "n_codes": n_codes,
        "token_base": n_tok, "n_filler": n_fill,
        "embed_rows": BASE_VOCAB + n_codes, "sentinel": SENTINEL,
        "code_token_prefix": "<R{c}>",
        "n_targets": len(sos), "targets": [n for n, _ in sos],
        "patch_bytes": int(sum(v.numel() * v.element_size() for v in patch.values())),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": ("GGUF cannot hold an adapter: these 310 weights are the merged "
                 "(pre-quantization) form. Merging changes the function -- see "
                 "runs/RNAQwen-preview/merge_drift.json. Code tokens 248320+c "
                 "carry rna_emb[c] as their input embedding."),
    }
    _write_json(a.work / "manifest.json", man)
    _log(f"[patch] 写出 {p}  {man['patch_bytes'] / 2 ** 30:.2f} GiB  {_mem()}")
    del model, patch, emb, rna, code_rows, head
    gc.collect()
    try:
        import torch as _t
        _t.cuda.empty_cache()
    except Exception:
        pass
    _log(f"[done] patch  {(time.time() - t0) / 60:.1f} min")
    return 0


# ---------------------------------------------------------------- convert
def build_src(a, base: Path, man: dict) -> Path:
    """A model dir the *converter* can read: hardlinked shards + patched meta.

    Hard links, not copies: the 26 shards are 67 GiB and C: does not have that
    much room.  The only files that actually differ are config.json (vocab_size
    grows by 4096 rows) and tokenizer.json (the 4096 code tokens).
    """
    src = a.work / "src"
    src.mkdir(parents=True, exist_ok=True)
    parts = sorted(base.glob("*.safetensors")) + [base / "model.safetensors.index.json"]
    for f in parts:
        if not f.is_file():
            raise FileNotFoundError(f)
        dst = src / f.name
        if dst.exists():
            if os.path.samefile(dst, f):
                continue
            dst.unlink()
        os.link(f, dst)
    _log(f"  [src] 硬链接 {len(parts)} 个文件 -> {src}")

    cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
    if int(cfg["text_config"].get("vocab_size", man["base_vocab"])) != man["base_vocab"]:
        raise RuntimeError("底座 config 的 vocab_size 与脚本假定的 base_vocab 不同")
    cfg["text_config"]["vocab_size"] = man["embed_rows"]
    _write_json(src / "config.json", cfg)
    for name in ("tokenizer_config.json", "vocab.json", "merges.txt",
                 "chat_template.jinja", "generation_config.json",
                 "preprocessor_config.json"):
        if (base / name).is_file():
            shutil.copyfile(base / name, src / name)

    tj = json.loads((base / "tokenizer.json").read_text(encoding="utf-8"))
    have = {t["content"] for t in tj.get("added_tokens", [])}
    n_tok = int(man["token_base"])
    n_fill = int(man["base_vocab"]) - n_tok
    extra = ([(n_tok + i, f"<|pad{n_tok + i}|>") for i in range(n_fill)]
             + [(man["base_vocab"] + c, f"<R{c}>") for c in range(man["n_codes"])])
    for i, content in extra:
        if content in have:
            raise RuntimeError(f"{content} 已在词表里")
        tj["added_tokens"].append({
            "id": i, "content": content, "single_word": False, "lstrip": False,
            "rstrip": False, "normalized": False, "special": False})
    _write_json(src / "tokenizer.json", tj)

    tcf = src / "tokenizer_config.json"
    tcd = json.loads(tcf.read_text(encoding="utf-8"))
    dec = tcd.get("added_tokens_decoder")
    if isinstance(dec, dict):
        for i, content in extra:
            key = str(i)
            if key in dec:
                raise RuntimeError(f"tokenizer_config 里已经有 id {key}")
            dec[key] = {"content": content, "single_word": False,
                        "lstrip": False, "rstrip": False,
                        "normalized": False, "special": False}
        _write_json(tcf, tcd)

    # Gate.  The converter numbers tokens from AutoTokenizer.vocab and
    # renumbers anything declared past the last named id -- that is exactly how
    # <R0> ended up on 248077 with rna_emb[0] on 248320.  Ask the tokenizer the
    # converter will ask, and refuse to convert if the answer moved.
    from transformers import AutoTokenizer
    av = AutoTokenizer.from_pretrained(str(src)).vocab
    got = {c: av.get(f"<R{c}>") for c in (0, 1, man["n_codes"] - 1)}
    want = {c: man["base_vocab"] + c for c in got}
    if got != want:
        raise RuntimeError(f"码 token 被重新编号：{got} != {want}")
    _log(f"  [src] 码 token id {man['base_vocab']}..{man['embed_rows'] - 1}"
         f"（{man['n_codes']} 个，文本形式 <R0>..<R{man['n_codes'] - 1}>）；"
         f"id {n_tok}..{man['base_vocab'] - 1} 是 {n_fill} 个 <|pad|> filler；"
         f"config.text_config.vocab_size = {man['embed_rows']}")
    return src


def phase_convert(a) -> int:
    # rnaqwen.env.shim must run before anything pulls in torch.distributed:
    # this ROCm Windows torch has no distributed backend, and transformers 5
    # touches torch.distributed.* at import time, so `from transformers import
    # AutoTokenizer` (which the converter needs for the vocab) dies without it.
    import rnaqwen  # noqa: F401
    import torch  # noqa: F401
    from safetensors.torch import load_file

    sys.path.insert(0, str(LLAMA))
    sys.path.insert(1, str(LLAMA / "gguf-py"))
    import gguf
    from conversion import ModelBase, ModelType, get_model_architecture, get_model_class

    man = json.loads((a.work / "manifest.json").read_text(encoding="utf-8"))
    base = Path(man["base_dir"])
    src = build_src(a, base, man)
    patch = load_file(str(a.work / "patch.safetensors"))
    _log(f"[patch] {len(patch)} 张量  "
         f"{sum(v.numel() * v.element_size() for v in patch.values()) / 2 ** 30:.2f} GiB")

    hparams = ModelBase.load_hparams(src, False)
    arch = get_model_architecture(hparams, ModelType.TEXT)
    Base = get_model_class(arch, mmproj=False)
    _log(f"[conv] arch {arch}  vocab_size "
         f"{hparams.get('text_config', {}).get('vocab_size')}  转换器 {LLAMA}")

    class Patched(Base):
        """Same converter, except these tensors come from the patch.

        Everything else is read from the hardlinked shards exactly as before,
        so the only semantics this changes is the 310 merged weights plus the
        two embedding matrices.
        """

        model_arch = Base.model_arch

        def index_tensors(self, remote_hf_model_id=None):
            d = super().index_tensors(remote_hf_model_id)
            # The patch is keyed by HF names; index_tensors() is keyed by the
            # names after filter_tensors() (which strips "language_model.").
            # Run the patch keys through the same filter instead of guessing.
            mapped: list[tuple[str, str]] = []
            missing: list[str] = []
            for k in patch:
                item = type(self).filter_tensors((k, lambda k=k: patch[k]))
                if item is None or item[0] not in d:
                    missing.append(k)
                    continue
                mapped.append((item[0], k))
            if missing:
                raise RuntimeError(
                    f"patch 里 {len(missing)} 个名字在底座里找不到，例如 {missing[:3]}")
            for tname, k in mapped:
                d[tname] = (lambda k=k: patch[k])
            if not getattr(self, "_patch_echoed", False):
                self._patch_echoed = True
                _log(f"[conv] patch 覆盖 {len(mapped)} 个张量，例如 "
                     f"{mapped[0][0]} / {mapped[-1][0]}")
            return d

    out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = Patched(src, gguf.LlamaFileType.MOSTLY_Q8_0, out,
                eager=False, use_temp_file=False, model_name=man["name"])
    m.write()
    _log(f"[conv] 写出 {out}  {out.stat().st_size / 2 ** 30:.2f} GiB  "
         f"{(time.time() - t0) / 60:.1f} min")
    write_assets(out.parent, man)
    _log(f"[done] convert  {_mem()}")
    return 0


FRONTEND = r'''"""Sequence + question -> the exact token ids this GGUF expects.

    python rna_prompt.py --seq AUAAAGGAAGUU --q "What are the first 4 nucleotides?"

The RNA side (ERNIE-RNA encoder + FSQ + the code table) cannot live inside a
GGUF, so it stays in the repo and this script does the translation: sequence ->
codes -> token id base_vocab + code.  The GGUF carries rna_emb[code] as that
token's input embedding row, so llama.cpp sees exactly the vectors training fed
the LLM.
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("RNAQ_REPO_DIR", str(HERE.parent)))
BASE = Path(os.environ.get(
    "RNAQ_QWEN36_DIR", str(REPO / "models" / "Qwen3.6-35B-A3B")
))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import importlib.util

import torch

import rnaqwen  # noqa: F401  must come before transformers/peft

_spec = importlib.util.spec_from_file_location("rc", REPO / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build(seq: str, question: str, max_nt: int = 1024):
    ct = json.loads((HERE / "code_tokens.json").read_text(encoding="utf-8"))
    tkc = torch.load(HERE / "tokenizer.pt", map_location="cpu", weights_only=False)
    lat, lev = int(tkc["latent"]), int(tkc["levels"])
    rc.LATENT, rc.LEVELS, rc.VOCAB = lat, lev, lev ** lat
    fsq = torch.load(HERE / "fsq.pt", map_location="cpu", weights_only=False)
    table = torch.load(HERE / "code_emb.pt", map_location="cpu", weights_only=False)
    enc = rc.FSQCode(768, lat, lev, int(table.shape[1]), pool=4).to(DEV).eval()
    enc.load_state_dict(fsq["state"])
    from transformers import AutoTokenizer
    from rnaqwen.models.encoder import RnaEncoder
    tk = rc.Tok(AutoTokenizer.from_pretrained(str(BASE)))
    rna = RnaEncoder(str(REPO / "models/ernierna"), dtype=torch.float32).to(DEV).eval()
    s = "".join(c for c in seq.strip().upper() if c.isalpha())[:max_nt]
    with torch.no_grad():
        o = rna([s])
        h = o.hidden.float()
        mask = o.mask.squeeze(-1) > 0
        if mask.shape[1] != h.shape[1]:
            mask = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        hh = h[0][mask[0]]
        if hh.shape[0] == len(s) + 2:
            hh = hh[1:-1]
        if hh.shape[0] != len(s):
            raise RuntimeError(f"positions {hh.shape[0]} != len {len(s)}")
        code = enc(hh.to(DEV, dtype=torch.float32))[0].detach().cpu().tolist()
    ids = (tk.sys_ids + tk.user_ids + tk.enc("Q: <RNA>")
           + [ct["base_vocab"] + int(c) for c in code]
           + tk.enc(f"</RNA>{question}")
           + tk.assistant_ids + tk.no_think)
    return ids, len(code)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--post", default="",
                    help="llama-server /completion url; the prompt is sent as token ids")
    ap.add_argument("--n-predict", type=int, default=16)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ids, n = build(a.seq, a.q)
    print(f"codes {n}  tokens {len(ids)}  head {ids[:8]}  tail {ids[-8:]}")
    if a.out:
        Path(a.out).write_text(json.dumps(ids), encoding="utf-8")
    if a.post:
        body = json.dumps({"prompt": ids, "n_predict": a.n_predict,
                           "temperature": 0.0, "cache_prompt": False}).encode()
        req = urllib.request.Request(a.post, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as f:
            r = json.loads(f.read().decode("utf-8"))
        print(r.get("content") or r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def write_assets(out: Path, man: dict) -> None:
    """Everything llama.cpp cannot carry: FSQ, code table, prompt builder."""
    import torch
    ck = torch.load(ROOT / man["ckpt"], map_location="cpu", weights_only=False)
    fsq_rel = man.get("fsq", "checkpoints/fsq_stream/tokenizer.pt")
    tkc_path = ROOT / fsq_rel
    if not tkc_path.is_file():
        raise RuntimeError(f"找不到 FSQ tokenizer：{tkc_path}")
    tkc = torch.load(tkc_path, map_location="cpu", weights_only=False)
    if ck.get("rna_emb") is None:
        raise RuntimeError("checkpoint 里没有 rna_emb，前端恢复不出码表")
    lat, lev = int(ck["fsq"]["enc.weight"].shape[0]), int(tkc["levels"])
    torch.save({"state": ck["fsq"], "latent": lat, "levels": lev}, out / "fsq.pt")
    torch.save(ck["rna_emb"], out / "code_emb.pt")
    shutil.copyfile(tkc_path, out / "tokenizer.pt")
    _write_json(out / "code_tokens.json", {
        "scheme": "token id = base_vocab + code; text form <R{code}>",
        "base_vocab": man["base_vocab"], "n_codes": man["n_codes"],
        "first_id": man["base_vocab"],
        "last_id": man["base_vocab"] + man["n_codes"] - 1,
        "sentinel_output_row": man["sentinel"],
    })
    (out / "rna_prompt.py").write_text(FRONTEND, encoding="utf-8")
    _log(f"[asset] fsq.pt / code_emb.pt / tokenizer.pt / code_tokens.json / "
         f"rna_prompt.py -> {out}")
    del ck


# --------------------------------------------------------------- quantize
def phase_quantize(a) -> int:
    exe = LLAMA_BIN / "llama-quantize.exe"
    if not exe.is_file():
        _log(f"  ** 找不到 {exe}")
        return 3
    if not a.out.is_file():
        _log(f"  ** 输入 GGUF 不存在：{a.out}（先跑 --phase convert）")
        return 3
    cmd = [str(exe), "--allow-requantize", str(a.out), str(a.q4), a.qtype,
           str(a.threads)]
    _log(f"[quant] {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(LLAMA_BIN))
    if r.returncode != 0:
        _log(f"  ** llama-quantize 退出码 {r.returncode}")
        return 3
    _log(f"[quant] {a.q4}  {a.q4.stat().st_size / 2 ** 30:.2f} GiB  "
         f"{(time.time() - t0) / 60:.1f} min")
    return 0


# ----------------------------------------------------------------- verify
def _gguf_tensor(reader, name: str):
    import numpy as np
    import gguf
    t = None
    for cand in reader.tensors:
        if cand.name == name:
            t = cand
            break
    if t is None:
        return None, None
    data = t.data
    tt = t.tensor_type
    if tt == gguf.GGMLQuantizationType.F32:
        a = np.asarray(data, dtype=np.float32)
    elif tt == gguf.GGMLQuantizationType.F16:
        a = np.asarray(data, dtype=np.float16).astype(np.float32)
    elif tt == gguf.GGMLQuantizationType.BF16:
        a = np.asarray(data, dtype=np.uint16).view(np.uint16)
        a = (a.astype(np.uint32) << 16).view(np.float32)
    else:
        a = gguf.quants.dequantize(np.asarray(data), tt).astype(np.float32)
    # GGUF shapes are reversed relative to numpy; dequantize() already returns
    # the numpy layout, so only reshape when the array is still flat.
    want_shape = tuple(reversed(t.shape))
    if a.shape != want_shape:
        a = a.reshape(want_shape)
    return a, tt.name


def phase_verify(a) -> int:
    import numpy as np
    from safetensors import safe_open
    from safetensors.torch import load_file

    sys.path.insert(0, str(LLAMA))
    sys.path.insert(1, str(LLAMA / "gguf-py"))
    import gguf

    man = json.loads((a.work / "manifest.json").read_text(encoding="utf-8"))
    patch = load_file(str(a.work / "patch.safetensors"))
    path = a.path
    if not path.is_file():
        _log(f"  ** 不存在：{path}")
        return 3
    reader = gguf.GGUFReader(str(path))
    _log(f"[verify] {path.name}  {path.stat().st_size / 2 ** 30:.2f} GiB  "
         f"{len(reader.tensors)} 张量")

    nv = len(reader.fields["tokenizer.ggml.tokens"].data)
    if nv != man["embed_rows"]:
        _log(f"  ** 词表 {nv} != embed 行数 {man['embed_rows']}")
        return 4
    _log(f"[verify] 词表 {nv} == embed 行数  ({man['base_vocab']} + "
         f"{man['n_codes']} 码)")

    tm = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35MOE, 41)

    def gname(hf: str):
        return tm.get_name(key=hf.replace("language_model.", ""),
                           try_suffixes=(".weight", ".bias"))

    bad = 0
    emb, tt = _gguf_tensor(reader, "token_embd.weight")
    if emb is None:
        _log("  ** GGUF 里没有 token_embd.weight")
        return 4
    want = patch[EMBED_KEY].float().numpy()
    got = emb[man["base_vocab"]:man["base_vocab"] + man["n_codes"]]
    tbl = want[man["base_vocab"]:]
    d = np.abs(got - tbl)
    spread = float(np.abs(tbl - tbl.mean()).mean())
    _log(f"[verify] 码 token 输入嵌入 ({tt})  max|d| {d.max():.3e}  "
         f"均值|d| {d.mean():.3e}  码表自身离散度 {spread:.3e}")
    # always relative to how much information the table carries: the K-quants
    # are allowed to be much noisier than Q8_0, but not to lose the table.
    if d.mean() > 0.25 * spread:
        _log("  ** 码 token 的嵌入与码表对不上")
        bad += 1

    fill = emb[man["token_base"]:man["base_vocab"]]
    fmax = float(np.abs(fill).max())
    if fmax > 1e-2:
        _log(f"  ** filler 那 {man['n_filler']} 行的输入嵌入不是 0（max|v| {fmax:.3e}）")
        bad += 1

    # The code tokens (and the filler tokens holding their ids in place) must
    # not be able to win the argmax: their output rows are the constant
    # man["sentinel"] (0.0), so the logit is exactly 0.
    outw, o_tt = _gguf_tensor(reader, "output.weight")
    if outw is None:
        _log("  ** GGUF 里没有 output.weight")
        return 4
    tail = outw[man["token_base"]:man["embed_rows"]]
    tmax = float(np.abs(tail).max())
    _log(f"[verify] 码 + filler token 输出行 ({o_tt})  max|v| {tmax:.3e}"
         f"（应等于 sentinel {man['sentinel']}）")
    if tmax > 1e-2:
        _log("  ** 码 token 的输出行不是常数 0，它们可能被采样到")
        bad += 1

    # identity-mapped merged tensors only: linear_attn.* gets a v-head reorder
    # in the converter, so comparing those raw would flag a correct file.
    base = Path(man["base_dir"])
    idx = json.loads((base / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards: dict = {}

    def base_weight(hf: str):
        shard = idx["weight_map"][hf + ".weight"]
        if shard not in shards:
            f = safe_open(str(base / shard), framework="pt", device="cpu")
            shards[shard] = f
        return shards[shard].get_tensor(hf + ".weight").float().numpy()

    cands = [t for t in man["targets"] if t.endswith(".self_attn.q_proj")]
    for hf in cands[:2]:
        gn = gname(hf + ".weight")
        arr, tt = _gguf_tensor(reader, gn)
        if arr is None:
            _log(f"  ** GGUF 里没有 {gn}（{hf}）")
            bad += 1
            continue
        p = patch[hf + ".weight"].float().numpy()
        b = base_weight(hf)
        d_merged = float(np.abs(arr - p).mean())
        d_base = float(np.abs(arr - b).mean())
        d_delta = float(np.abs(p - b).mean())
        _log(f"[verify] {gn} ({tt})  均值|gguf-合并| {d_merged:.3e}  "
             f"均值|gguf-底座| {d_base:.3e}  （合并本身的量级 {d_delta:.3e}）")
        if not d_merged < d_base:
            _log("  ** 合并后的权重没进到 GGUF 里（它离底座更近）")
            bad += 1
    if not cands:
        _log("  ** 没有可逐值核对的 target（全是线性注意力层？）")
        bad += 1

    if bad:
        _log(f"  ** 校验失败 {bad} 项")
        return 4
    _log("[verify] 张量级校验通过")
    return 0


def phase_smoke(a) -> int:
    """Load it in llama.cpp and actually generate something."""
    exe = LLAMA_BIN / "llama-cli.exe"
    if not exe.is_file():
        _log(f"  ** 找不到 {exe}")
        return 3
    # -no-cnv: otherwise llama-cli drops into interactive mode and waits on
    # stdin forever (a 20 GB mmap load takes minutes, so it looks like a hang).
    cmd = [str(exe), "-m", str(a.path), "-ngl", "0", "--no-warmup", "-no-cnv",
           "-n", str(a.smoke_n), "--temp", "0", "-p", a.smoke_prompt]
    _log(f"[smoke] {' '.join(cmd)}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(LLAMA_BIN), capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       stdin=subprocess.DEVNULL, timeout=a.smoke_timeout)
    tail = (r.stdout or "")[-2000:]
    _log(tail)
    if r.returncode != 0:
        _log(f"  **(stderr) {(r.stderr or '')[-1500:]}")
        _log(f"  ** llama-cli 退出码 {r.returncode}")
        return 3
    _log(f"[smoke] ok  {(time.time() - t0) / 60:.1f} min")
    return 0


# ------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["patch", "convert", "assets", "quantize", "verify", "smoke"])
    ap.add_argument("--ckpt", default="runs/e2e_D_qwen36_sos/ckpt/final.pt")
    ap.add_argument("--fsq", default="checkpoints/fsq_stream/tokenizer.pt",
                    help="与新 checkpoint 配套的 FSQ tokenizer")
    ap.add_argument("--work", default="work/gguf")
    ap.add_argument("--out", default="gguf/RNAQwen-preview-Q8_0.gguf")
    ap.add_argument("--q4", default="gguf/RNAQwen-preview-Q4_K_M.gguf")
    ap.add_argument("--qtype", default="Q4_K_M")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--name", default="RNAQwen-preview")
    ap.add_argument("--path", default="", help="verify/smoke 用的 gguf（默认 --q4）")
    ap.add_argument("--smoke-prompt", default="Q: <RNA></RNA>What is RNA? A:")
    ap.add_argument("--smoke-n", type=int, default=16)
    ap.add_argument("--smoke-timeout", type=int, default=2400)
    args = ap.parse_args()
    args.work = ROOT / args.work
    args.out = ROOT / args.out
    args.q4 = ROOT / args.q4
    args.path = Path(args.path) if args.path else args.q4

    if args.phase == "patch":
        return phase_patch(args)
    if args.phase == "convert":
        return phase_convert(args)
    if args.phase == "assets":
        man = json.loads((args.work / "manifest.json").read_text(encoding="utf-8"))
        write_assets(args.out.parent, man)
        return 0
    if args.phase == "quantize":
        return phase_quantize(args)
    if args.phase == "verify":
        return phase_verify(args)
    return phase_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
