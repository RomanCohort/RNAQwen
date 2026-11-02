"""确认评测 prompt 里到底放了什么 —— 一行一行打出来。

起因
----
`64_acc_nothink.py` / `67_acc_dump.py` 的 `gen()` 把 `build()` 返回的 **ids 整个**
喂给了 `generate(inputs_embeds=...)`，而 `build()` 的 ids = head + `" A: {answer}{eos}"`。
`lab` 才是用来算 loss 的。也就是说：

    评测时 prompt 的结尾是 **" A: <金标准答案><|im_end|>"**，模型是从"完整的一轮问答之后"
    接着往下写的。

transformers 的 `generate` 在只给 `inputs_embeds` 时，`input_ids` 初始化为 `(B, 0)`
（`generation/utils.py:694`），**返回的序列只含新生成的 token**，不含 prompt。
于是 `txt.split("A:")[-1]` 拿到的是「模型在金标准答案之后的续写」——
它多半会自己起一轮新的问答再答一遍。这个数不是「第一次作答的准确率」。

这个脚本把两种 prompt 逐字打出来对比：
    leak  : head + " A: {answer}<|im_end|>"   （64_/67_ 实际喂进去的）
    clean : head                              （应该喂的：只到 assistant 头 + 空 think 块）
并各生成一次，原始输出一起打出来。

跑法:
    python scripts/_probe_eval_prompt.py --artifacts runs/exp_14000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

DEV = rc.DEV
HID = ROOT / "data/processed/_fsq_full_pos.pt"
QWEN = ROOT / "models/Qwen3-8B"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="runs/exp_14000")
    ap.add_argument("--data", default="data/processed/rnaqa_seq30_split.jsonl")
    args = ap.parse_args()

    art = ROOT / args.artifacts
    fsq_ck = torch.load(art / "fsq.pt", map_location="cpu", weights_only=False)
    LAT, LEV = int(fsq_ck["latent"]), int(fsq_ck["levels"])
    rc.LATENT, rc.LEVELS, rc.VOCAB = LAT, LEV, LEV ** LAT

    test = []
    with open(ROOT / args.data, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") == "test":
                test.append(r)
    byq = {}
    for r in test:
        byq.setdefault(r["question"], []).append(r)
    r = byq["What are the first 6 nucleotides of this RNA?"][0]

    d = torch.load(HID, map_location="cpu", weights_only=False)
    hid, lens = d["x"], d["lens"]
    allr = rc.load_records(None, sort=False)
    off_of = {x["uid"]: int(d["offsets"][j]) for j, x in enumerate(allr)}
    len_of = {x["uid"]: int(lens[j]) for j, x in enumerate(allr)}

    from transformers import AutoTokenizer
    _tk0 = AutoTokenizer.from_pretrained(str(QWEN))
    tk = rc.Tok(_tk0)
    model = rc.E2EModel(8, "table", pool=4, latent=LAT, levels=LEV)
    tk.base_vocab = model.base_vocab
    model.fsq.load_state_dict(fsq_ck["state"], strict=False)
    model.rna_emb.weight.data.copy_(
        torch.load(art / "code_emb.pt", map_location="cpu").to(
            model.rna_emb.weight.device, model.rna_emb.weight.dtype))
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    set_peft_model_state_dict(model.llm,
                              load_file(str(art / "lora_adapter" / "adapter_model.safetensors")))
    model.eval()
    P = 4

    NO_THINK = _tk0(" thinking\n\n<｜end▁of▁thinking｜>\n\n", add_special_tokens=False).input_ids
    clen = len_of[r["uid"]] // P
    x = hid[off_of[r["uid"]]:off_of[r["uid"]] + clen * P].to(DEV, dtype=torch.float32)
    with torch.no_grad():
        v, _, _, _ = model.rna_vectors(x, tau=LAT / 2)

    head = (tk.sys_ids + tk.user_ids + tk.enc("Q: <RNA>") + [tk.base_vocab] * clen
            + tk.enc(f"</RNA>{r['question']}") + tk.assistant_ids + NO_THINK)
    tail = tk.enc(f" A: {r['answer']}{tk.eos}")

    print(f"样本 uid {r['uid']}  问题 {r['question']!r}  答案 {r['answer']!r}")
    print(f"码长 clen {clen}  head 长度 {len(head)}  answer 段长度 {len(tail)}")
    print(f"\n[head 解码] {_tk0.decode(head, skip_special_tokens=False)[-160:]!r}")
    print(f"[tail 解码] {_tk0.decode(tail, skip_special_tokens=False)!r}")

    for tag, ids in (("leak （64_/67_ 实际喂的）", head + tail), ("clean（应该喂的）", head)):
        ii = torch.tensor([ids], device=DEV)
        with torch.no_grad():
            emb = model.embed(ii, v)
            import torch as _t
            out = model.llm.generate(inputs_embeds=emb, attention_mask=_t.ones_like(ii),
                                     max_new_tokens=24, do_sample=False,
                                     pad_token_id=tk.pad_id)
        gen_no_special = _tk0.decode(out[0], skip_special_tokens=True)
        gen_raw = _tk0.decode(out[0], skip_special_tokens=False)
        pred = gen_no_special.split("A:")[-1].strip() if "A:" in gen_no_special else gen_no_special.strip()
        print(f"\n=== {tag} ===")
        print(f"  prompt 结尾 : {_tk0.decode(ids, skip_special_tokens=True)[-90:]!r}")
        print(f"  新生成 token: {len(out[0])}")
        print(f"  续写(去特殊): {gen_no_special[:160]!r}")
        print(f"  续写(原样)  : {gen_raw[:160]!r}")
        print(f"  评测口径的预测: {pred[:40]!r}   金标准 {r['answer']!r}"
              f"   {'✓' if pred == r['answer'] else '✗'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
