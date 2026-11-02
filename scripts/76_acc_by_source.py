"""job ②：合并数据集 (rnaqa_merged) 上**按来源拆分**的三臂准确率。

为什么必须拆开看
----------------
rnaqa_merged = seq8 + freetext + rfam，三个来源的任务性质不同：

  seq8      8 个模板，全部经过 code_answer_probe >= 0.446 筛选 —— 答案**在**码信道里
  rfam      "Which Rfam family" —— 探针 14.95%（地板 0.05%），有信号但远弱于 seq8
  freetext  答案平均 229 字符（最长 465），来源是文献/LLM 描述 —— 码信道**装不下**

总准确率是被这三者加权平均出来的数字，它既不能证明合并对，也不能证明合并错。
判据是**每个来源自己的** A - max(B, C) 差：

   差 > 0   这个来源在学码信道 → 保留它是对的
   差 ≈ 0   A 臂和零码表一样 → 这个来源只贡献梯度噪声，和那 22 个死模板同类

三臂（同 71_，§11 修正版：prompt 里一个答案 token 都不许有）
    A 真实 RNA    B 换一条同来源 RNA    C 码向量置零

输出 reports/acc_by_source.json
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# ⚠️ 强制 stdout/stderr 用 utf-8。控制台默认编码可能是 GBK（cp936），而日志里带
#    ✓ / ⚠ 这类非 GBK 字符 —— 一个 print 就抛 UnicodeEncodeError，把生成式评测
#    整条打死（生成要跑一小时，代价极高）。放在脚本里自带免疫，
#    不依赖调用方记得设 PYTHONIOENCODING。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch  # noqa: E402

import rnaqwen  # noqa: F401,E402  必须先于 transformers/peft
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("rc", ROOT / "scripts/50_routeC_e2e.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

DEV = rc.DEV
QWEN = ROOT / "models/Qwen3-8B"


def norm(s: str) -> str:
    return "".join(c for c in s.strip().upper() if c.isalnum() or c in ".%").rstrip(".")


def cut(s: str) -> str:
    """只取第一个换行之前的部分。

    2026-10-18 审计：模型答完正确答案后**不会自己停**，会接着生成 chat 模板
    （"\\nuser\\n..." / "\\nassistant\\n<think>..."）。拿整段去比对，
    first-4 模板会从 93% 掉到 17%，全体从 70% 掉到 11% —— 那是评分口径的错，
    不是模型没学会。服务层早就按换行截断了（serve/server.py 的 STOP），
    这个评测脚本漏了同一步。见 reports/B3_score_audit.md。
    """
    return (s or "").split("\n", 1)[0]


def char_f1(pred: str, gold: str) -> float:
    """字符级 F1。freetext 的答案是几百字符的描述，exact match 恒为 0，
    没有区分度；F1 至少能看出「码表换了之后输出有没有变差」。"""
    from collections import Counter
    a, b = Counter(norm(pred)), Counter(norm(gold))
    if not a or not b:
        return 0.0
    ov = sum((a & b).values())
    if ov == 0:
        return 0.0
    p, r = ov / sum(a.values()), ov / sum(b.values())
    return 2 * p * r / (p + r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="训练 checkpoint（含 args）")
    ap.add_argument("--data", default="data/processed/rnaqa_merged.jsonl")
    ap.add_argument("--per-source", type=int, default=30, help="每个来源抽多少条（按模板均匀）")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--long-sources", default="freetext",
                    help="这些来源用长生成（答案几百字符）")
    ap.add_argument("--long-max-new", type=int, default=48)
    ap.add_argument("--out", default="reports/acc_by_source.json")
    ap.add_argument("--group-by", choices=["source", "template"], default="source",
                    help="按哪一维拆：source（合并数据集用）或 template（校正数据集用）。"
                         "校正数据集只有 10 个模板、答案全可复算，按模板拆才看得出"
                         "head 和 tail 的分界 —— tail 那组就是能力边界")
    ap.add_argument("--dry-run", action="store_true",
                    help="不载 LLM，只跑缓存/抽样/地板/prompt/leak 自检。"
                         "训练在跑时用这个做冒烟，避免两个 8B 同时占显存（会 0xC0000005）")
    args = ap.parse_args()

    ck_path = ROOT / args.ckpt
    if not ck_path.exists():
        print(f"[err] 没有 {ck_path}")
        return 2
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cargs = ck.get("args", {})
    print(f"[ckpt] {ck_path.name}  step {ck.get('step')}  "
          f"{ck.get('t_elapsed', 0)/3600:.2f} h")

    # ---- 复用训练时的缓存挂载（主 = _fsq_full_pos，副 = rfam memmap）----
    # 主缓存路径是模块常量（50_ 没有 --hid 参数），不要猜。
    d = torch.load(rc.HID, map_location="cpu", weights_only=False)
    hid_p = rc.HID
    allr = rc.load_records(None, sort=False)
    rc.register_cache("main", d["x"], d["lens"], d["offsets"], [r["uid"] for r in allr])
    print(f"[hid] 主缓存 {tuple(d['x'].shape)}  序列 {len(allr)}")

    hid2 = cargs.get("hid2", "")
    if hid2:
        import numpy as np
        d2 = torch.load(ROOT / cargs["hid2_idx"], map_location="cpu", weights_only=False)
        mm = np.memmap(ROOT / hid2, dtype=np.float16, mode="r",
                       shape=(int(d2["n_pos"]), int(d2["dim"])))
        rc.register_cache("aux", torch.from_numpy(mm), d2["lens"], d2["offsets"], d2["uids"])
        print(f"[hid] 副缓存 {tuple(mm.shape)}  序列 {len(d2['uids'])}")

    # ---- 配置：LATENT/LEVELS 的单一真相来源是 tokenizer.pt（同 50_）----
    tkc = torch.load(rc.FSQ_CKPT, map_location="cpu", weights_only=False)
    LAT, LEV = int(tkc["latent"]), int(tkc["levels"])
    POOL = int(cargs.get("pool", 4))
    LORA_R = int(cargs.get("lora_r", 8))
    rc.LATENT, rc.LEVELS, rc.VOCAB = LAT, LEV, LEV ** LAT

    # ⚠️ 底座必须跟着 checkpoint 走。写死 8B 的话，拿 35B 的档跑会**静默**加载 8B、
    #    再 strict=False 装适配器 —— 匹配不上的参数不报错，日志正常、数字全假。
    #    派生逻辑抽在 rnaqwen.qwen36.base_spec_from_ckpt_args 里，两个脚本共用。
    from rnaqwen.qwen36 import base_spec_from_ckpt_args
    base_dir, mk_kw = base_spec_from_ckpt_args(cargs)
    rc.LATENT, rc.LEVELS, rc.VOCAB = LAT, LEV, LEV ** LAT
    print(f"[cfg] {rc.FSQ_CKPT.name}: LATENT={LAT} LEVELS={LEV} VOCAB={rc.VOCAB} "
          f"pool={POOL} lora_r={LORA_R}")
    print(f"[cfg] 底座 {base_dir.name}  adapter={mk_kw['adapter']} "
          f"preset={mk_kw['lora_preset']} mode={mk_kw['base_mode']}")

    from transformers import AutoTokenizer
    tk0 = AutoTokenizer.from_pretrained(str(base_dir))
    tk = rc.Tok(tk0)

    # ---- 抽样：按来源、按模板均匀 ----
    # ⚠️ **必须流式读，不能把全文解析进内存。**
    #    原来是 `recs = [json.loads(l) for l in open(...)]` —— 1,496,820 条 dict
    #    约 1.2 GB，而这 1.2 GB 会**一直压到 35B 装载的峰值上**。
    #    实测同段代码时好时坏：页面文件来得及长就过，来不及长就在 safetensors 的
    #    `.item()` 处以 0xC0000005 崩（2026-10-14 连崩三次，而 87_ 从没崩过 ——
    #    差别就是 87_ 先建模型再读数据，76_ 反过来）。
    #    现在只留 test（实测 44k 条 ≈ 48 MB，全量 test ≈ 180 MB）；
    #    train **只用来算众数地板**，那就只累计计数，不保留记录。
    #    （Counter 必须在这里导入：L65 那个是 char_f1 内部的局部导入，main 里看不到。）
    from collections import Counter
    test: list = []
    floorc: dict = {}
    key_of = ((lambda r: r["source"]) if args.group_by == "source"
              else (lambda r: r["question"]))
    with open(ROOT / args.data, encoding="utf-8") as _fh:
        for _line in _fh:
            if not _line.strip():
                continue
            _r = json.loads(_line)
            _sp = _r.get("split")
            if _sp == "test":
                test.append(_r)
            elif _sp == "train":
                floorc.setdefault(key_of(_r), Counter())[norm(_r["answer"])] += 1
    print(f"[data] 流式读：test {len(test):,} 条；train 只留计数（{len(floorc)} 组）")
    long_set = {s.strip() for s in args.long_sources.split(",") if s.strip()}

    picks: dict[str, list] = {}

    def spread(lst: list, n: int) -> list:
        """等距取 n 条，不要取前 n 条。

        **数据文件按 source 顺序写**（seq8 在前、rfa 在后），取前 n 条
        会系统性只抽到 seq8（码长 255），而 rfa 那 3,946 条一条都进不来。
        实测：按模板拆时如果不 spread，10 个模板的码长示例全是 255。
        """
        if n <= 0 or len(lst) <= n:
            return lst
        st = len(lst) / n
        return [lst[min(int(i * st), len(lst) - 1)] for i in range(n)]

    for src in sorted({key_of(r) for r in test}):
        pool = [r for r in test if key_of(r) == src and r["uid"] in rc.IDX]
        byq: dict[str, list] = {}
        for r in pool:
            byq.setdefault(r["question"], []).append(r)
        picks[src] = spread(pool, args.per_source) if args.group_by == "source" else \
            [r for q in sorted(byq) for r in spread(byq[q], max(1, args.per_source // max(len(byq), 1)))]

    # 众数地板：训练集里该组最常见的答案（"永远答众数"的准确率）
    floor = {}
    for src in picks:
        c = floorc.get(src) or Counter()
        if not c:
            floor[src] = (0.0, "")
            continue
        floor[src] = (c.most_common(1)[0][1] / max(sum(c.values()), 1),
                      c.most_common(1)[0][0][:24])

    print("\n[抽样]")
    for src, v in picks.items():
        print(f"  {src:<14} {len(v):>3} 条  {len({r['question'] for r in v})} 个模板  "
              f"地板 {100*floor[src][0]:.2f}%")

    P = POOL

    def clen_of(uid: int) -> int:
        return rc.IDX[uid][2] // P

    @torch.no_grad()
    def gen(rec, src_key, arm):
        L = clen_of(rec["uid"])
        x = rc.slice_hid(rec["uid"])
        v, _, _, _ = model.rna_vectors(x, tau=LAT / 2)
        if arm == "C":
            v = torch.zeros_like(v)
        ids = (tk.sys_ids + tk.user_ids + tk.enc("Q: <RNA>")
               + [tk.base_vocab] * L
               + tk.enc(f"</RNA>{rec['question']}")
               + tk.assistant_ids + tk.no_think)          # 答案一个 token 都没有
        ii = torch.tensor([ids], device=DEV)
        emb = model.embed(ii, v)
        mn = args.long_max_new if rec["source"] in long_set else args.max_new
        out = model.llm.generate(inputs_embeds=emb,
                                 attention_mask=torch.ones_like(ii),
                                 max_new_tokens=mn, do_sample=False,
                                 pad_token_id=tk.pad_id)
        txt = tk0.decode(out[0], skip_special_tokens=True).strip()
        if "A:" in txt:
            txt = txt.split("A:")[-1].strip()
        return txt

    # dry-run 不载模型，base_vocab 从 config.json 读（= embedding 行数）
    # ⚠️ base_dir 而不是写死的 QWEN —— 否则跑 35B 的档时哨兵取成 151936（8B 的值），
    #    leak 自检和 prompt 构造都会用错的行数做边界。虽然不报错，但那是假的通过。
    # ⚠️ 两个底座的 config 结构不同：35B 是 VLM 类，vocab_size 只在 text_config 下
    #    （顶层没有）；8B 是普通 LM，只有顶层。所以**不能用 dict.get 的默认值**——
    #    默认值是立即求值的，会 KeyError（这里踩过一次）。
    if args.dry_run:
        _cfg = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
        _v = _cfg.get("vocab_size")
        if _v is None:
            _v = (_cfg.get("text_config") or {}).get("vocab_size")
        if _v is None:
            raise SystemExit(f"{base_dir.name}/config.json 里找不到 vocab_size")
        tk.base_vocab = int(_v)
        print(f"[cfg] base_vocab {tk.base_vocab}（{base_dir.name}/config.json，未载模型）")

    # ---- 静态 leak 自检 ----
    print("\n=== 静态 leak 自检 ===")
    nchk = 0
    for src, v in picks.items():
        for rec in v[:12]:
            L = clen_of(rec["uid"])
            ids = (tk.sys_ids + tk.user_ids + tk.enc("Q: <RNA>")
                   + [tk.base_vocab] * L + tk.enc(f"</RNA>{rec['question']}")
                   + tk.assistant_ids + tk.no_think)
            txt = tk0.decode([i for i in ids if i < tk.base_vocab], skip_special_tokens=True)
            a = norm(rec["answer"])
            if a and len(a) >= 4 and a in norm(txt):
                raise SystemExit(f"prompt 泄漏答案 {a[:30]!r} —— 结果不可信（§11）")
            nchk += 1
    print(f"  {nchk} 条 prompt 不含答案  OK")
    _ne = [(s, v) for s, v in picks.items() if v]   # 跳过空来源组（缺对应缓存时）
    print(f"  码长示例: " + "  ".join(
        f"{src[:20]}={clen_of(v[0]['uid'])}" for src, v in _ne[:2]))
    # 抽样的**分布**要看，不能只看第一条：数据按 source 顺序写，
    # 取前 n 条会全是 seq8（码长 255），rfa（码长 20~255）一条都进不来。
    for src, v in _ne[:2]:
        cl = sorted(clen_of(r["uid"]) for r in v)
        n255 = sum(1 for x in cl if x == 255)
        print(f"  [{src[:34]}] 码长 中位 {cl[len(cl)//2]}  最小 {cl[0]}  最大 {cl[-1]}  "
              f"=255 的 {n255}/{len(cl)}")
    srcs = Counter(r["source"] for v in picks.values() for r in v)
    print(f"  抽样来源分布: {dict(srcs)}")

    if args.dry_run:
        print("\n[dry-run] 缓存/抽样/地板/prompt/leak 全部通过，未载 LLM。")
        print("[dry-run] 训练在跑时只能用这个。")
        return 0

    print(f"\n[model] 载入 {base_dir.name} + adapter + 码表 …")
    model = rc.E2EModel(LORA_R, cargs.get("rna_emb", "table"),
                        latent=LAT, levels=LEV, **mk_kw)
    # 占位 token 的哨兵必须**越过 embedding 行数**（= model.base_vocab）。
    # 它不等于 tokenizer 的 vocab_size：Qwen3-8B 是 151936 vs 151643，
    # embedding 有填充行。写成相等会误报（这次就栽在这）。
    tk.base_vocab = model.base_vocab
    assert model.base_vocab >= tk0.vocab_size, (model.base_vocab, tk0.vocab_size)
    print(f"[tok] 哨兵 base_vocab={model.base_vocab}  "
          f"tokenizer vocab={tk0.vocab_size}  pad={tk.pad_id}")
    model.fsq.load_state_dict(ck["fsq"])
    if ck.get("rna_emb") is not None:
        model.rna_emb.weight.data.copy_(
            ck["rna_emb"].to(model.rna_emb.weight.device, model.rna_emb.weight.dtype))
    miss = model.llm.load_state_dict(ck["lora"], strict=False)
    bad = [k for k in (getattr(miss, "unexpected_keys", None) or [])]
    print(f"[lora] {len(ck['lora'])} 张量  unexpected {len(bad)}")
    if bad:
        print(f"  ** 有未匹配 key，adapter 没挂上：{bad[:4]}")
        return 3
    # ⚠️ 反向也要查：**模型的可训参数是否被 checkpoint 全覆盖了**。
    #    只看 unexpected_keys 会漏掉更危险的那一半 —— 拿 8B 的档装进 35B 时，
    #    8B 的 key 全都能匹配（unexpected=0），而 35B 的 SOS 适配器全是随机初始化，
    #    日志正常、数字全假。判据必须是「可训参数集合 == checkpoint 里的集合」。
    want = {n for n, p in model.llm.named_parameters() if p.requires_grad}
    got = set(ck["lora"].keys())
    if want - got:
        print(f"  ** 有 {len(want - got)} 个可训参数不在 checkpoint 里"
              f"（adapter 与底座对不上）：{sorted(want - got)[:4]}")
        return 3
    print(f"[lora] 可训参数 {len(want)} 个，全部被 checkpoint 覆盖 ")
    model.eval()

    # ---- 三臂 ----
    print("\n=== 生成 ===")
    res: dict[str, dict[str, list]] = {}
    t0 = time.time()
    # ⚠️ 异常不能只记类型。2026-10-14 踩过：`pred = f"<ERR {type(e).__name__}>"`
    #    把 120 次生成失败全吞成同一个占位串，于是三臂准确率齐刷刷 0.0%，
    #    看起来像"模型没学会"，实际是**生成一次都没跑通**。白等了 25 分钟。
    #    现在第一次失败就把完整栈打出来。
    _err_shown = [False]
    _err_run = [0]
    for src, v in picks.items():
        g = torch.Generator().manual_seed(7)
        perm = torch.randperm(len(v), generator=g).tolist()
        res[src] = {a: [] for a in ("A", "B", "C")}
        for i, rec in enumerate(v):
            for a in ("A", "B", "C"):
                src_rec = v[perm[i]] if a == "B" else rec
                try:
                    pred = gen(src_rec, src, a)
                    _err_run[0] = 0
                except Exception as e:
                    _err_run[0] += 1
                    pred = f"<ERR {type(e).__name__}: {e}>"
                    if not _err_shown[0]:
                        _err_shown[0] = True
                        print(f"\n  ** 首次生成失败（{src} / {a} 臂），完整栈如下 —— "
                              f"后面同样的错误只记一行 **", flush=True)
                        import traceback as _tb
                        _tb.print_exc()
                        print()
                    if _err_run[0] >= 3:
                        print(f"\n  ** 连续 3 次生成失败，判定为系统性错误，提前退出 **"
                              f"\n     （上一次是异常被吞成占位串，120 次全废、白等 25 分钟）",
                              flush=True)
                        return 4
                res[src][a].append((norm(cut(pred)), norm(rec["answer"]),
                                    char_f1(cut(pred), rec["answer"]),
                                    pred, rec["question"]))
            line = "  ".join(
                f"{a} {100*sum(1 for p, gg, _, _, _ in res[src][a] if p == gg)/len(res[src][a]):.1f}%"
                for a in ("A", "B", "C"))
            print(f"  [{src}] {i+1}/{len(v)}  {line}   {time.time()-t0:.0f}s", flush=True)

    # ---- 生成一结束就落盘 ----
    # ⚠️ 生成是最贵的一步（这里 120 条三臂约 23 分钟，freetext 上更长）。
    #    而下面全是分析代码 —— 2026-10-14 就出过「生成全跑完、结果只在内存里，
    #    死在 head/tail 分组的 TypeError 上，整轮白跑」。
    #    先无条件把原始生成结果写下来，分析的 bug 就不会再吃掉生成成本。
    raw_p = ROOT / (args.out + ".raw.json")
    raw_p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[raw] 生成结果已落盘 {raw_p.name}（分析崩了也不用重跑生成）", flush=True)

    # ---- 汇总 ----
    print("\n" + "=" * 78)
    print(f"按{'来源' if args.group_by == 'source' else '模板'}拆分（exact match / char-F1）")
    print("=" * 78)
    lab = "来源" if args.group_by == "source" else "模板"
    print(f"  {lab:<40}{'n':>4}  {'accA':>7}{'accB':>7}{'accC':>7}  "
          f"{'A-C':>7}  {'地板':>6}  判定")
    out = {"ckpt": str(ck_path), "step": ck.get("step"), "group_by": args.group_by,
           "groups": {}}
    for src, v in picks.items():
        acc = {a: sum(1 for p, gg, _, _, _ in res[src][a] if p == gg) / len(v)
               for a in ("A", "B", "C")}
        f1 = {a: sum(x[2] for x in res[src][a]) / len(v) for a in ("A", "B", "C")}
        d_ac = acc["A"] - acc["C"]
        d_ab = acc["A"] - acc["B"]
        if d_ac > 0.10 and d_ab > 0.10:
            verdict = "学码信道 ✅"
        elif d_ac > 0.03 and d_ab > 0.03:
            verdict = "弱信号 ⚠️"
        else:
            verdict = "没学 ❌"
        name = src if len(src) <= 38 else src[:37] + "…"
        print(f"  {name:<40}{len(v):>4}  {100*acc['A']:>6.1f}%{100*acc['B']:>6.1f}%"
              f"{100*acc['C']:>6.1f}%  {100*d_ac:>+6.1f}pp  "
              f"{100*floor[src][0]:>5.2f}%  {verdict}")
        out["groups"][src] = dict(n=len(v), acc=acc, f1=f1, d_ac=d_ac, d_ab=d_ab,
                                  floor=floor[src][0], verdict=verdict)
    # 加权总
    tot = sum(len(v) for v in picks.values())
    accA = sum(sum(1 for p, gg, _, _, _ in res[s]["A"] if p == gg) for s in picks) / tot
    accB = sum(sum(1 for p, gg, _, _, _ in res[s]["B"] if p == gg) for s in picks) / tot
    accC = sum(sum(1 for p, gg, _, _, _ in res[s]["C"] if p == gg) for s in picks) / tot
    print(f"  {'加权总':<40}{tot:>4}  {100*accA:>6.1f}%{100*accB:>6.1f}%{100*accC:>6.1f}%")
    out["overall"] = dict(n=tot, acc={"A": accA, "B": accB, "C": accC})

    # 同一批输出、不做换行截断的口径。两者差距就是「模型答对了但没停」的比例，
    # 必须一起报，否则只看一个数会重演 11% vs 70% 那次误判。
    exactA = sum(1 for s in picks for _, gg, _, pr, _ in res[s]["A"]
                 if norm(pr) == gg) / tot
    print(f"  {'  其中：不截断的 exact':<40}{tot:>4}  A {100*exactA:>6.1f}%"
          f"   （截断口径 A {100*accA:.1f}%）")
    out["overall"]["exact_notrunc_A"] = exactA

    # ---- 按模板拆时，额外给 head / tail 分组 ----
    # 这是能力边界所在：head 是「读序列开头」，tail 是「读序列结尾」，
    # 两者答案同样可复算，唯一区别是读的位置。探针说 tail 弱（0.09~0.15），
    # 这一栏就是把那个相关性变成因果的地方。
    if args.group_by == "template":
        # 约定必须和上面「按来源」那段一致：**一个来源一项，三项分别是 A/B/C 的预测列表**。
        # 原写法塞的是 `(preds, len(v))` 二元组，下面却按 `x[1]` 取 B 臂 ——
        # x[1] 实际是 int，于是 `for p, gg, ... in x[1]` 抛 TypeError。
        # 这段只在 --group-by template 才走到，所以之前按 source 跑从没暴露过。
        grp = {"head": [], "tail": []}
        for src, v in picks.items():
            g = "tail" if src.startswith("What are the last") else "head"
            grp[g].append((res[src]["A"], res[src]["B"], res[src]["C"], len(v)))
        print()
        print("  --- head / tail 分组（同一批 RNA、同一个信道、只换读的位置）---")
        for g, items in grp.items():
            if not items:
                continue
            n = sum(it[3] for it in items)                 # 组内样本数（每臂各 n 条）
            if n == 0:
                continue
            a = sum(sum(1 for p, gg, _, _, _ in it[0] if p == gg) for it in items) / n
            b = sum(sum(1 for p, gg, _, _, _ in it[1] if p == gg) for it in items) / n
            c = sum(sum(1 for p, gg, _, _, _ in it[2] if p == gg) for it in items) / n
            print(f"    {g:<6} n={n:>4}  accA {100*a:>5.1f}%  accB {100*b:>5.1f}%  "
                  f"accC {100*c:>5.1f}%   A-C {100*(a-c):>+6.1f}pp")
            out.setdefault("head_tail", {})[g] = dict(n=n, accA=a, accB=b, accC=c)
        ht = out.get("head_tail", {})
        if "head" in ht and "tail" in ht:
            dh, dt = ht["head"]["accA"], ht["tail"]["accA"]
            print(f"    head - tail 的 A 臂差 {100*(dh-dt):+.1f}pp  "
                  + ("-> **位置是因果的**（同一批 RNA，只换读的位置就差这么多）"
                     if dh - dt > 0.15 else
                     "-> 位置差异不显著，探针那个相关性没有转成因果"))

    print("\n各组 A 臂样例:")
    for src, v in picks.items():
        for p, gg, f, raw, q in res[src]["A"][:1]:
            print(f"  [{src[:44]}]")
            print(f"     生成 {raw[:56]!r}")
            print(f"     答案 {gg[:40]!r}  f1 {f:.2f}  {'' if p == gg else '✗'}")

    (ROOT / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    print(f"\n[out] {args.out}")
    return 0


def _hard_exit(code: int) -> None:
    """绕过解释器收尾硬退。

    实测（2026-10-14）：python 抛异常退出后**不会真的死** —— 它挂在 ROCm 的上下文
    销毁上，进程一直留着，继续占着 18 GiB 显存，而且 `taskkill /F /T` 报
    「could not be terminated / There is no running instance of the task」，
    只有把**父 shell** 也杀掉才回收得掉。表现是"评测崩了，但显存一直不还"。
    os._exit 跳过会挂的那段析构；句柄和显存由操作系统回收。
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


if __name__ == "__main__":
    try:
        _code = main()
    except SystemExit as e:
        _code = e.code if isinstance(e.code, int) else 1
    except BaseException:
        import traceback
        traceback.print_exc()
        _code = 1
    _hard_exit(_code)
