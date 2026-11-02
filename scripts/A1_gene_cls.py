"""Gene-level dsRNA classifier on top of the SOS-LoRA representation.

Phase `extract`
    gene record -> genomic 101 nt windows -> ERNIE-RNA -> FSQ codes
                -> rna_emb vectors -> Qwen3.6 body -> hidden state
    The body runs under torch.inference_mode(), so only the head ever needs
    gradients.  Cached to runs/immune_gene_cls/feats_<tag>.pt.

Phase `train`
    cached features -> Linear(d, 2), pair-grouped splits, class-balanced CE.
    Reported next to the length / WT-level / position baselines, because the
    gene table is only weakly matched and those confounds must stay visible.

The code channel is the same one the generative arm uses, so `--tag perm`
reproduces route C's vsperm: a random bijection over the 4096 code ids.
Same table, same marginals, same scale; only code<->content is cut.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import rnaqwen  # noqa: F401,E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = ROOT / "data/processed/immune_gene_dsrna.jsonl"
FASTA = ROOT / "data/raw/genome/hg38.fasta"
OUTDIR = ROOT / "runs/immune_gene_cls"
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def _mod(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


# ---------------------------------------------------------------- data
def load_records(path=DATA):
    recs = [json.loads(l) for l in path.open(encoding="utf-8")]
    for r in recs:
        if "length" not in r:
            if "end" in r and "start" in r:
                r["length"] = int(r["end"]) - int(r["start"])
            elif r.get("seq"):
                r["length"] = len(r["seq"])
            else:
                r["length"] = 0
    return recs


def assign_pairs(recs, len_tol=0.15):
    """Greedy 1:1 length matching, same chromosome first.

    Mirrors 99_gene_dsrna_task.py but works on the already-matched table, so
    the positive/negative pair that must never straddle a train/test split has
    an explicit id.  Sorted by uid so the result is deterministic.
    """
    pos = sorted([r for r in recs if r["label"] == 1], key=lambda r: r["uid"])
    neg = [r for r in recs if r["label"] == 0]
    by_chrom = defaultdict(list)
    for r in neg:
        by_chrom[r["chrom"]].append(r)
    used, pid = set(), 0
    for p in pos:
        L = max(p["length"], 1)
        pick = None
        for c in sorted(by_chrom.get(p["chrom"], []), key=lambda r: abs(r["length"] - L)):
            if c["uid"] in used:
                continue
            if abs(c["length"] - L) / L > len_tol:
                break
            pick = c; break
        if pick is None:
            rest = sorted((r for r in neg if r["uid"] not in used),
                          key=lambda r: abs(r["length"] - L))
            for c in rest:
                if abs(c["length"] - L) / L <= len_tol:
                    pick = c; break
        if pick is None:
            continue
        used.add(pick["uid"])
        p["pair_id"] = pid; pick["pair_id"] = pid
        pid += 1
    for r in recs:
        r.setdefault("pair_id", -1 - (zlib.crc32(r["uid"].encode()) % 10 ** 9))
    return pid


def windows_for(rec, fasta, k=3, width=101):
    """k windows centred inside the gene body, plus strand orientation.

    Windows are allowed to spill past the gene boundary (clipped to the
    chromosome) so every input has the same 101 nt the code channel was
    trained on, even for the genes that are shorter than that.
    """
    S, E = int(rec["start"]), int(rec["end"])
    L = max(E - S + 1, 1)
    half = width // 2
    out = []
    for i in range(k):
        c = S + int(L * (i + 0.5) / k)
        w = fasta.get(rec["chrom"], c - half, c - half + width - 1)
        if not w:
            break
        if rec.get("strand") == "-":
            w = revcomp(w)
        out.append(w)
    return out


# ---------------------------------------------------------------- model
def load_model(ckpt: Path):
    rc = _mod("rc", "scripts/50_routeC_e2e.py")
    from rnaqwen.qwen36 import base_spec_from_ckpt_args
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cargs = ck["args"]
    base_dir, mk_kw = base_spec_from_ckpt_args(cargs)
    lora_r = int(cargs.get("lora_r", 8))
    tkc = torch.load(ROOT / "checkpoints/fsq_full/tokenizer.pt",
                     map_location="cpu", weights_only=False)
    lat = int(ck["fsq"]["enc.weight"].shape[0]); lev = int(tkc["levels"])
    rc.LATENT, rc.LEVELS, rc.VOCAB = lat, lev, lev ** lat
    print(f"[cfg] step={ck.get('step')} base={base_dir.name} latent={lat} levels={lev} vocab={lev ** lat}", flush=True)
    model = rc.E2EModel(lora_r, cargs.get("rna_emb", "table"), latent=lat, levels=lev, **mk_kw)
    model.fsq.load_state_dict(ck["fsq"])
    if ck.get("rna_emb") is not None:
        model.rna_emb.weight.data.copy_(ck["rna_emb"].to(model.rna_emb.weight.device,
                                                         model.rna_emb.weight.dtype))
    miss = model.llm.load_state_dict(ck["lora"], strict=False)
    print(f"[load] missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}", flush=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return rc, model, cargs, lev ** lat


def load_code_only(ckpt: Path):
    """FSQ + code table only -- no 35B.  ~5 s instead of ~6 min."""
    from types import SimpleNamespace
    rc = _mod("rc", "scripts/50_routeC_e2e.py")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    tkc = torch.load(ROOT / "checkpoints/fsq_full/tokenizer.pt",
                     map_location="cpu", weights_only=False)
    lat = int(ck["fsq"]["enc.weight"].shape[0]); lev = int(tkc["levels"])
    rc.LATENT, rc.LEVELS, rc.VOCAB = lat, lev, lev ** lat
    d = int(ck["rna_emb"].shape[1])
    fsq = rc.FSQCode(768, lat, lev, d).to(DEV).eval()
    fsq.load_state_dict(ck["fsq"])
    emb = nn.Embedding(lev ** lat, d).to(DEV)
    emb.weight.data.copy_(ck["rna_emb"].to(DEV, emb.weight.dtype))
    emb.eval()
    print(f"[code-only] latent={lat} levels={lev} vocab={lev ** lat} d={d}", flush=True)
    return rc, SimpleNamespace(fsq=fsq, rna_emb=emb, base_vocab=248320), ck["args"], lev ** lat


def make_prompt(tk, ncode: int, question: str):
    return (list(tk.sys_ids) + list(tk.user_ids) + list(tk.enc("Q: <RNA>"))
            + [tk.base_vocab] * ncode
            + list(tk.enc("</RNA>" + question)) + list(tk.assistant_ids)
            + list(tk.no_think) + list(tk.enc(" A:")))


# ---------------------------------------------------------------- extract
def _flush(partdir, feats, feats_mean, meta, a):
    idx = len(list(partdir.glob("part*.pt")))
    f = partdir / f"part{idx:03d}.pt"
    torch.save(dict(X=torch.stack(feats).float(), Xmean=torch.stack(feats_mean).float(),
                    meta=list(meta), tag=a.tag, k=a.k, win=a.win), f)
    print(f"[flush] {f.name} rows={len(meta)} total_saved={idx + 1} parts", flush=True)


def extract(a):
    from transformers import AutoTokenizer
    from rnaqwen.models.encoder import RnaEncoder

    OUTDIR.mkdir(parents=True, exist_ok=True)
    recs = load_records(Path(a.data))
    if recs and all("pair_id" in r for r in recs):
        npair = len({r["pair_id"] for r in recs})
    else:
        npair = assign_pairs(recs)
    print(f"[data] rows={len(recs)} pairs={npair}", flush=True)

    f93 = _mod("f93", "scripts/93_genome_windows.py")
    fasta = f93.Fasta(FASTA)
    have_chrom = set(fasta.idx)

    f94 = _mod("f94", "scripts/94_structure_features.py")

    if a.no_body:
        rc, model, cargs, nvocab = load_code_only(ROOT / a.ckpt)
    else:
        rc, model, cargs, nvocab = load_model(ROOT / a.ckpt)
    from rnaqwen.qwen36 import base_spec_from_ckpt_args
    base_dir, _ = base_spec_from_ckpt_args(cargs)
    tk = rc.Tok(AutoTokenizer.from_pretrained(str(base_dir)))
    # route C forces this (50_routeC_e2e.py:1073-1075); without it the placeholder
    # id and the is_rna mask disagree and every special token counts as RNA.
    if tk.base_vocab != model.base_vocab:
        print(f"[vocab] tokenizer {tk.base_vocab} != embedding {model.base_vocab}; using the latter",
              flush=True)
        tk.base_vocab = model.base_vocab
    if not a.no_body:
        print(f"[shape] base_emb={tuple(model.base_emb.weight.shape)} "
              f"rna_emb={tuple(model.rna_emb.weight.shape)}", flush=True)
    enc = RnaEncoder(str(ROOT / "models/ernierna"), dtype=torch.float32).to(DEV).eval()
    body, _lmh = (None, None) if a.no_body else rc._lm_parts(model)

    perm = None
    if a.tag == "perm":
        g = torch.Generator(device="cpu").manual_seed(2024)
        perm = torch.randperm(nvocab, generator=g).to(DEV)
        print(f"[perm] random bijection over {nvocab} codes, seed 2024", flush=True)
    elif a.tag == "zero":
        print("[zero] code table zeroed", flush=True)

    todo = []
    skipped = 0
    for r in recs:
        if "chr" + r["chrom"] not in have_chrom and r["chrom"] not in have_chrom:
            skipped += 1
            continue
        if r.get("seq"):
            ws = [r["seq"] if r.get("strand") != "-" else revcomp(r["seq"])]
        else:
            ws = windows_for(r, fasta, k=a.k, width=a.win)
        if a.tag == "shuffle":
            rr = random.Random(zlib.crc32(r["uid"].encode()))
            ws = ["".join(rr.sample(list(w), len(w))) for w in ws]
        for j, w in enumerate(ws):
            if len(w) < a.win:
                continue
            todo.append((r, j, w))
    if a.limit:
        random.Random(0).shuffle(todo)
        todo = todo[: a.limit]
    partdir = OUTDIR / f"{a.tag}_{Path(a.data).stem}_parts"
    partdir.mkdir(parents=True, exist_ok=True)
    existing = sorted(partdir.glob("part*.pt"))
    done_before = 0
    for pf in existing:
        try:
            done_before += len(torch.load(pf, map_location="cpu", weights_only=False)["meta"])
        except Exception:
            pass
    if done_before:
        todo = todo[done_before:]
    print(f"[win] windows={len(todo)} resumed_from={done_before} "
          f"genes_skipped_no_chrom={skipped}", flush=True)

    feats, feats_mean, meta = [], [], []
    import time as _t
    st = _t.time()
    done = 0
    with torch.inference_mode():
        for b0 in range(0, len(todo), a.bs):
            chunk = todo[b0:b0 + a.bs]
            seqs = [w for _r, _j, w in chunk]
            o = enc(seqs); h = o.hidden.float(); m = o.mask.squeeze(-1) > 0
            pools = []
            for bi, w in enumerate(seqs):
                hh = h[bi][m[bi]]
                if hh.shape[0] == len(w) + 2:
                    hh = hh[1:-1]
                pools.append(hh)
            # FSQCode pools 4:1 internally, on ONE sequence's token hidden.
            # Feeding it pre-pooled rows silently yields 6 codes instead of 25.
            codes = [model.fsq(q.to(DEV))[0] for q in pools]
            nlist = [int(c.shape[0]) for c in codes]
            if len(set(nlist)) == 1:
                ncode = nlist[0]
                code = torch.cat(codes, 0)
                idx = perm[code] if perm is not None else code
                if a.tag == "zero":
                    v = torch.zeros(idx.shape[0], model.rna_emb.weight.shape[1],
                                    device=DEV, dtype=model.rna_emb.weight.dtype)
                else:
                    v = model.rna_emb(idx)
                if a.no_body:
                    hs = hm = v.reshape(len(chunk), ncode, -1).mean(1).float().cpu()
                    ids, it, e = None, None, None
                else:
                    ids = make_prompt(tk, ncode, a.question)
                    it = torch.tensor(ids, device=DEV)
                    e = model.base_emb(it.clamp(max=tk.base_vocab - 1))
                if not a.no_body:
                  e = e.unsqueeze(0).repeat(len(chunk), 1, 1)
                  e[:, it >= tk.base_vocab, :] = v.reshape(len(chunk), ncode, e.shape[-1]).to(e.dtype)
                  am = torch.ones(len(chunk), it.shape[0], device=DEV, dtype=torch.long)
                  o3 = body(inputs_embeds=e, attention_mask=am, use_cache=False)
                  hs_all = o3.last_hidden_state
                  slot = (it >= tk.base_vocab)
                  hs = hs_all[:, -1].float().cpu()
                  hm = hs_all[:, slot, :].mean(1).float().cpu()
            else:
                outs = []
                for c in codes:
                    idx = perm[c] if perm is not None else c
                    vv = model.rna_emb(idx)
                    ids = make_prompt(tk, vv.shape[0], a.question)
                    it = torch.tensor(ids, device=DEV).unsqueeze(0)
                    ee = model.embed(it, vv)
                    o4 = body(inputs_embeds=ee, attention_mask=torch.ones_like(it),
                              use_cache=False).last_hidden_state
                    outs.append((o4[:, -1], o4[0, it.squeeze(0) >= tk.base_vocab, :].mean(0, keepdim=True)))
                hs = torch.cat([a_ for a_, _b in outs], 0).float().cpu()
                hm = torch.cat([b_ for _a, b_ in outs], 0).float().cpu()

            for (r, j, w), row, rowm in zip(chunk, hs, hm):
                feats.append(row)
                feats_mean.append(rowm)
                meta.append(dict(uid=r["uid"], gene=r.get("gene", r.get("mark", "")),
                                 chrom=r["chrom"], pair_id=r["pair_id"],
                                 label=int(r["label"]), win=j,
                                 length=int(r["length"]),
                                 wt_dsrna=float(r.get("wt_dsrna", 0.0)),
                                 rel_change=float(r.get("rel_change", 0.0)),
                                 gc=float((w.count("G") + w.count("C")) / max(len(w), 1)),
                                 split=r.get("split", ""),
                                 mark=r.get("mark", ""),
                                 st=f94.features(w)))
            done += len(chunk)
            if (b0 // a.bs) % 5 == 0 or done >= len(todo):
                el = _t.time() - st
                print(f"[extract] {done}/{len(todo)}  {el / max(done,1)*1000:.0f} ms/win  "
                      f"eta {(len(todo) - done) * el / max(done,1) / 60:.1f} min", flush=True)
            model._usage_parts = None
            if len(feats) >= a.flush:
                _flush(partdir, feats, feats_mean, meta, a)
                feats, feats_mean, meta = [], [], []

    if feats:
        _flush(partdir, feats, feats_mean, meta, a)
    feats, feats_mean, meta = [], [], []
    for pf in sorted(partdir.glob("part*.pt")):
        d_ = torch.load(pf, map_location="cpu", weights_only=False)
        feats.append(d_["X"]); feats_mean.append(d_["Xmean"]); meta.extend(d_["meta"])
    feats = [torch.cat(feats, 0)] if feats else [torch.empty(0, 2048)]
    feats_mean = [torch.cat(feats_mean, 0)] if feats_mean else [torch.empty(0, 2048)]

    X = feats[0].to(torch.float32)
    Xm = feats_mean[0].to(torch.float32)
    tag = f"{a.tag}_k{a.k}_n{len(meta)}"
    path = OUTDIR / f"feats_{tag}.pt"
    torch.save(dict(X=X, Xmean=Xm, meta=meta, tag=a.tag, k=a.k, win=a.win,
                    question=a.question, ckpt=str(a.ckpt)), path)
    print(f"[save] {path}  X={tuple(X.shape)}  {path.stat().st_size / 1e6:.1f} MB", flush=True)
    print(f"[codes] windows={len(meta)} genes={len({m['uid'] for m in meta})}", flush=True)
    return 0


# ---------------------------------------------------------------- train
def auc(y, s):
    """Rank AUC, no sklearn dependency."""
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    n1 = sum(1 for v in y if v == 1); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    sp = sum(ranks[i] for i in range(len(y)) if y[i] == 1)
    return (sp - n1 * (n1 + 1) / 2) / (n1 * n0)


def _prune_ckpt(d, keep=10):
    """Keep best.pt plus the newest keep rolling snapshots."""
    import os
    snaps = sorted(q for q in d.glob("last_ep*.pt"))
    for q in snaps[:-keep]:
        try:
            os.remove(q)
        except OSError:
            pass
    return len(snaps[-keep:])


def _gene_agg(meta, logit):
    """Mean over the windows of each gene -> one score per gene."""
    by, lab = defaultdict(list), {}
    for row, l in zip(meta, logit):
        by[row["uid"]].append(l)
        lab[row["uid"]] = row["label"]
    keys = list(by)
    return keys, [sum(by[k]) / len(by[k]) for k in keys], [lab[k] for k in keys]


def train(a):
    path = OUTDIR / a.feats
    if path.is_dir():
        parts = sorted(path.glob("part*.pt"))
        ds = [torch.load(q, map_location="cpu", weights_only=False) for q in parts]
        d = dict(X=torch.cat([q["X"] for q in ds], 0),
                 Xmean=torch.cat([q["Xmean"] for q in ds], 0),
                 meta=[m for q in ds for m in q["meta"]],
                 tag=ds[0]["tag"], k=ds[0]["k"], win=ds[0]["win"])
        path_name = str(path.relative_to(OUTDIR))
    else:
        d = torch.load(path, map_location="cpu", weights_only=False)
        path_name = path.name
    key = "Xmean" if a.readout == "mean" else "X"
    X, meta = d.get(key, d["X"]), d["meta"]
    print(f"[readout] {a.readout} ({key})", flush=True)
    print(f"[load] {path_name}  X={tuple(X.shape)}  tag={d['tag']}", flush=True)
    if a.mark:
        keep = [i for i, r in enumerate(meta) if r.get("mark") == a.mark]
        X = X[keep]; meta = [meta[i] for i in keep]
        print(f"[mark] {a.mark}: {len(meta)} rows", flush=True)

    if a.use_split and meta and meta[0].get("split"):
        tr = {"train"}; va = {"val"}; te = {"test"}
        def _sp(sel):
            return [i for i, r in enumerate(meta) if r.get("split") in sel]
        itr_, iva_, ite_ = _sp(tr), _sp(va), _sp(te)
        print(f"[split] chrom holdout train={len(itr_)} val={len(iva_)} test={len(ite_)}", flush=True)
        pairs = None
    else:
        pairs = sorted({r["pair_id"] for r in meta if r["pair_id"] >= 0})
        rng = random.Random(2024); rng.shuffle(pairs)
        n = len(pairs)
        tr = set(pairs[: int(n * 0.70)])
        va = set(pairs[int(n * 0.70): int(n * 0.85)])
        te = set(pairs[int(n * 0.85):])
    def mask(sel):
        return [i for i, r in enumerate(meta) if r["pair_id"] in sel]

    if pairs is not None:
        print(f"[split] pairs train={len(tr)} val={len(va)} test={len(te)}", flush=True)
        itr, iva, ite = mask(tr), mask(va), mask(te)
    else:
        itr, iva, ite = itr_, iva_, ite_

    if a.zscore:
        mu, sd = X[itr].mean(0), X[itr].std(0).clamp(min=1e-6)
        X = (X - mu) / sd
        print(f"[zscore] per-dim standardised on train split", flush=True)

    if a.struct:
        keys = ["mfe_per_nt", "ens", "paired_frac", "longest_stem",
                "acc_min", "acc_mean", "gc"]
        S = torch.tensor([[float((m.get("st") or {}).get(k, 0.0)) for k in keys]
                          for m in meta], dtype=torch.float32)
        mu, sd = S[itr].mean(0), S[itr].std(0).clamp(min=1e-6)
        X = torch.cat([X, (S - mu) / sd], 1)
        print(f"[struct] +{len(keys)} ViennaRNA features -> d={X.shape[1]}", flush=True)
    Xtr = X[itr].to(DEV); ytr = torch.tensor([meta[i]["label"] for i in itr], device=DEV)
    Xva = X[iva].to(DEV); yva = torch.tensor([meta[i]["label"] for i in iva], device=DEV)
    Xte = X[ite].to(DEV); yte = torch.tensor([meta[i]["label"] for i in ite], device=DEV)

    ckdir = OUTDIR / f"ckpt_{path.stem}"
    ckdir.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] {ckdir}  (best.pt + newest {a.keep})", flush=True)

    d_in = X.shape[1]
    head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_in, 2)).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    w = torch.tensor([1.0, 1.0], device=DEV)
    best, best_state, bad = -1.0, None, 0
    for ep in range(a.epochs):
        head.train()
        p = torch.randperm(len(itr), device=DEV)
        tot = 0.0
        for s in range(0, len(itr), a.bs):
            idx = p[s:s + a.bs]
            lg = head(Xtr[idx])
            loss = F.cross_entropy(lg, ytr[idx], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        head.eval()
        with torch.no_grad():
            lv = head(Xva); pr = lv.argmax(-1)
            av = float((pr == yva).float().mean())
            s = torch.softmax(lv, -1)[:, 1].cpu().tolist()
            keys, gs, gy = _gene_agg([meta[i] for i in iva], s)
            gauc = auc(gy, gs)
        if gauc > best:
            best, bad = gauc, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            torch.save(dict(state=best_state, ep=ep, val_gene_auc=gauc,
                            struct=a.struct, zscore=a.zscore, readout=a.readout,
                            feats=path_name), ckdir / "best.pt")
        else:
            bad += 1
        torch.save(dict(state={k: v.detach().clone() for k, v in head.state_dict().items()},
                        ep=ep, val_gene_auc=gauc, train_loss=tot / len(itr)),
                   ckdir / f"last_ep{ep:03d}.pt")
        nkept = _prune_ckpt(ckdir, a.keep)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"[ep {ep}] loss={tot / len(itr):.4f} val_win_acc={av:.3f} "
                  f"val_gene_auc={gauc:.4f} ckpt={nkept}", flush=True)
        if bad >= a.patience:
            print(f"[early] stop at ep {ep}", flush=True); break
    if best_state:
        head.load_state_dict(best_state)
    torch.save(dict(state=head.state_dict(), best_val_gene_auc=best,
                    struct=a.struct, zscore=a.zscore, readout=a.readout,
                    feats=path_name), ckdir / "head.pt")

    head.eval()
    with torch.no_grad():
        lv = head(Xva); ptr = head(Xte)
    res = {}
    for nm, L, idxs, yy in (("val", lv, iva, yva), ("test", ptr, ite, yte)):
        pr = L.argmax(-1).cpu()
        s = torch.softmax(L, -1)[:, 1].cpu().tolist()
        keys, gs, gy = _gene_agg([meta[i] for i in idxs], s)
        res[nm] = dict(win_acc=float((pr == yy.cpu()).float().mean()),
                       win_auc=auc(yy.cpu().tolist(), s),
                       gene_auc=auc(gy, gs),
                       gene_acc=float(sum(1 for a_, b_ in zip(gs, gy) if (a_ > 0.5) == bool(b_)) / len(gy)),
                       n_win=len(idxs), n_gene=len(gy))
        print(f"[{nm}] win_acc={res[nm]['win_acc']:.3f} win_auc={res[nm]['win_auc']:.4f} "
              f"gene_auc={res[nm]['gene_auc']:.4f} gene_acc={res[nm]['gene_acc']:.3f} "
              f"n_win={res[nm]['n_win']} n_gene={res[nm]['n_gene']}", flush=True)

    # keyword-only baselines: how much of this is just gene length / expression
    base = {}
    for nm, idxs in (("val", iva), ("test", ite)):
        yy = [meta[i]["label"] for i in idxs]
        for f in ("length", "wt_dsrna", "gc"):
            base[f"{nm}_{f}"] = auc(yy, [meta[i][f] for i in idxs])
    print("[baseline] " + "  ".join(f"{k}={v:.3f}" for k, v in base.items()), flush=True)

    outp = OUTDIR / f"report_{path.stem}.json"
    outp.write_text(json.dumps(dict(feats=path.name, tag=d["tag"],
                                    results=res, baselines=base,
                                    best_val_gene_auc=best), indent=2), encoding="utf-8")
    print(f"[save] {outp}", flush=True)
    return 0


# ---------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--ckpt", default="runs/e2e_cont_rff/ckpt/final.pt")
    e.add_argument("--data", default=str(DATA))
    e.add_argument("--tag", default="real", choices=["real", "perm", "zero", "shuffle"])
    e.add_argument("--k", type=int, default=3, help="windows per gene")
    e.add_argument("--win", type=int, default=101)
    e.add_argument("--limit", type=int, default=0)
    e.add_argument("--bs", type=int, default=8, help="windows per forward")
    e.add_argument("--no-body", action="store_true", help="skip the 35B, use code vectors")
    e.add_argument("--flush", type=int, default=8000, help="rows per part file")
    e.add_argument("--question", default="In NSUN2 knockout cells, does the dsRNA level of this RNA increase? Answer yes or no.")

    t = sub.add_parser("train")
    t.add_argument("--feats", required=True)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--bs", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=1e-2)
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--struct", action="store_true", help="concat ViennaRNA features")
    t.add_argument("--zscore", action="store_true", help="per-dim standardise on train")
    t.add_argument("--readout", default="mean", choices=["last", "mean", "bins"],
                   help="mean = bag readout (Xmean); bins/last = position-aware (X)")
    t.add_argument("--use-split", action="store_true", help="use the split field in meta")
    t.add_argument("--mark", default="", help="keep only one mark (m6A / m5C)")
    t.add_argument("--keep", type=int, default=10, help="rolling snapshots to keep")

    c = sub.add_parser("compare")
    c.add_argument("--reports", nargs="+", required=True)

    a = ap.parse_args()
    if a.cmd == "compare":
        return compare(a)
    if a.cmd == "extract":
        return extract(a)
    return train(a)


if __name__ == "__main__":
    raise SystemExit(main())


def compare(a):
    """Side-by-side of two report_*.json files (real vs perm)."""
    out = {}
    for f in a.reports:
        d = json.loads((OUTDIR / f).read_text(encoding="utf-8"))
        out[f] = d
        print(f"== {f}  tag={d['tag']}")
        for sp in ("val", "test"):
            r = d["results"][sp]
            print(f"   {sp}: win_auc={r['win_auc']:.4f} gene_auc={r['gene_auc']:.4f} "
                  f"win_acc={r['win_acc']:.3f} gene_acc={r['gene_acc']:.3f} n={r['n_gene']}")
        b = d["baselines"]
        print("   baseline " + "  ".join(f"{k}={v:.3f}" for k, v in b.items()))
    if len(out) == 2:
        (k1, d1), (k2, d2) = list(out.items())
        print("\n== delta (first - second) on test")
        for m in ("win_auc", "gene_auc"):
            v = d1["results"]["test"][m] - d2["results"]["test"][m]
            print(f"   {m}: {v:+.4f}")
        print(f"\nreal={k1}  perm={k2}")
    return 0
