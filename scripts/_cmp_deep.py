import json, collections, statistics as st

fam_of, seq_of = {}, {}
with open("data/processed/rfam_seqs.jsonl", encoding="utf-8") as fh:
    for line in fh:
        r = json.loads(line)
        fam_of[r["uid"]] = r.get("family", "")
        seq_of[r["uid"]] = r.get("sequence", "")
print(f"[ref] rfam uid->family {len(fam_of):,}")

def analyze(path, has_seq=True):
    rows = 0; uids = set(); tpl = collections.Counter()
    split_fams = collections.defaultdict(set)
    split_rows = collections.Counter()
    ans_tok = 0; code_tok = 0; q_tok = 0
    alen = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line); rows += 1
            u = r.get("uid", ""); uids.add(u)
            tpl[r.get("template", "?")] += 1
            sp = r.get("split", "")
            split_rows[sp] += 1
            f = fam_of.get(u, "")
            if f: split_fams[sp].add(f)
            a = r.get("answer", "") or ""; q = r.get("question", "") or ""
            alen.append(len(a))
            ans_tok += max(1, len(a) // 4)
            q_tok += max(1, len(q) // 4)
            s = seq_of.get(u, "") if not has_seq else (r.get("sequence") or seq_of.get(u, ""))
            code_tok += max(1, len(s) // 4) if s else 0
    tr = split_fams.get("train", set()); te = split_fams.get("test", set()); va = split_fams.get("val", set())
    leak = len((tr & te) | (tr & va) | (va & te))
    tot = ans_tok + code_tok + q_tok
    return dict(rows=rows, uids=len(uids), tpl=len(tpl), tempos=split_rows,
                leak=leak, fams={k: len(v) for k, v in split_fams.items()},
                ans_tok=ans_tok, code_tok=code_tok, q_tok=q_tok, tot=tot,
                ans_pct=100 * ans_tok / max(tot, 1),
                a95=sorted(alen)[int(len(alen) * 0.95)])

for path, label in [("data/processed/rnaqa_seq30_full.jsonl", "OLD seq30_full"),
                    ("data/processed/rnaqa_cont_rff.jsonl", "OLD cont_rff"),
                    ("data/processed/rnaqa_v2_l23.jsonl", "NEW v2_l23")]:
    d = analyze(path)
    print(f"\n== {label}")
    print(f"   rows={d['rows']:,} uids={d['uids']:,} templates={d['tpl']} answer p95={d['a95']} chars")
    print(f"   split rows={dict(d['tempos'])}")
    print(f"   families/split={d['fams']}  cross-split family overlap={d['leak']}")
    print(f"   est tokens: answer={d['ans_tok']/1e6:.1f}M code={d['code_tok']/1e6:.1f}M "
          f"question={d['q_tok']/1e6:.1f}M  total={d['tot']/1e6:.1f}M")
    print(f"   answer-token share = {d['ans_pct']:.2f}%")
