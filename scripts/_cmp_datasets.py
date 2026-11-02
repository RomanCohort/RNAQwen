import json, collections, statistics as st, sys

def stats(f):
    n = 0
    uids = set()
    tpl = collections.Counter()
    ql, al = [], []
    per = collections.Counter()
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            n += 1
            u = r.get("uid", "")
            uids.add(u)
            tpl[r.get("template", "?")] += 1
            q = r.get("question", "") or ""
            a = r.get("answer", "") or ""
            ql.append(len(q)); al.append(len(a)); per[u] += 1
    return dict(rows=n, uids=len(uids), per=round(n / max(len(uids), 1), 1),
                tpl=len(tpl), q=int(st.median(ql)), a=int(st.median(al)),
                a95=sorted(al)[int(len(al) * 0.95)], top=tpl.most_common(4))

for f, label in [("data/processed/rnaqa_seq30_full.jsonl", "OLD seq30_full"),
                 ("data/processed/rnaqa_cont_rff.jsonl", "OLD cont_rff"),
                 ("data/processed/rnaqa_v2_l23.jsonl", "NEW v2_l23")]:
    s = stats(f)
    print(f"{label}: rows={s['rows']:,} uids={s['uids']:,} q/uid={s['per']} "
          f"templates={s['tpl']} qchars={s['q']} achars={s['a']} a95={s['a95']}")
    for t, c in s["top"]:
        print(f"    {c:>8,}  {t[:70]}")
