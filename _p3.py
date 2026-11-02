"""按 token 预算组批：固定 batch 在长度 16~1022 的分布上必然爆。跑完即删。

OOM 现场：batch 8 × 最长桶（1022 码 ≈ 1400 token）backward -> at::detail::empty_cuda。
桶宽 256 只能保证同桶内相似，桶与桶之间差 60 倍。

修法：每步按累计码长封顶组批，短序列自动多放、长序列自动少放。
顺带保留真正想要的 batch 上限，避免短序列组出过大的 batch。
"""
import ast
import pathlib

p = pathlib.Path("scripts/50_routeC_e2e.py")
s = p.read_text(encoding="utf-8")

old = '    ap.add_argument("--bucket", type=int, default=256,'
new = ('    ap.add_argument("--token-budget", type=int, default=6000,\n'
       '                    help="每步累计码长的上限；固定 batch 会在长序列上 OOM")\n'
       '    ap.add_argument("--bucket", type=int, default=256,')
assert old in s, "锚点1"
s = s.replace(old, new, 1)

old = """        bk = buckets[torch.randint(0, len(buckets), (1,)).item()]
        ex = [tr_ex[bk[torch.randint(0, len(bk), (1,)).item()]] for _ in range(args.batch)]"""
new = """        bk = buckets[torch.randint(0, len(buckets), (1,)).item()]
        # 按累计码长封顶组批：长序列少放，短序列多放
        ex, tot = [], 0
        while len(ex) < args.batch:
            cand = tr_ex[bk[torch.randint(0, len(bk), (1,)).item()]]
            if ex and tot + cand[2] > args.token_budget:
                break
            ex.append(cand); tot += cand[2]
        if not ex:
            ex = [tr_ex[bk[0]]]"""
assert old in s, "锚点2"
s = s.replace(old, new, 1)

p.write_text(s, encoding="utf-8")
ast.parse(s)
print("patched ok")
