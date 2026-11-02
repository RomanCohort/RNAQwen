#!/bin/bash
cd /mnt/c/qwen-api/RNAQwen/data/external/rfam

# 已知的 tRNA（酵母 tRNA-Phe）—— 如果 Rfam 索引是好的，这条必须命中 RF00005
cat > known.fa <<'EOF'
>known_tRNA
GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA
>known_5S
GGUGCGAUCAUACGUAGGAUCCACGCGAAAGUACUCGGUCGAUACCGGCGCACGG
EOF

echo "=== 已知序列 cmsearch（应该命中 RF00005 tRNA）==="
cmsearch --rfam --noali --cpu 4 -o known.out Rfam.cm known.fa 2>&1 | tail -2
echo "--- 命中行 ---"
grep -E "RF[0-9]{5}" known.out | head -10
echo "--- 汇总 ---"
grep -E "Total CM hits|Query model" known.out | tail -3

echo
echo "=== 我们序列的前 5 条长什么样 ==="
head -10 test200.fa
echo
echo "=== 长度分布（前 200 条）==="
python3 - <<'PY'
seqs={}
name=None
for line in open('test200.fa'):
    line=line.strip()
    if line.startswith('>'):
        name=line[1:]; seqs[name]=''
    elif name:
        seqs[name]+=line
L=sorted(len(s) for s in seqs.values())
print(f'条数 {len(L)}  长度 min/中位/max = {L[0]}/{L[len(L)//2]}/{L[-1]}')
import collections
c=collections.Counter()
for s in seqs.values():
    c.update(set(s))
print('碱基组成:', dict(c.most_common()))
PY

echo
echo "=== cmscan 正确用法（--fmt 2 不是 cmscan 的选项）==="
cmscan -h 2>&1 | grep -E "^\s+-(o|E|Z|fmt|tblout|rfam)" | head -8
