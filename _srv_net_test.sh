#!/bin/bash
# 在服务器上跑：测 RNAcentral API 通不通
echo "=== UA = curl 默认 ==="
curl -sS -o /dev/null --max-time 20 -w "  http=%{http_code}  size=%{size_download}B  speed=%{speed_download}B/s\n" \
  "https://rnacentral.org/api/v1/rna/?external_id=NR_136892.1" || true

echo "=== UA = 浏览器 ==="
curl -sS -o /tmp/rna1.json --max-time 25 -A "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36" \
  -H "Accept: application/json" \
  -w "  http=%{http_code}  size=%{size_download}B  speed=%{speed_download}B/s\n" \
  "https://rnacentral.org/api/v1/rna/?external_id=NR_136892.1" || true
echo "--- 返回前 600 字符 ---"
head -c 600 /tmp/rna1.json 2>/dev/null; echo

echo
echo "=== 试另一种端点写法 ==="
for u in \
  "https://rnacentral.org/api/v1/rna/URS000075B951/?format=json" \
  "https://rnacentral.org/api/v1/rna/?external_id=NR_136892.1&format=json" ; do
  code=$(curl -sS -o /tmp/rna2.json --max-time 20 -A "Mozilla/5.0" -w "%{http_code}" "$u" || echo ERR)
  echo "  [$code] $u"
done
head -c 400 /tmp/rna2.json 2>/dev/null; echo

echo
echo "=== HF 更实际的测法（跟一个重定向，看能不能下到大文件）==="
curl -sSL -o /dev/null --max-time 25 -r 0-33554431 \
  -w "  HF range: http=%{http_code}  speed=%{speed_download}B/s  size=%{size_download}B\n" \
  "https://huggingface.co/multimolecule/rnafm/resolve/main/model.safetensors" || true
