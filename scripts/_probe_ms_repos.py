"""快速探测一批 ModelScope repo id 是否存在、有哪些文件。"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request

CANDIDATES = [
    "google-bert/bert-base-uncased",
    "AI-ModelScope/bert-base-uncased",
    "dienstag/bert-base-uncased",
    "tiansz/bert-base-uncased",
    "iic/nlp_bert_backbone_base_std",
    "AI-ModelScope/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "dienstag/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "AI-ModelScope/pubmedbert-base-uncased-abstract-fulltext",
    "google-bert/bert-base-uncased-vocab",
]

WANT = {"config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json"}
API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=True"


def probe(repo: str):
    url = API.format(repo=repo)
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "rnaqwen/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        return repo, "ERR", f"{type(exc).__name__}: {str(exc)[:80]}", []
    data = payload.get("Data") or {}
    files = data.get("Files") or []
    rows = [(f["Path"], int(f["Size"])) for f in files if f.get("Type") != "tree" and f.get("Size") is not None]
    if not rows:
        return repo, "EMPTY", "", []
    names = {p.split("/")[-1] for p, _ in rows}
    st = "OK" if WANT <= names and any(p.endswith("model.safetensors") for p, _ in rows) else "PARTIAL"
    return repo, st, "", rows


def main() -> int:
    repos = sys.argv[1:] or CANDIDATES
    for repo in repos:
        r, status, err, rows = probe(repo)
        total = sum(s for _, s in rows)
        print(f"{status:8s} {r:70s} files={len(rows):3d} total={total/1e6:8.1f}MB {err}")
        if status in ("OK", "PARTIAL"):
            for p, s in sorted(rows, key=lambda x: -x[1])[:12]:
                print(f"           {s/1e6:9.2f}MB  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
