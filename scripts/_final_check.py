"""终验：文件清单 / sha256 比对 / safetensors 与 bin 两条加载路径输出一致性。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _dtensor_shim  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

MS_API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=True"
DIRS = {
    "bert-base-uncased": (Path(r"C:\qwen-api\RNAQwen\models\bert-base-uncased"), "google-bert/bert-base-uncased"),
    "pubmedbert": (
        Path(r"C:\qwen-api\RNAQwen\models\pubmedbert"),
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    ),
}
GEN = {"model.safetensors", "tokenizer.json"}
TEXT = "the RNA regulates gene expression."


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ms_hashes(repo: str) -> dict:
    req = urllib.request.Request(MS_API.format(repo=repo), headers={"User-Agent": "rnaqwen/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.loads(r.read().decode())
    return {
        f["Path"]: (f.get("Sha256") or f.get("Sha256Hash") or "")
        for f in (payload.get("Data") or {}).get("Files") or []
        if f.get("Type") != "tree" and f.get("Size") is not None
    }


for name, (d, repo) in DIRS.items():
    print("=" * 76)
    print(f"[{name}]  <- {repo}")
    declared = ms_hashes(repo)
    for f in sorted(d.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(d).as_posix()
        local = sha256(f)
        ref = declared.get(rel, "")
        if not ref:
            tag = "本地生成 (上游仓库本身无此文件)"
        elif ref.lower() == local:
            tag = "sha256 == ModelScope 声明值 OK"
        else:
            tag = f"!!! MISMATCH ms={ref[:16]}"
        print(f"  {f.stat().st_size:>12d} B  {rel:<22s} {tag}")

    print(f"  -- 加载路径一致性 ({TEXT!r}) --")
    tok = AutoTokenizer.from_pretrained(str(d))
    enc = tok(TEXT, return_tensors="pt")
    m_st = AutoModel.from_pretrained(str(d))  # 有 safetensors 时默认走它
    with torch.no_grad():
        o_st = m_st(**enc).last_hidden_state
    print(f"     默认(safetensors) hidden={m_st.config.hidden_size} shape={tuple(o_st.shape)}")
    if (d / "pytorch_model.bin").exists():
        m_bin = AutoModel.from_pretrained(str(d), use_safetensors=False)
        with torch.no_grad():
            o_bin = m_bin(**enc).last_hidden_state
        print(f"     强制 .bin        hidden={m_bin.config.hidden_size} shape={tuple(o_bin.shape)}")
        print(f"     两条路径输出逐位一致: {torch.equal(o_st, o_bin)}  (max|Δ|={float((o_st - o_bin).abs().max())})")
    else:
        print("     (目录内无 pytorch_model.bin，跳过双路径比对)")
    print(f"     tokenizer: {type(tok).__name__}  input_ids={enc['input_ids'].tolist()[0]}")
print("=" * 76)
