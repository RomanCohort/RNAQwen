"""模型下载器 —— 复现所需权重一次性拉齐。

清单与用途：
  Qwen/Qwen3-8B-Instruct            底座 LLM（PLAN §2；hidden=4096，与 Llama-3-8B 同维，投影层零改动）
  multimolecule/rnafm               序列编码器（冻结）；hidden=640，12 层，max_position_embeddings=1026
  google-bert/bert-base-uncased     评测嵌入 S_BERT
  microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext   评测嵌入 S_Pub (PubMedBERT)

跑法：
    python scripts/00_download_models.py                # 全部
    python scripts/00_download_models.py --only rnafm   # 只下一个

网络：**默认直连 huggingface.co**。
实测 hf-mirror.com 不能用 —— `huggingface_hub` 的 HEAD 元数据校验拿不到
`x-repo-commit` / etag，会抛
`FileMetadataError: Distant resource does not seem to be on huggingface.co`。
本机有 TUN 代理，直连 huggingface.co 正常（resolve 200，实测可下大文件）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS: dict[str, dict] = {
    "qwen3-8b": {
        "repo": "Qwen/Qwen3-8B",
        "why": "底座 LLM（PLAN §2；hidden=4096，与 Llama-3-8B 同维，投影层零改动）",
        "allow": ["*.json", "*.safetensors", "*.txt", "*.model"],
    },
    "rnafm": {
        "repo": "multimolecule/rnafm",
        "why": "RNA-FM 序列编码器（冻结）",
        # 跳过 pytorch_model.bin（与 model.safetensors 同一份权重，398MB，别下两遍）
        "allow": ["*.json", "*.txt", "*.safetensors", "*.md", ".gitattributes"],
    },
    "bert": {
        "repo": "google-bert/bert-base-uncased",
        "why": "评测嵌入 S_BERT",
        "allow": ["*.json", "*.txt", "*.safetensors", "*.model"],
    },
    "pubmedbert": {
        "repo": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "why": "评测嵌入 S_Pub (PubMedBERT)",
        "allow": ["*.json", "*.txt", "*.safetensors", "*.model"],
    },
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None, help="只下指定 key（可多次）")
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for k, v in MODELS.items():
            print(f"{k:20s} {v['repo']:70s} {v['why']}")
        return 0

    os.environ["HF_ENDPOINT"] = args.endpoint
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    from huggingface_hub import snapshot_download

    keys = args.only or list(MODELS)
    rc = 0
    for key in keys:
        if key not in MODELS:
            print(f"[skip] unknown key: {key}")
            rc = 1
            continue
        spec = MODELS[key]
        print(f"[get ] {key:20s} {spec['repo']}  ({spec['why']})", flush=True)
        t0 = time.time()
        try:
            path = snapshot_download(
                repo_id=spec["repo"],
                allow_patterns=spec.get("allow"),
                max_workers=4,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[fail] {key}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            rc = 1
            continue
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        print(f"[ok  ] {key:20s} {human(size):>9s}  {time.time() - t0:6.1f}s  -> {path}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
