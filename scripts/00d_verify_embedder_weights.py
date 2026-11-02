"""权重忠实性验证（长期留存，比 _cmp_* 一次性诊断更值得留）。

为什么需要它
------------
本项目用的两个嵌入模型里，`pubmedbert` 目录下的 `model.safetensors` 与
`tokenizer.json` **是本地生成的**（上游只给 `pytorch_model.bin` + `vocab.txt`）。
而 PROGRESS.md 坑 7 记录过一次"大小完全正确、内容全是坏的"的静默损坏事故。

所以下面这个失败模式必须能排除：
    "PubMedBERT 上 semantic F1 地板高到 0.95" —— 是真模型性质，还是权重坏了？

办法：从上游拿原始 `pytorch_model.bin`（ModelScope，带官方 sha256），逐张量比对。

本脚本对每个模型做三件事：
  1. 本地文件 sha256 vs ModelScope 声明的 sha256（上游真有的文件）；本地生成的文件标注清楚
  2. 本地 safetensors vs 上游 pytorch_model.bin 的**逐键 bitwise** 比对
  3. 两条加载路径的**前向输出**是否逐位一致

跑法：
    python scripts/00d_verify_embedder_weights.py                 # 两个模型都验
    python scripts/00d_verify_embedder_weights.py --models pubmedbert
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import rnaqwen  # noqa: E402,F401  —— env.shim

import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

MS_API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=True"

# 模型的仓库名与本地目录名，以及**本地生成**（上游没有）的文件清单
MODELS = {
    "pubmedbert": {
        "dir": ROOT / "models" / "pubmedbert",
        "repo": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        "locally_generated": ["model.safetensors", "tokenizer.json"],
        "upstream_bin": ROOT / "models" / "pubmedbert_upstream" / "pytorch_model.bin",
    },
    "bert": {
        "dir": ROOT / "models" / "bert-base-uncased",
        "repo": "google-bert/bert-base-uncased",
        "locally_generated": [],
        "upstream_bin": None,  # 上游自带 safetensors，无需比对 .bin
    },
}

TEXTS = [
    "The RNA molecule is a non-coding RNA that regulates gene expression.",
    "The protein adopts an eight-stranded beta-barrel fold.",
    "The central bank raised its policy rate by twenty-five basis points.",
    "Sear the lamb shoulder on all sides until the fat renders golden.",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ms_manifest(repo: str) -> dict:
    """ModelScope 官方清单（含官方 sha256）。拿不到就返回空 —— 别让网络问题假装成权重问题。"""
    try:
        req = urllib.request.Request(MS_API.format(repo=repo), headers={"User-Agent": "rnaqwen/0.1"})
        with urllib.request.urlopen(req, timeout=90) as r:
            payload = json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        print(f"    [warn] 拉不到 ModelScope 清单（{type(e).__name__}: {e}）—— 跳过 sha256 对照")
        return {}
    return {
        f["Path"]: {"sha256": f.get("Sha256") or "", "size": f.get("Size")}
        for f in (payload.get("Data") or {}).get("Files") or []
        if f.get("Type") != "tree"
    }


def file_check(name: str, cfg: dict) -> list[dict]:
    print(f"\n[1] 文件级 sha256 ({name})", flush=True)
    man = ms_manifest(cfg["repo"])
    rows = []
    for f in sorted(cfg["dir"].iterdir()):
        if not f.is_file():
            continue
        local = sha256(f)
        ref = man.get(f.name, {}).get("sha256", "")
        if f.name in cfg["locally_generated"]:
            verdict = "本地生成（上游无此文件）"
        elif not ref:
            verdict = "上游清单里没有这个文件名"
        elif ref.lower() == local:
            verdict = "OK  sha256 == ModelScope 声明值"
        else:
            verdict = f"MISMATCH  ms={ref[:24]}"
        rows.append({"file": f.name, "size": f.stat().st_size, "sha256": local, "verdict": verdict})
        print(f"    {f.stat().st_size:>12d} B  {f.name:<22s} {verdict}", flush=True)
    return rows


def tensor_and_forward_check(name: str, cfg: dict) -> dict:
    up = cfg["upstream_bin"]
    out: dict = {"upstream_bin": str(up) if up else None}
    if not up or not up.exists():
        print(f"\n[2/3] 跳过逐张量比对（无上游 .bin：{up}）", flush=True)
        return out

    print(f"\n[2] 逐张量比对（{name}：本地 safetensors vs 上游 .bin）", flush=True)
    from safetensors.torch import load_file

    sd_up = torch.load(up, map_location="cpu", weights_only=True)
    sd_lo = load_file(str(cfg["dir"] / "model.safetensors"))

    def norm(k: str) -> str:
        return k[5:] if k.startswith("bert.") else k

    HEAD = ("cls.", "pooler.")
    up_n = {norm(k): v for k, v in sd_up.items() if not k.startswith(HEAD)}
    lo_n = {norm(k): v for k, v in sd_lo.items()}
    common = sorted(set(up_n) & set(lo_n))
    diffs = {}
    shape_bad = []
    for k in common:
        a, b = up_n[k], lo_n[k]
        if a.shape != b.shape:
            shape_bad.append(k)
            continue
        diffs[k] = float((a.float() - b.float()).abs().max())
    exact = sum(1 for d in diffs.values() if d == 0.0)
    worst = sorted(diffs.items(), key=lambda kv: -kv[1])[:3]
    print(f"    上游 {len(sd_up)} 张量（含 MLM 头 {len(sd_up) - len(up_n)}）→ 归一后共同键 {len(common)}", flush=True)
    print(f"    bitwise 相同: {exact}/{len(diffs)}   shape 不一致: {len(shape_bad)}", flush=True)
    print(f"    max|Δ| 最大: " + ", ".join(f"{k}={v:.2e}" for k, v in worst), flush=True)
    out.update({
        "tensors_upstream": len(sd_up), "tensors_local": len(sd_lo),
        "common_keys": len(common), "bitwise_equal": exact,
        "max_abs_diff": max(diffs.values()) if diffs else None,
        "all_bitwise_equal": exact == len(diffs) and not shape_bad,
    })

    print(f"\n[3] 前向比对（{name}）", flush=True)
    tok = AutoTokenizer.from_pretrained(str(cfg["dir"]))
    enc = tok(TEXTS, return_tensors="pt", padding=True, truncation=True, max_length=512)
    m_lo = AutoModel.from_pretrained(str(cfg["dir"])).eval()
    m_up = AutoModel.from_config(m_lo.config).eval()
    report = m_up.load_state_dict({norm(k): v for k, v in sd_up.items() if not k.startswith("cls.")}, strict=False)
    print(f"    上游 .bin -> model: missing={len(report.missing_keys)} unexpected={len(report.unexpected_keys)}", flush=True)
    # load_state_dict 是 in-place 的，但漏装任何键都会让下面的比对变成"随机 vs 训练"
    assert not report.missing_keys, f"有键没装上，比对无意义: {report.missing_keys[:5]}"
    with torch.no_grad():
        h_lo = m_lo(**enc).last_hidden_state
        h_up = m_up(**enc).last_hidden_state
    same = torch.equal(h_lo, h_up)
    maxd = float((h_lo - h_up).abs().max())
    print(f"    last_hidden_state 逐位相同: {same}   max|Δ| = {maxd:.3e}", flush=True)
    out.update({"forward_bitwise_equal": same, "forward_max_abs_diff": maxd})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="bert,pubmedbert")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "embedder_weight_verification.json")
    args = ap.parse_args()

    report = {"purpose": "排除'地板高/指标异常'由权重静默损坏导致", "models": {}}
    for key in [s.strip() for s in args.models.split(",") if s.strip()]:
        cfg = MODELS.get(key)
        if not cfg:
            print(f"[skip] 未知模型 {key}")
            continue
        print("=" * 78)
        print(f"模型 {key}  ->  {cfg['repo']}")
        print("=" * 78)
        report["models"][key] = {
            "dir": str(cfg["dir"]),
            "repo": cfg["repo"],
            "files": file_check(key, cfg),
            **tensor_and_forward_check(key, cfg),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] -> {args.out}")

    ok = all(
        m.get("all_bitwise_equal", True) and m.get("forward_bitwise_equal", True)
        for m in report["models"].values()
    )
    print("[OK] 权重忠实" if ok else "[FAIL] 有权重不一致，先别用这些分数")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
