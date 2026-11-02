# RNAQwen 35B Colab

`RNAQwen_v3_Full.ipynb` is the resumable Colab entry point.

## Required Drive layout

```text
MyDrive/RNAQwen/
  RNAQwen_code.zip
  assets/
    _fsq_full_pos.pt
    _rfam_hidden.f16
    _rfam_index.pt
    rnaqa_v3_l234.jsonl
```

`RNAQwen_code.zip` is produced by:

```powershell
powershell -File colab/build_bundle.ps1
```

The bundle contains source, scripts and the small FSQ/code-embedding
checkpoints. It excludes model weights, full hidden-state caches, old runs and
the local virtual environment.

## Runtime

- Colab GPU: A100 40GB or better.
- The first run downloads `Qwen/Qwen3.6-35B-A3B` from ModelScope to local
  Colab disk.
- Training checkpoints are written under
  `MyDrive/RNAQwen/runs/e2e_v3_full/` through a symlink.
- Run the 10-step cell first. If it passes, run the full cell.
- After a disconnect, reconnect, rerun setup and execute the full cell again.
  It discovers the newest `step_*.pt` and resumes automatically.

## Environment override

The training code now reads:

```text
RNAQ_QWEN36_DIR
RNAQ_QWEN8_DIR
```

This keeps the existing Windows behavior while allowing Linux paths in Colab.
