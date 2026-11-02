# Launch card: reports/B6_run_card.md
#
# Resumes the D_qwen36_sos checkpoint (step 7500) on the mixed dataset:
#   half class questions (7 classes), half the ten local-character templates.
# --steps is an ABSOLUTE target (range(start_step, steps+1)), so 9000 means
# 1500 further steps at ~7.4 s/step on this box.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"
# The OOM message points here; on this APU the binding ceiling is system commit,
# but expandable_segments still removes one source of fragmentation.
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$py = "C:\ana\envs\comfyui\python.exe"

& $py scripts\50_routeC_e2e.py `
  --base qwen36-35b --qwen36-mode int4 --lora-preset D `
  --adapter sos --sos-k 4 --sos-gamma-max 2.5 --sos-dropout 0.05 --sos-ortho-coef 1.0 `
  --rna-emb table --pool 4 --lora-r 8 --batch 8 --lr 2e-4 `
  --data data/processed/rnaqa_class_mix.jsonl `
  --hid2 data/processed/_rfa_full_hidden.f16 `
  --hid2-idx data/processed/_rfa_full_index.pt `
  --init-code-emb checkpoints/fsq_full/code_emb_contentinit_d2048.pt `
  --ent-coef 1.0 --commit-coef 0.01 --ortho-coef 0.01 `
  --answer-only-loss 1 --token-budget 384 --bucket 256 --trim-every 50 `
  --steps 9000 --eval-every 250 --eval-n 8 --log-every 20 `
  --early-stop 0 --save-every 100 --keep-ckpt 6 --save-minutes 10 `
  --ablation-every 0 --diag 0 `
  --resume runs/e2e_D_qwen36_sos/ckpt/final.pt `
  --tag class_mix

exit $LASTEXITCODE
