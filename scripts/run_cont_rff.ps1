# cont_rff -- continue the D arm (SOS-LoRA, preset D) on a corpus that adds Rfam
# family classification and free-text annotation QA back on top of a
# sequence-task replay set.
#
# Base checkpoint: runs/e2e_D_qwen36_sos/ckpt/best.pt
#   The D arm's own early stop chose step 7500 on val CE_A.  The newest file in
#   that directory is step_010500.pt, which is what run_D_qwen36.ps1 -Resume
#   would have picked -- wrong starting point, so it is named explicitly here.
#
# Data: data/processed/rnaqa_cont_rff.jsonl
#   seqtask train strided 1,297,910 -> 250,000, val/test untouched
#   rfam train x4  (31,560 -> 126,240)
#   freetext train x4, cleaned (10,832 -> 43,328)
#   rfam keeps its own test split (7,892); freetext is pinned to train so it can
#   never be selected on.
#
# Why the striding: at natural proportions rfam+freetext are about 3 percent of
# a 1.5M-row train list, so a few thousand steps would barely sample them.  The
# mixture below puts roughly 38 / 24 / 38 percent of the answer tokens on
# seqtask / rfam / freetext, against 16 / 25 / 59 in the old C_merged run.
#
# ASCII only, no BOM: PowerShell 5 reads a BOM-less UTF-8 file as GBK.

param(
    # Step numbering -- this already bit us once, on 2026-10-16.
    # --steps in 50_routeC_e2e.py is an ABSOLUTE end step, not a count:
    #     for step in range(start_step, args.steps + 1)
    # Resuming at 7501 while passing 6000 makes that range EMPTY.  The trainer then
    # announces "training 6000 steps from 7501", takes zero steps, exits, and writes a
    # final.pt whose step label is a lie.  So this script takes the number of
    # ADDITIONAL steps and derives the absolute end itself.
    [int]$ExtraSteps = 6000,
    [string]$Tag = "cont_rff",
    [double]$SaveMinutes = 30,
    # Crash insurance.  Every attempt resumes from the newest checkpoint in
    # runs/$Tag/ckpt; the loop stops early if an attempt finished without saving one.
    [int]$MaxTries = 6,
    # Where to continue FROM, and the step stored inside it.  These used to be
    # hardcoded to the D arm's best.pt / 7500, which meant every crash threw away all
    # of this run's own checkpoints and restarted from 7500.  On 2026-10-16 that cost
    # the 440 steps already sitting in runs/e2e_cont_rff/ckpt as step_007618 .. 007913.
    # To pick up the newest one:
    #   -ResumeFrom runs/e2e_cont_rff/ckpt/step_007913.pt -ResumeStep 7913
    [string]$ResumeFrom = "runs/e2e_D_qwen36_sos/ckpt/best.pt",
    [int]$ResumeStep = 7500
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTORCH_HIP_ALLOC_CONF = "expandable_segments:True"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

$PY = "C:\ana\envs\comfyui\python.exe"
Set-Location "C:\qwen-api\RNAQwen"

$base = $ResumeFrom
if (-not (Test-Path -LiteralPath $base)) { throw "base checkpoint missing: $base" }

$Steps = $ResumeStep + $ExtraSteps

$cli = @(
  "--hid2", "data/processed/_rfa_full_hidden.f16",
  "--hid2-idx", "data/processed/_rfa_full_index.pt",
  "--base", "qwen36-35b",
  "--qwen36-mode", "int4",
  "--lora-preset", "D",
  "--pool", "4",
  "--lora-r", "8",
  "--init-code-emb", "checkpoints/fsq_full/code_emb_contentinit_d2048.pt",
  "--batch", "8",
  # (historical: the two attempts below chased the wrong quantity -- see --token-budget)
  # 768, not the D arm's 1200.  Measured on 2026-10-16 after a clean boot:
  #   baseline commit        50.3 GB / 127.65
  #   after the 35B load     81.1 GB
  #   first step peak       123.9 GB   -> the step alone wants ~42.8 GB, and roughly
  #                                      35 GB of that scales with the token count
  # It died twice at ~7520 with "Tried to allocate 20 MiB" while the device still
  # reported 30.5 GiB free -- on this iGPU the ceiling is system commit, not the
  # device budget, and build_batch packs every step up to token-budget, so the step
  # that dies is whichever one lands near the top of the range.
  # 768 scales that ~35 GB down to ~22 GB: expected peak ~111 GB, ~16 GB of headroom.
  # Cost: 6000 steps cover ~4.6 M tokens instead of ~7.2 M, so the gradient is
  # noisier.  This arm's ablation baseline is measured at its own first eval point,
  # so it stays internally consistent -- it is only the cross-arm token count that
  # no longer matches the D arm exactly.
  # 512, not 768.  At 768 the run survived 440 steps and then still died in
  # total.backward() asking for 360 MiB (device reported 33 GiB free; the real
  # ceiling on this iGPU is system commit at 127.65 GB, baseline ~50 GB).
  # Fitting the two measured points -- step cost 42.8 GB at token-budget 1200 and
  # 31 GB at 768 -- gives cost ~= 10 GB fixed + 0.0273 GB/token, so 512 lands at
  # ~24 GB and an expected peak near 114 GB, i.e. ~13 GB of headroom instead of 3.
  # 512/768 chased the wrong quantity, and both still died. What --token-budget used to
  # count was the compressed code length (clen) only -- it ignored the text tokens of
  # the question and the answer.
  #   seqtask answers are 6-8 chars, so clen ~= real length.  That is why the D arm
  #   survived 7500 steps of this exact code path.
  #   freetext answers are whole paragraphs (median 278 chars, max 551), so a row whose
  #   clen is recorded as 4 still packed 8 per step and pushed the real sequence length
  #   into the 400s.  B is the batch size, T the padded sequence length, and it is B*T
  #   that carries the activation memory.
  # Recomputed offline from the real uid->clen index and the real tokenizer, 20000
  # sampled steps, old accounting:
  #   cont_rff  budget 1200: B*T p50 1080  p99 2072  MAX 2392   (37% of steps > 1260)
  #   D arm     budget 1200: B*T p50  968  p99 1608  MAX 1656
  # So cont_rff's worst step was 44% above the worst step the D arm ever survived -- and
  # the D arm survived it with only 5-7 GiB of system memory free, so "system free" was
  # never the ceiling.  The ceiling on this iGPU is how much system memory the integrated
  # GPU is allowed to hold, and the device's own "42.99 GiB free" report is fiction.
  # 50_ now budgets by real step length (clen + text tokens) and additionally enforces
  # T*(B+1) <= budget, which hard-caps B*T at the budget:
  #   new accounting at 1024: B*T p50 ~870  p99 1024  MAX 1024   (old mean was 1089)
  # With the measured cost of roughly 10 GB + 0.0273 GB per position that is an expected
  # peak near 38 GB, below the 42.8 GB the D arm proved survivable.  Throughput cost of
  # the tighter cap is about 14%, which is cheap next to four crashes in one day.
    # 2026-10-17: 1024 died at step 11060 in total.backward() ("bad allocation") after the
  # run's own [mem] line had already fallen to 7.0 GiB free physical.  B*T 1024 is still
  # too close to the commit ceiling over a long run, so this arm continues at 512.
  "--token-budget", "512",
  "--ent-coef", "1.0",
  "--tau", "0",
  "--trim-every", "50",
  "--diag", "0",
  "--adapter", "sos",
  "--sos-k", "4",
  "--sos-gamma-max", "2.5",
  "--sos-dropout", "0.05",
  "--sos-ortho-coef", "1.0",
  "--data", "data/processed/rnaqa_cont_rff.jsonl",
  "--steps", "$Steps",
  "--log-every", "20",
  "--early-stop", "1",
  "--min-delta", "0.005",
  "--patience", "4",
  # Absolute, same semantics as --steps: patience cannot begin counting before this
  # step.  Pinning it to the end of the phase makes this arm run a fixed budget.
  # Early stopping on val seqtask CE_A is NOT informative here: the D arm stopped on
  # that very metric at step 7500, so the value we resume from is already that
  # metric's own optimum and can only stay flat or get worse.  Selecting on it would
  # truncate the observation window exactly when dilution starts to bite.  The
  # 750-step ablation is the instrument that actually sees the channel.
  "--min-steps", "$Steps",
  "--eval-every", "750",
  "--eval-n", "32",
  # 150 rather than 300: a crash costs 150 steps instead of 300, and the loop at the
  # bottom of this file restarts from the newest checkpoint.  Each checkpoint is 0.18 GB
  # and --keep-ckpt 10 trims the directory, so the disk cost does not grow.
  "--save-every", "150",
  "--save-minutes", "$SaveMinutes",
  "--keep-ckpt", "10",
  "--answer-only-loss", "1",
  # 0, not 5.  The verify steps are the memory peak of the entire run: the reference
  # pass calls model.llm(labels=...) twice, and at B=8 / T=1200 that builds
  # [8,1200,248320] fp32 logits (~9.5 GB) plus a fp32 upcast for the CE (~9.5 GB),
  # on top of the backward graph the manual loss has already retained.  On
  # 2026-10-16 that pushed system commit past 127.6 GB and the run died with
  # CUDA OOM at step 7503 while the device still reported 52 GiB free -- on an iGPU
  # the real ceiling is commit, not the reported device budget.
  #
  # This is not skipping the check on faith.  The same loss code path was verified
  # by the D arm, and the three steps that did run here show |manual-HF| of
  # 1.23e-02 / 1.54e-02 / 2.06e-02 -- BELOW HF's own run-to-run jitter of
  # 1.79e-02 / 7.21e-02 / 3.65e-02.  Misalignment is deterministic and would be off
  # on every step.  Turn it back on when the box has the 12 GB back; the embed
  # server on 8092 holds that much in session 0 and this account cannot stop it.
  #
  # 2026-10-16 evening: the reasoning above no longer holds.  The real culprit was
  # --token-budget counting clen only (see the note at --token-budget), not the transient
  # verify peak, and with B*T capped at 1024 the two [B,T,248320] fp32 logit tensors cost
  # about 1 GB.  So verify is back on at the D arm's 5 steps -- it is the only instrument
  # that catches a silent answer_only_loss misalignment, and it is better to fail there.
  "--verify-loss", "5",
  "--ablation-every", "750",
  "--resume", $base,
  "--tag", $Tag,
  "--save-dir", "runs/$Tag/artifacts"
)

Write-Host "[$Tag] resume from $base"
Write-Host "[$Tag] steps $ResumeStep -> $Steps (absolute end), fixed budget, patience pinned off"
Write-Host "[$Tag] 5-arm ablation every 750 steps -- this is the dilution readout"
Write-Host "[$Tag] model load about 9 min (35B int4, 20480 expert Linears)"

# PowerShell 5.1 turns a native command's stderr into ErrorRecord objects as soon as
# it is redirected into the pipeline, and this script runs under
# $ErrorActionPreference = "Stop".  So the very first warning the trainer printed to
# stderr aborted the wrapper -- which is what killed the 2026-10-16 run right after
# "memmap ok", on a numpy non-writable-array UserWarning from torch.
# Relax the preference for the call only: stderr still lands in the log through Tee,
# and the trainer's exit code is propagated either way.
# Auto-resume.  Four separate OOMs on 2026-10-16 each threw away a whole attempt
# because relaunching was a manual step; this loop takes the newest checkpoint and goes
# again.  It gives up when an attempt saves no checkpoint at all (that means it died on
# the first steps, so retrying would only spin) or after $MaxTries.
#
# Note that --steps is an absolute end step, so the target $Steps stays fixed across
# retries; only the resume point moves.  The extra "--resume" appended here overrides the
# one inside $cli, because argparse keeps the last occurrence of a flag.
$ErrorActionPreference = "Continue"
$curFrom, $curStep = $base, $ResumeStep
for ($try = 1; $try -le $MaxTries; $try++) {
    Write-Host "[$Tag] === attempt $try/$MaxTries : $curFrom -> absolute step $Steps ==="
    $eargs = $cli + @("--resume", $curFrom)
    # Do NOT pipe the trainer through Tee-Object.  On 2026-10-17 the trainer died with
    # "bad allocation" and the wrapper then hung for 15+ minutes instead of retrying,
    # because a grandchild process kept the inherited stdout handle open and Tee-Object
    # waited for the pipe to close.  Start-Process with file redirection has no pipe to
    # wait on, so -Wait returns as soon as the trainer exits.
    $outf = "reports/$Tag.try$try.out.log"
    $errf = "reports/$Tag.try$try.err.log"
    $eargs = @("-u", "scripts/50_routeC_e2e.py") + $eargs
    $proc = Start-Process -FilePath $PY -ArgumentList $eargs -NoNewWindow -Wait -PassThru `
                          -RedirectStandardOutput $outf -RedirectStandardError $errf
    $code = $proc.ExitCode
    # Merge into one per-attempt log.  stdout is the step stream, stderr is warnings and
    # tracebacks, so stdout-then-stderr reproduces the readable order.
    $merged = "reports/$Tag.try$try.log"
    Get-Content -LiteralPath $outf -Encoding UTF8 | Set-Content -LiteralPath $merged -Encoding UTF8
    Get-Content -LiteralPath $errf -Encoding UTF8 | Add-Content -LiteralPath $merged -Encoding UTF8
    # Keep reports/$Tag.log pointing at the current attempt: the training watchdog
    # automation reads that path.
    Copy-Item -LiteralPath $merged "reports/$Tag.log" -Force
    Write-Host "[$Tag] python exited with code $code"
    if ($code -eq 0) { exit 0 }
    # The trainer's output dir is runs/e2e_<tag>, NOT runs/<tag>: 50_routeC_e2e.py does
    #   outdir = ROOT / "runs" / f"e2e_{tag}"
    # and writes checkpoints to outdir/ckpt.  The first version of this loop looked in
    # "runs/$Tag/ckpt", found nothing, and gave up with "no checkpoint at all" on the
    # 2026-10-17 bad-allocation crash at step 11060 -- so the auto-resume never fired.
    $newest = Get-ChildItem "runs/e2e_$Tag/ckpt/step_*.pt" -ErrorAction SilentlyContinue |
              Sort-Object LastWriteTime | Select-Object -Last 1
    if (-not $newest) {
        Write-Host "[$Tag] no checkpoint at all from that attempt -- not retrying"
        break
    }
    $n = [int]($newest.BaseName -replace '^step_0*', '')
    if ($n -le $curStep) {
        Write-Host "[$Tag] no progress ($n <= $curStep) -- not retrying"
        break
    }
    $curFrom = "runs/e2e_$Tag/ckpt/$($newest.Name)"
    $curStep = $n
    Write-Host "[$Tag] restarting from $curFrom (step $n)"
}
exit 1
