$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $PSScriptRoot "RNAQwen_code.zip"
$stage = Join-Path $env:TEMP ("rnaqwen-colab-" + [guid]::NewGuid().ToString("N"))

try {
    New-Item -ItemType Directory -Path $stage | Out-Null

    foreach ($name in @("src", "scripts", "configs", ".venv")) {
        if ($name -eq ".venv") { continue }
        $src = Join-Path $root $name
        if (Test-Path -LiteralPath $src) {
            Copy-Item -LiteralPath $src -Destination $stage -Recurse
        }
    }

    foreach ($name in @(
        "README.md",
        "RUN.md",
        "PLAN_v2.md",
        "PROGRESS.md"
    )) {
        $src = Join-Path $root $name
        if (Test-Path -LiteralPath $src) {
            Copy-Item -LiteralPath $src -Destination $stage
        }
    }

    foreach ($name in @("requirements.txt", "README.md")) {
        $src = Join-Path $PSScriptRoot $name
        Copy-Item -LiteralPath $src -Destination (Join-Path $stage $name)
    }

    $ckSrc = Join-Path $root "checkpoints\fsq_stream"
    $ckDst = Join-Path $stage "checkpoints\fsq_stream"
    New-Item -ItemType Directory -Path $ckDst -Force | Out-Null
    foreach ($name in @("tokenizer.pt", "code_emb_contentinit.pt")) {
        $src = Join-Path $ckSrc $name
        if (-not (Test-Path -LiteralPath $src)) {
            throw "Missing checkpoint: $src"
        }
        Copy-Item -LiteralPath $src -Destination $ckDst
    }

    if (Test-Path -LiteralPath $out) {
        Remove-Item -LiteralPath $out
    }
    Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $out
    Get-Item -LiteralPath $out | Select-Object FullName, Length, LastWriteTime
}
finally {
    if (Test-Path -LiteralPath $stage) {
        Remove-Item -LiteralPath $stage -Recurse -Force
    }
}
