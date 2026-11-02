$ErrorActionPreference = "Continue"

# --- cleanup previous attempts ---
Get-Process aria2c -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Get-ChildItem "C:\qwen-api\RNAQwen" -Filter "322723f496949a6075148e2c6748c4aaa0f42c38f3905611a068555906f9af98*" -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

$dest  = "C:\qwen-api\RNAQwen\models\rnafm"
$tmp   = "C:\qwen-api\RNAQwen\_rnafm_parts"
$total = 398117588
$N     = 32
$chunk = [math]::Ceiling($total / $N)
$hf    = "https://huggingface.co/multimolecule/rnafm/resolve/main/model.safetensors"
$mir   = "https://hf-mirror.com/multimolecule/rnafm/resolve/main/model.safetensors"

Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Remove-Item "$dest\model.safetensors" -Force -ErrorAction SilentlyContinue
Remove-Item "$dest\model.safetensors.aria2" -Force -ErrorAction SilentlyContinue

function Start-Part($i) {
    $s = $i * $chunk
    $e = [math]::Min($s + $chunk - 1, $total - 1)
    if ($s -gt $e) { return $null }
    $url = if ($i % 2 -eq 0) { $mir } else { $hf }
    $out = Join-Path $tmp ("part_{0:D2}.bin" -f $i)
    $cargs = @("-sS", "-L", "-r", "$s-$e", "--retry", "20", "--retry-delay", "2",
              "--connect-timeout", "20", "--speed-limit", "800", "--speed-time", "60",
              "--max-time", "3600", "-o", $out, $url)
    $p = Start-Process -FilePath "curl.exe" -ArgumentList $cargs -NoNewWindow -PassThru
    return [pscustomobject]@{ Idx = $i; Proc = $p; Out = $out; Start = $s; End = $e; Expect = ($e - $s + 1) }
}

function Test-Running($procs) {
    $n = 0
    foreach ($x in $procs) {
        try { if (-not $x.Proc.HasExited) { $n++ } } catch { }
    }
    return $n
}

$sw = [System.Diagnostics.Stopwatch]::StartNew()
$parts = @()
for ($i = 0; $i -lt $N; $i++) { $p = Start-Part $i; if ($p) { $parts += $p } }
Write-Output ("[launch] {0} parts x {1:N0} bytes, {2} routes" -f $parts.Count, $chunk, 2)

# --- monitor ---
$last = 0
while ($true) {
    Start-Sleep -Seconds 20
    $running = Test-Running $parts
    $got = (Get-ChildItem "$tmp\part_*.bin" -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    if (-not $got) { $got = 0 }
    $inst = ($got - $last) / 20 / 1MB
    $last = $got
    $eta = if ($inst -gt 0.01) { ($total - $got) / 1MB / $inst } else { -1 }
    Write-Output ("[mon] {0:N1}/{1:N1} MB ({2:N1}%)  running={3}  {4:N2} MB/s  ETA {5:N1} min  elapsed {6:N1} min" -f `
        ($got/1MB), ($total/1MB), (100*$got/$total), $running, $inst, ($eta/60), ($sw.Elapsed.TotalMinutes))
    if ($running -eq 0) { break }
    if ($sw.Elapsed.TotalMinutes -gt 45) { Write-Output "[mon] TIMEOUT"; break }
}

# --- retry incomplete parts ---
for ($round = 1; $round -le 6; $round++) {
    $bad = @($parts | Where-Object {
        $len = if (Test-Path $_.Out) { (Get-Item $_.Out).Length } else { 0 }
        $len -ne $_.Expect
    })
    if ($bad.Count -eq 0) { Write-Output "[retry] all parts complete"; break }
    Write-Output ("[retry {0}] {1} incomplete parts: {2}" -f $round, $bad.Count, (($bad | ForEach-Object { $_.Idx }) -join ","))
    Remove-Item "$tmp\*.aria2" -Force -ErrorAction SilentlyContinue
    $new = @()
    foreach ($b in $bad) {
        Remove-Item $b.Out -Force -ErrorAction SilentlyContinue
        $p = Start-Part $b.Idx
        if ($p) { $new += $p }
    }
    $new | ForEach-Object { $_ } | Out-Null
    while ((Test-Running $new) -gt 0) { Start-Sleep -Seconds 10 }
    Start-Sleep -Seconds 2
}

# --- concatenate in index order ---
Write-Output "[cat] concatenating..."
$outPath = Join-Path $dest "model.safetensors"
$fs = [System.IO.File]::Create($outPath)
$buf = New-Object byte[] (4MB)
$ok = $true
foreach ($p in ($parts | Sort-Object Idx)) {
    if (-not (Test-Path $p.Out)) { Write-Output "[cat] MISSING $($p.Out)"; $ok = $false; break }
    $len = (Get-Item $p.Out).Length
    if ($len -ne $p.Expect) { Write-Output ("[cat] BAD part {0}: {1} != {2}" -f $p.Idx, $len, $p.Expect); $ok = $false; break }
    $src = [System.IO.File]::OpenRead($p.Out)
    while (($n = $src.Read($buf, 0, $buf.Length)) -gt 0) { $fs.Write($buf, 0, $n) }
    $src.Close()
}
$fs.Close()

$size = (Get-Item $outPath).Length
Write-Output ("[cat] done size={0:N0} expected={1:N0} match={2}" -f $size, $total, ($size -eq $total))

# --- verify sha256 ---
$h = (Get-FileHash $outPath -Algorithm SHA256).Hash.ToLower()
$want = "ec69333eb894dbefb580e690b0cbe1cbae9845fefd2bc78da914cd14226a2828"
Write-Output "[sha256] got  $h"
Write-Output "[sha256] want $want"
Write-Output ("[sha256] MATCH={0}" -f ($h -eq $want))
Write-Output ("[time] total {0:N1} min" -f $sw.Elapsed.TotalMinutes)
if ($ok -and $size -eq $total -and $h -eq $want) { Write-Output "[RESULT] SUCCESS" } else { Write-Output "[RESULT] FAILURE" }
