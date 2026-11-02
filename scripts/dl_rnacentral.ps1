param(
  [string]$Url   = "ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz",
  [string]$Dest  = "C:\qwen-api\RNAQwen\data\raw\rnacentral\rnacentral_active.fasta.gz",
  [long]  $Total = 11858076614,
  [int]   $Parts = 16,
  [string]$Log   = "C:\qwen-api\RNAQwen\logs\rnacentral_dl.log"
)
$ErrorActionPreference = "Continue"
$dir = Split-Path $Dest
$tmp = Join-Path $dir "_parts"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

function Log([string]$m) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  Add-Content -Path $Log -Value $line -Encoding UTF8
}
function PartLen($f) {
  if (Test-Path $f) { return (Get-Item $f).Length } else { return 0 }
}
function AppendFile($src, $dst) {
  $a = [System.IO.File]::Open($src, 'Open', 'Read', 'ReadWrite')
  $b = [System.IO.File]::Open($dst, 'Append', 'Write', 'ReadWrite')
  $buf = New-Object byte[] 4194304
  while (($n = $a.Read($buf, 0, $buf.Length)) -gt 0) { $b.Write($buf, 0, $n) }
  $b.Flush(); $b.Close(); $a.Close()
}

$chunk = [long][Math]::Ceiling($Total / $Parts)
Log "start total=$Total parts=$Parts chunk=$chunk"

$ranges = New-Object System.Collections.ArrayList
for ($i = 0; $i -lt $Parts; $i++) {
  $s = [long]($i * $chunk)
  $e = [long][Math]::Min($s + $chunk - 1, $Total - 1)
  if ($s -gt $Total - 1) { break }
  $f = Join-Path $tmp ("p{0:D2}.bin" -f $i)
  [void]$ranges.Add([pscustomobject]@{ i = $i; s = $s; e = $e; f = $f })
}

$round = 0
while ($true) {
  $round++
  $got = [long]0
  $want = [long]0
  $pend = @()
  foreach ($r in $ranges) {
    $w = $r.e - $r.s + 1
    $want += $w
    $h = [long](PartLen $r.f)
    $got += $h
    if ($h -lt $w) { $pend += ,@{ r = $r; have = $h } }
  }
  Log ("round {0}: {1:N3}/{2:N3} GB  pending={3}" -f $round, ($got/1GB), ($want/1GB), $pend.Count)
  if ($pend.Count -eq 0) { break }

  $procs = @()
  foreach ($p in $pend) {
    $r = $p.r
    $from = $r.s + $p.have
    $t = "$($r.f).tmp"
    if (Test-Path $t) { Remove-Item -LiteralPath $t -Force -ErrorAction SilentlyContinue }
    $a = @("-s", "--retry", "20", "--retry-delay", "10", "--retry-all-errors",
           "--max-time", "3600", "-r", "$from-$($r.e)", "-o", $t, $Url)
    $pr = Start-Process -FilePath "curl.exe" -ArgumentList $a -PassThru -WindowStyle Hidden
    $procs += [pscustomobject]@{ p = $pr; tmp = $t; part = $r.f }
  }
  while ($true) {
    $alive = @($procs | Where-Object { -not $_.p.HasExited })
    if ($alive.Count -eq 0) { break }
    Start-Sleep -Seconds 120
  }
  foreach ($x in $procs) {
    if ((Test-Path $x.tmp) -and ((Get-Item $x.tmp).Length -gt 0)) {
      AppendFile $x.tmp $x.part
      Remove-Item -LiteralPath $x.tmp -Force -ErrorAction SilentlyContinue
    }
  }
  Start-Sleep -Seconds 5
}

Log "all parts complete; concatenating"
if (Test-Path $Dest) { Remove-Item -LiteralPath $Dest -Force }
$dstFS = [System.IO.File]::Open($Dest, 'Create', 'Write')
foreach ($r in $ranges) {
  $src = [System.IO.File]::OpenRead($r.f)
  $src.CopyTo($dstFS, 4194304)
  $src.Close()
}
$dstFS.Close()
$final = (Get-Item $Dest).Length
Log "final=$final expected=$Total"
if ($final -eq $Total) { Log "OK complete" } else { Log "MISMATCH" }
