param(
  [string]$Url   = "ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz",
  [string]$Dest  = "C:\qwen-api\RNAQwen\data\raw\rnacentral\rnacentral_active.fasta.gz",
  [long]  $Total = 11858076614,
  [string]$Log   = "C:\qwen-api\RNAQwen\logs\rnacentral_single.log"
)
$mutex = New-Object System.Threading.Mutex($false, "Global\RNAcentralDownload")
if (-not $mutex.WaitOne(0)) { exit 0 }
$dir = Split-Path $Dest
New-Item -ItemType Directory -Force -Path $dir | Out-Null
function Log([string]$m) {
  $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m
  Add-Content -Path $Log -Value $line -Encoding UTF8
}
Log "resume start total=$Total"
$round = 0
while ($true) {
  $round++
  $have = if (Test-Path $Dest) { (Get-Item $Dest).Length } else { 0 }
  if ($have -ge $Total) { break }
  Log ("round {0}: have={1:N2} GB" -f $round, ($have/1GB))
  & curl.exe -s --retry 10 --retry-delay 15 --retry-all-errors --max-time 3600 -C - -o $Dest $Url
  $now = if (Test-Path $Dest) { (Get-Item $Dest).Length } else { 0 }
  Log ("after round {0}: {1:N2} GB (delta {2:N1} MB)" -f $round, ($now/1GB), (($now-$have)/1MB))
  if ($now -le $have) { Start-Sleep -Seconds 120 }
}
$final = (Get-Item $Dest).Length
Log "done final=$final expected=$Total"
if ($final -eq $Total) { Log "OK complete" } else { Log "MISMATCH" }
