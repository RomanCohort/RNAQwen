param(
  [string]$Url  = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE242nnn/GSE242724/suppl/GSE242724_ct.FR.mpmat.txt.gz",
  [string]$Dest = "C:\qwen-api\RNAQwen\data\raw\immune\GSE242724_ct.FR.mpmat.txt.gz",
  [long]  $Total = 307546033,
  [string]$Log  = "C:\qwen-api\RNAQwen\logs\gse242724_dl.log"
)
function Log([string]$m) {
  Add-Content -Path $Log -Value ("[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) -Encoding UTF8
}
Log "start total=$Total"
$round = 0
while ($true) {
  $round++
  $have = if (Test-Path $Dest) { (Get-Item $Dest).Length } else { 0 }
  if ($have -ge $Total) { break }
  Log ("round {0}: have={1:N1} MB" -f $round, ($have/1MB))
  & curl.exe -s --retry 10 --retry-delay 10 --retry-all-errors --max-time 5400 -C - -o $Dest $Url
  $now = if (Test-Path $Dest) { (Get-Item $Dest).Length } else { 0 }
  Log ("after: {0:N1} MB (delta {1:N1} MB)" -f ($now/1MB), (($now-$have)/1MB))
  if ($now -le $have) { Start-Sleep -Seconds 60 }
}
Log ("done " + (Get-Item $Dest).Length)
