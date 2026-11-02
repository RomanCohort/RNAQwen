# Auto-retry wrapper.  This environment silently kills processes (observed 4 times),
# so long jobs must be "die and restart" tolerant.
#
# The wrapped command MUST be idempotent / resumable (i.e. it takes a --resume flag).
#
# NOTE: ASCII only on purpose.  Windows PowerShell 5.1 reads .ps1 as ANSI unless the file
# carries a UTF-8 BOM, so non-ASCII comments here turn into parse errors.
#
# Usage: powershell -File _retry.ps1 -Cmd "<command>" -Log <logfile> -MaxTries 20

param(
    [Parameter(Mandatory = $true)][string]$Cmd,
    [Parameter(Mandatory = $true)][string]$Log,
    [int]$MaxTries = 20,
    [int]$SleepSec = 20
)

$ErrorActionPreference = "Continue"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONIOENCODING = "utf-8"

function Note([string]$msg) {
    "[retry] $msg  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -Append $Log -Encoding utf8
}

Note "wrapping: $Cmd"
for ($i = 1; $i -le $MaxTries; $i++) {
    Note "attempt $i/$MaxTries"
    $p = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", $Cmd -PassThru -WindowStyle Hidden
    $p.WaitForExit()
    Note "exit code $($p.ExitCode)"
    if ($p.ExitCode -eq 0) {
        Note "completed successfully"
        break
    }
    Note "failed; sleeping $SleepSec s before retry"
    Start-Sleep -Seconds $SleepSec
}
Note "wrapper done"
