# Raise C: page file from 32 GB to 64 GB.
#
# Why: the 35B load needs more commit charge than this box has. Commit limit is
# 95.6 GB (physical 63.6 + fixed page file 32). Moving the int4 weights onto the
# APU device memory alone accounts for ~18 GB of that, and the loader used to
# burn another 67 GB on 26 open safetensors shard mappings (that part is fixed,
# see src/rnaqwen/qwen36.py ShardReader). Symptom before the fix: the load dies
# around layer 30-39 with "os error 1455" or a bare 0xC0000005.
#
# Needs an elevated window. Log goes to reports/pagefile_fix.log whatever the
# window does, so the result is checkable from a normal shell.
#
# ASCII ONLY: Windows PowerShell 5.1 reads a .ps1 without a BOM as ANSI/GBK.
# Non-ASCII comments here mangle the parser (hit that on 2026-10-14).

$sizeMB = 65536     # 64 GB -> commit limit becomes ~128 GB
$log = Join-Path (Join-Path $PSScriptRoot "..\..") "reports\pagefile_fix.log"

function W($m) {
    "$((Get-Date).ToString('HH:mm:ss'))  $m" | Tee-Object -FilePath $log -Append
}

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
W "=== start ==="
W "running elevated: $isAdmin"

$cur = Get-CimInstance Win32_PageFileSetting | Where-Object { $_.Name -like '*pagefile.sys' }
if (-not $cur) {
    W "** no pagefile setting found, abort"
    exit 2
}
W "before: Name=$($cur.Name)  Initial=$($cur.InitialSize) MB  Max=$($cur.MaximumSize) MB"
W "AutomaticManagedPagefile=$((Get-CimInstance Win32_ComputerSystem).AutomaticManagedPagefile)"

# Try both APIs. Win32_PageFileSetting is an old WMI class; Get-WmiObject + Put()
# works everywhere, Set-CimInstance is the modern spelling and gets refused on
# some builds.
$done = $false
try {
    $pf = Get-WmiObject Win32_PageFileSetting | Where-Object { $_.Name -like '*pagefile.sys' }
    $pf.InitialSize = $sizeMB
    $pf.MaximumSize = $sizeMB
    $r = $pf.Put()
    W "Get-WmiObject + Put() ok (ReturnValue=$($r.ReturnValue))"
    $done = $true
} catch {
    W "Get-WmiObject + Put() failed: $($_.Exception.Message)"
}
if (-not $done) {
    try {
        Set-CimInstance -InputObject $cur -Property @{
            InitialSize = $sizeMB
            MaximumSize = $sizeMB
        } -ErrorAction Stop
        W "Set-CimInstance ok"
        $done = $true
    } catch {
        W "Set-CimInstance failed: $($_.Exception.Message)"
    }
}

$new = Get-CimInstance Win32_PageFileSetting | Where-Object { $_.Name -like '*pagefile.sys' }
W "after:  Initial=$($new.InitialSize) MB  Max=$($new.MaximumSize) MB"

$u = Get-CimInstance Win32_PageFileUsage
W "allocated now: $($u.AllocatedBaseSize) MB   in use: $($u.CurrentUsage) MB   peak: $($u.PeakUsage) MB"

$os = Get-CimInstance Win32_OperatingSystem
W ("commit limit {0:N1} GB   used {1:N1} GB   free {2:N1} GB" -f `
    ($os.TotalVirtualMemorySize / 1MB), `
    (($os.TotalVirtualMemorySize - $os.FreeVirtualMemory) / 1MB), `
    ($os.FreeVirtualMemory / 1MB))

if ($new.InitialSize -eq $sizeMB) {
    W "=== done. If 'allocated now' is still 32768 MB, a REBOOT is required ==="
} else {
    W "=== done, but the setting did NOT change (see messages above) ==="
}
