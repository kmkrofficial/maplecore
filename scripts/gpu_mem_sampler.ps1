# Samples dedicated and shared (system-RAM-backed) GPU memory of every process using > 50 MB, every N seconds,
# appending to a CSV. Shared usage > 0 for a training process means Windows is paging its VRAM into system memory.
# Usage: powershell -File scripts/gpu_mem_sampler.ps1 -Out logs/gpu_mem.csv -Seconds 30 -Hours 6
param([string]$Out = "logs/gpu_mem.csv", [int]$Seconds = 30, [double]$Hours = 6)
if (-not (Test-Path $Out)) { "time,pid,process,dedicated_mb,shared_mb" | Out-File -Encoding utf8 $Out }
$end = (Get-Date).AddHours($Hours)
while ((Get-Date) -lt $end) {
  $t = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
  $s = (Get-Counter "\GPU Process Memory(*)\Dedicated Usage", "\GPU Process Memory(*)\Shared Usage" -ErrorAction SilentlyContinue).CounterSamples
  $by = @{}
  foreach ($x in $s) {
    if ($x.InstanceName -match 'pid_(\d+)') {
      $id = [int]$Matches[1]
      if (-not $by.ContainsKey($id)) { $by[$id] = @{d = 0; sh = 0} }
      if ($x.Path -like '*dedicated usage') { $by[$id].d += $x.CookedValue } else { $by[$id].sh += $x.CookedValue }
    }
  }
  foreach ($id in $by.Keys) {
    $d = [math]::Round($by[$id].d / 1MB); $sh = [math]::Round($by[$id].sh / 1MB)
    if ($d -gt 50 -or $sh -gt 50) {
      $n = (Get-Process -Id $id -ErrorAction SilentlyContinue).ProcessName
      "$t,$id,$n,$d,$sh" | Out-File -Append -Encoding utf8 $Out
    }
  }
  Start-Sleep -Seconds $Seconds
}
