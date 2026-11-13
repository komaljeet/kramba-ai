# Host-side spill signal, sampled every ~2 s until the stop file appears.
# Writes one JSON line per sample: nvidia-smi used/total, every GPU adapter's
# dedicated/shared usage, and every GPU process instance's dedicated/shared usage
# for the given pid (the training process).
param([string]$Out, [string]$Stop, [int]$TrainPid = 0)
while (-not (Test-Path $Stop)) {
    $t = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0
    $smi = (& nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits) -split ','
    $c = @('\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage')
    $p = $TrainPid
    if ($p -le 0) {   # the training process: the python with the largest working set
        $q = Get-Process python -ErrorAction SilentlyContinue | Sort-Object WorkingSet64 -Descending | Select-Object -First 1
        if ($q) { $p = $q.Id }
    }
    if ($p -gt 0) {
        $c += "\GPU Process Memory(pid_$($p)_*)\Dedicated Usage"
        $c += "\GPU Process Memory(pid_$($p)_*)\Shared Usage"
    }
    $s = (Get-Counter $c -ErrorAction SilentlyContinue).CounterSamples
    $d = [ordered]@{ wall = $t; pid = $p; smi_used_mib = [int]$smi[0]; smi_total_mib = [int]$smi[1] }
    foreach ($x in $s) {
        $k = ($x.Path -replace '^.*\\gpu (adapter|process) memory\(', '$1(') -replace ' usage$', ''
        $d[$k] = [int64]$x.CookedValue
    }
    ($d | ConvertTo-Json -Compress) | Add-Content -Path $Out
    Start-Sleep -Milliseconds 1500
}
