# #211: log WDDM GPU memory and host paging counters every N seconds.
# CSV columns: wall, NVIDIA adapter dedicated / shared MiB, host available MiB,
# hard-fault pages read per s (Memory\Pages Input/sec), page file % used, then
# per-process GPU dedicated/shared MiB for processes with > 16 MiB on the adapter.
param([string]$Out, [int]$Every = 5, [string]$Luid = "0x000176eb")
$paths = '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage',
         '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage',
         '\Memory\Available MBytes','\Memory\Pages Input/sec','\Paging File(_Total)\% Usage'
"wall,adapter_ded_mib,adapter_sh_mib,avail_mib,pages_in_s,pagefile_pct,procs" | Out-File -Encoding ascii $Out
while ($true) {
  try {
    $all = (Get-Counter $paths -ErrorAction SilentlyContinue).CounterSamples
    $s = $all | Where-Object { $_.InstanceName -like "*$Luid*" }
    $ad = ($s | Where-Object { $_.Path -like '*adapter memory*dedicated*' } | Measure-Object CookedValue -Sum).Sum / 1MB
    $as = ($s | Where-Object { $_.Path -like '*adapter memory*shared*' } | Measure-Object CookedValue -Sum).Sum / 1MB
    $av = ($all | Where-Object { $_.Path -like '*available mbytes*' }).CookedValue
    $pi = ($all | Where-Object { $_.Path -like '*pages input/sec*' }).CookedValue
    $pf = ($all | Where-Object { $_.Path -like '*paging file*' }).CookedValue
    $pp = @{}
    foreach ($x in ($s | Where-Object { $_.Path -like '*process memory*' })) {
      if ($x.InstanceName -match 'pid_(\d+)_') { $p = $Matches[1] } else { continue }
      if (-not $pp.ContainsKey($p)) { $pp[$p] = @(0, 0) }
      if ($x.Path -like '*dedicated*') { $pp[$p][0] += $x.CookedValue / 1MB } else { $pp[$p][1] += $x.CookedValue / 1MB }
    }
    $procs = ($pp.GetEnumerator() | Where-Object { $_.Value[0] + $_.Value[1] -gt 16 } | Sort-Object { -$_.Value[0] } |
              ForEach-Object { "{0}:{1:F0}/{2:F0}" -f $_.Key, $_.Value[0], $_.Value[1] }) -join ' '
    "{0:F3},{1:F0},{2:F0},{3:F0},{4:F0},{5:F1},{6}" -f ([DateTimeOffset]::Now.ToUnixTimeMilliseconds() / 1000.0), $ad, $as, $av, $pi, $pf, $procs | Out-File -Encoding ascii -Append $Out
  } catch {}
  Start-Sleep -Seconds $Every
}
