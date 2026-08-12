# Stop the training run cleanly once a checkpoint is safely on disk.
#
# Author: Rambarun Komaljeet
#
# Killing mid-interval loses everything since the last save: checkpoints land
# every 500 steps, so a kill at step 1,300 resumes from 1,000 and throws away
# ~45 minutes. This waits for the target checkpoint FILE to exist (not merely
# for the CSV to reach the step -- the row is written before the .pt is closed),
# then stops the process.
#
# Usage:
#   powershell -File scripts\stop_after_checkpoint.ps1 -TargetStep 1000

param(
    [int]$TargetStep = 1000,
    [string]$CheckpointDir = "checkpoints\code_157m",
    [string]$PidFile = "logs\code_157m.pid",
    [int]$PollSeconds = 30
)

$ckpt = Join-Path $CheckpointDir "edge_conversational_step$TargetStep.pt"
$csv  = Join-Path $CheckpointDir "training_log.csv"

Write-Host "[watch] waiting for $ckpt"
Write-Host "[watch] polling every $PollSeconds s; the run keeps training until it appears"

while ($true) {
    if (Test-Path $ckpt) {
        # The .pt is written then closed; wait for its size to stop changing so
        # we never kill mid-write and leave a truncated checkpoint.
        $a = (Get-Item $ckpt).Length
        Start-Sleep -Seconds 10
        $b = (Get-Item $ckpt).Length
        if ($a -eq $b -and $b -gt 0) {
            Write-Host "[watch] checkpoint complete ($([math]::Round($b/1MB)) MB) -- stopping run"
            break
        }
        Write-Host "[watch] checkpoint still being written ($([math]::Round($b/1MB)) MB)..."
        continue
    }
    if (Test-Path $csv) {
        $last = Get-Content $csv -Tail 1
        Write-Host "[watch] $(Get-Date -Format 'HH:mm:ss')  last row: $last"
    }
    Start-Sleep -Seconds $PollSeconds
}

# Stop every python process: the launcher spawns children (dataloader workers,
# the real interpreter), and leaving orphans behind holds VRAM and competes with
# whatever runs next -- which cost an hour of misdiagnosis earlier today.
Get-Process python -ErrorAction SilentlyContinue | ForEach-Object {
    try { Stop-Process -Id $_.Id -Force -ErrorAction Stop; Write-Host "[watch] stopped PID $($_.Id)" } catch {}
}
Start-Sleep -Seconds 4
$left = (Get-Process python -ErrorAction SilentlyContinue | Measure-Object).Count
Write-Host "[watch] python processes left: $left"
Write-Host "[watch] final rows:"
Get-Content $csv -Tail 3
Write-Host "[watch] SAFE TO RESTART. Re-run the same training command to resume from step $TargetStep."
