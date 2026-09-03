# Show progress of a finetune/train.py run by reading its log.jsonl.
#   powershell -File finetune\watch_train.ps1                    # one-shot, default run dir
#   powershell -File finetune\watch_train.ps1 -Follow            # refresh until finished
#   powershell -File finetune\watch_train.ps1 -RunDir <dir>      # another run
param(
    [string]$RunDir = "$PSScriptRoot\runs\pv_v1",
    [switch]$Follow,
    [int]$Every = 30
)

function Show-Progress($runDir) {
    $logPath = Join-Path $runDir 'log.jsonl'
    if (-not (Test-Path $logPath)) { Write-Host "no log yet at $logPath"; return $false }

    $maxSteps = 0
    $argsPath = Join-Path $runDir 'args.json'
    if (Test-Path $argsPath) { $maxSteps = (Get-Content $argsPath -Raw | ConvertFrom-Json).max_steps }

    $rows = Get-Content $logPath -Tail 40 | ForEach-Object { try { $_ | ConvertFrom-Json } catch {} }
    if (-not $rows) { Write-Host 'log is empty'; return $false }
    $last = $rows[-1]

    # rate from the most recent pair of log entries, which reflects current GPU contention
    $rate = $null
    if ($rows.Count -ge 2) {
        $prev = $rows[-2]
        $ds = $last.step - $prev.step
        $dt = $last.elapsed_s - $prev.elapsed_s
        if ($ds -gt 0 -and $dt -gt 0) { $rate = $dt / $ds }
    }

    $pct = if ($maxSteps) { 100 * $last.step / $maxSteps } else { 0 }
    $line = "step {0}/{1} ({2:N1}%)  loss {3:N4}  ema {4:N4}  vram {5} GiB" -f `
        $last.step, $maxSteps, $pct, $last.loss, $last.ema, $last.vram_gib
    if ($rate) {
        $eta = [TimeSpan]::FromSeconds(($maxSteps - $last.step) * $rate)
        $line += "  {0:N1} s/step  ETA {1:hh\:mm\:ss}" -f $rate, $eta
    }
    Write-Host $line
    if ($last.per_kind) {
        $kinds = $last.per_kind.PSObject.Properties | Sort-Object Name | ForEach-Object { "{0} {1:N4}" -f $_.Name, $_.Value }
        Write-Host ("  per-kind loss: " + ($kinds -join '   '))
    }

    $ckpt = Get-ChildItem $runDir -Filter 'lora_step*.pt' -ErrorAction SilentlyContinue | Sort-Object Name
    if ($ckpt) { Write-Host ("  checkpoints: " + (($ckpt | ForEach-Object { $_.Name }) -join ', ')) }

    $check = Get-ChildItem $runDir -Filter 'check_step*.json' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime
    foreach ($c in $check) {
        $s = (Get-Content $c.FullName -Raw | ConvertFrom-Json).summary
        $parts = $s.PSObject.Properties | ForEach-Object { "{0} {1:N4}" -f $_.Name, $_.Value.mean }
        Write-Host ("  holdout @ $($c.BaseName): " + ($parts -join '   '))
    }

    $alive = [bool](Get-CimInstance Win32_Process -Filter "name='python.exe'" |
        Where-Object { $_.CommandLine -like '*train.py*' })
    $gpu = (nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader) -join ''
    Write-Host ("  process: {0}   gpu: {1}" -f $(if ($alive) { 'running' } else { 'NOT running' }), $gpu)
    return ($maxSteps -gt 0 -and $last.step -ge $maxSteps)
}

do {
    if ($Follow) { Write-Host ("--- " + (Get-Date -Format 'HH:mm:ss') + " ---") }
    $done = Show-Progress $RunDir
    if ($Follow -and -not $done) { Start-Sleep -Seconds $Every }
} while ($Follow -and -not $done)
