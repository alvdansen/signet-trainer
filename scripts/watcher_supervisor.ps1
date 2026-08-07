# watcher_supervisor.ps1 - OBS-01 (D-OBS-2): supervise the parallel-inference watcher so a healthy
# DETACHED training run can NEVER again be silently frozen by a dead local watcher (the 2026-07-12
# incident: the watcher process died - external kill, clean log end - and the grid froze while a
# healthy detached run kept training; it was misread as "inference hung").
#
# Started ONCE via Start-Process by the operator, then it LOOPS - so it must survive Claude-session
# death (the incident's root cause was a tool-call-bound process). It launches the watcher DETACHED,
# relaunches it on crash with bounded backoff up to a BOUNDED relaunch cap (fork-bomb / crash-loop
# guard - a metered dispatcher can never be respawned unboundedly), kills + relaunches on a heartbeat
# STALL (WATCHER-stall, logged DISTINCTLY from training/render stalls), and STOPS on a run-complete
# sentinel or the watcher's DISTINCT cap-stop exit code (session cap reached - do NOT relaunch).
# Every launch / relaunch / stop is appended to the supervisor log.
#
# CONFIG-FIRST (D-NOHARDCODE): the three thresholds - watcher_heartbeat_stall_minutes,
# watcher_relaunch_backoff_seconds, watcher_relaunch_cap - are read from the signet config, NEVER
# hardcoded here. The heartbeat / sentinel / log file paths are derived LOCAL conventions (they match
# the watcher's OUTPUT_DIR-derived paths), not behaviour thresholds.
#
# Start detached (survives THIS shell - the whole point):
#   Start-Process -WindowStyle Hidden -FilePath powershell -ArgumentList @(
#     "-NoProfile","-ExecutionPolicy","Bypass","-File","scripts/watcher_supervisor.ps1",
#     "-WatcherScript","scripts/watch_parallel_inference.py",
#     "-ConfigPath","configs/<your-render-config>.yaml",
#     "-HeartbeatFile","_watcher_heartbeat",
#     "-SentinelFile","_watcher_complete",
#     "-CapStopFile","_watcher_capstop")

param(
    [string]$WatcherScript          = "scripts/watch_parallel_inference.py",
    [string]$ConfigPath             = "configs/<your-render-config>.yaml",
    [string]$HeartbeatFile          = "_watcher_heartbeat",
    [string]$SentinelFile           = "_watcher_complete",
    # WR-03: the DISTINCT cap-stop sentinel the watcher writes on a session-cap stop (dual signal
    # alongside CAP_STOP_EXIT). The supervisor checks it BEFORE reading the child exit code so a
    # null/unreliable ExitCode can never relaunch the dispatcher straight back into the same cap.
    [string]$CapStopFile            = "_watcher_capstop_campaign_r5",
    [string]$LogFile                = "_watcher_supervisor.log",
    # Supervisor poll cadence (LOCAL implementation detail, not an OBS threshold): how often to check
    # process state + heartbeat age. Kept well below watcher_heartbeat_stall_minutes so a stall is
    # caught promptly. Surfaced as a param so it is config-visible, never an anonymous literal.
    [int]$MonitorIntervalSeconds    = 60
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot   # scripts/ -> repo root
Set-Location $repo
$env:PYTHONUTF8 = "1"

$logPath      = Join-Path $repo $LogFile
$sentinelPath = Join-Path $repo $SentinelFile
$heartbeatPath = Join-Path $repo $HeartbeatFile
$capStopPath  = Join-Path $repo $CapStopFile

function Write-Log([string]$msg) {
    $line = "[{0}][supervisor] {1}" -f (Get-Date -Format "s"), $msg
    Write-Host $line
    Add-Content -Path $logPath -Value $line
}

# --- CONFIG-FIRST threshold sourcing (D-NOHARDCODE). Read the three OBS-01 tunables from the signet
#     config via a small python reader - the config is the SINGLE source of truth; nothing below is a
#     hardcoded literal. ---
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { Write-Log "FATAL: python not on PATH - cannot source config thresholds."; exit 1 }

# One-line python reader (avoids here-string line-ending fragility in Windows PowerShell). The three
# OBS-01 field names appear verbatim so the config is provably the source - no hardcoded thresholds.
$reader = "import sys; sys.path.insert(0, 'src'); from signet_trainer.config.load import load_config; m = load_config(r'$ConfigPath').modal; print(m.watcher_heartbeat_stall_minutes); print(m.watcher_relaunch_backoff_seconds); print(m.watcher_relaunch_cap)"
$cfgLines = & python -c $reader
if ($LASTEXITCODE -ne 0 -or $cfgLines.Count -lt 3) {
    Write-Log "FATAL: could not read OBS-01 thresholds from config '$ConfigPath'."; exit 1
}
$HeartbeatStallMinutes  = [double]$cfgLines[0]   # watcher_heartbeat_stall_minutes
$RelaunchBackoffSeconds = [double]$cfgLines[1]   # watcher_relaunch_backoff_seconds
$RelaunchCap            = [int]$cfgLines[2]       # watcher_relaunch_cap
Write-Log ("config-first thresholds: watcher_heartbeat_stall_minutes={0} watcher_relaunch_backoff_seconds={1} watcher_relaunch_cap={2}" -f $HeartbeatStallMinutes, $RelaunchBackoffSeconds, $RelaunchCap)

# Exit-code PROTOCOL (contract mirrored from the watcher's CAP_STOP_EXIT - NOT a tunable threshold):
# the watcher exits with this DISTINCT code on a session-cap stop; the supervisor then STOPS without
# relaunching (the metered dispatcher must not be respawned into the same cap).
$CapStopExit = 3

# Fresh supervision: clear any STALE run-complete sentinel so a prior run's completion can't
# short-circuit this session. WR-03: also clear a stale cap-stop sentinel from a prior session so a
# leftover one can't stop this run before the watcher even launches.
if (Test-Path $sentinelPath) { Remove-Item $sentinelPath -Force }
if (Test-Path $capStopPath) { Remove-Item $capStopPath -Force }

# WR-02: clear any STALE heartbeat file too. Heartbeats persist across sessions in the repo root; a
# leftover one with an old LastWriteTime would make the FIRST monitor iteration compute ageMin >
# HeartbeatStallMinutes while the freshly-launched watcher is still doing its startup volume ls calls
# and has not yet touched the heartbeat, spuriously killing the healthy watcher as a WATCHER-stall and
# burning a relaunch. Clearing it mirrors the sentinel cleanup so the stall gate only ever fires on a
# heartbeat the CURRENT watcher actually wrote.
if (Test-Path $heartbeatPath) { Remove-Item $heartbeatPath -Force }

$procArgs = @($WatcherScript)
if ($ConfigPath) { $procArgs += $ConfigPath }   # generalized watcher reads argv[1]; campaign fork ignores it

$relaunches = 0
while ($true) {
    if (Test-Path $sentinelPath) { Write-Log "run-complete sentinel present - stopping (no relaunch)."; break }
    if ($relaunches -ge $RelaunchCap) {
        Write-Log "relaunch cap reached ($relaunches/$RelaunchCap) - giving up (fork-bomb / crash-loop guard)."; break
    }

    Write-Log ("launching watcher DETACHED (relaunch {0}/{1}): python {2}" -f $relaunches, $RelaunchCap, ($procArgs -join ' '))
    # Detached launch - copy serve_gridwatch.ps1's Start-Process idiom; -PassThru captures the proc so
    # we can monitor exit + kill it on a heartbeat stall.
    $proc = Start-Process -PassThru -NoNewWindow -FilePath "python" -ArgumentList $procArgs

    $stallKilled = $false
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds $MonitorIntervalSeconds
        if (Test-Path $sentinelPath) { break }   # run-complete: let the outer loop stop
        # Heartbeat-stall gate (D-OBS-3): a frozen loop stops touching the heartbeat file; if its age
        # exceeds watcher_heartbeat_stall_minutes while the process is still ALIVE, the loop is wedged
        # - kill + relaunch, logged as WATCHER-stall (DISTINCT from training/render stalls).
        if (Test-Path $heartbeatPath) {
            $ageMin = ((Get-Date) - (Get-Item $heartbeatPath).LastWriteTime).TotalMinutes
            if ($ageMin -gt $HeartbeatStallMinutes) {
                Write-Log ("[WATCHER-stall] heartbeat age {0:N1}m > gate {1}m while watcher alive - the poll loop is wedged; killing + relaunching (distinct from training/render stalls)." -f $ageMin, $HeartbeatStallMinutes)
                try { Stop-Process -Id $proc.Id -Force } catch {}
                $proc.WaitForExit()
                $stallKilled = $true
                break
            }
        }
    }

    if (Test-Path $sentinelPath) { Write-Log "run-complete sentinel present - stopping (no relaunch)."; break }

    if ($stallKilled) {
        Write-Log ("backing off {0}s after WATCHER-stall kill, then relaunching." -f $RelaunchBackoffSeconds)
        Start-Sleep -Seconds $RelaunchBackoffSeconds
        $relaunches++
        continue
    }

    # Process exited on its own - branch on the exit-code contract.
    # WR-03 dual-signal backstop: check the cap-stop sentinel BEFORE trusting the exit code. Run-complete
    # is protected by two signals (sentinel + exit 0); cap-stop now gets the same treatment. Start-Process
    # -PassThru -NoNewWindow can surface a null/unreliable ExitCode; if the cap-stop code were missed the
    # watcher would relaunch straight back into the same cap and burn the relaunch budget. The sentinel
    # the watcher writes on its cap-stop makes that impossible.
    if (Test-Path $capStopPath) {
        Write-Log "cap-stop sentinel present (session cap reached) - NOT relaunching (dual-signal backstop)."; break
    }
    # Robust exit-code capture: ensure the process is fully reaped so ExitCode is populated (belt-and-braces
    # against a null ExitCode). A genuinely null code falls through to the crash-relaunch path below.
    $proc.WaitForExit()
    $code = $proc.ExitCode
    if ($code -eq 0) { Write-Log "watcher exited 0 (run complete) - stopping (no relaunch)."; break }
    if ($code -eq $CapStopExit) {
        Write-Log "watcher exited $CapStopExit (session cap reached) - NOT relaunching (cap-stop)."; break
    }
    Write-Log ("watcher crashed (exit {0}) - backing off {1}s then relaunching." -f $code, $RelaunchBackoffSeconds)
    Start-Sleep -Seconds $RelaunchBackoffSeconds
    $relaunches++
}

Write-Log "supervisor exiting."
