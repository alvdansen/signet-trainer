# serve_gridwatch.ps1 — serve a gridwatch grid live + best-effort tunnel (operator standing rule).
# Usage: powershell -File scripts/serve_gridwatch.ps1 [-GridDir _grid_campaign_r1_native] [-Port 8000]
# Tunnel: uses ngrok if in PATH (`ngrok http <port>`), else cloudflared quick tunnel
# (`cloudflared tunnel --url http://127.0.0.1:<port>`), else localhost-only with an install hint.
# ngrok setup (when someone wants it): https://ngrok.com/download -> `ngrok config add-authtoken <token>`
# -> re-run this script; nothing else to change (scaffolded 2026-07-11, not prioritized).
param([string]$GridDir = "_grid_campaign_r1_native", [int]$Port = 8000)
$grid = Join-Path $PSScriptRoot "..\_tools\finetune-gridwatch\.venv\Scripts\grid.exe"
Start-Process -NoNewWindow -FilePath $grid -ArgumentList @("watch", $GridDir)
Write-Host "[serve_gridwatch] live at http://127.0.0.1:$Port"
$ngrok = Get-Command ngrok -ErrorAction SilentlyContinue
$cf = Get-Command cloudflared -ErrorAction SilentlyContinue
if ($ngrok) { Write-Host "[serve_gridwatch] tunneling via ngrok..."; & ngrok http $Port }
elseif ($cf) { Write-Host "[serve_gridwatch] tunneling via cloudflared..."; & cloudflared tunnel --url "http://127.0.0.1:$Port" }
else { Write-Host "[serve_gridwatch] no tunnel tool in PATH (ngrok/cloudflared) - localhost only. To enable: install ngrok + authtoken, re-run." }
