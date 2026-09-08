# Starts the outbound-caller worker. Leave this window open — it must keep
# running to answer call dispatches. Stop it with Ctrl+C.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Starting outbound-caller worker..." -ForegroundColor Cyan
python agent.py dev
