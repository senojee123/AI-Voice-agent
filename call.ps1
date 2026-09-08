# Places an outbound call through the running agent worker.
#
#   .\call.ps1                 # calls the default number (OUTBOUND_PHONE_NUMBER)
#   .\call.ps1 0740525967      # calls a specific number
#   .\call.ps1 +94740525967    # +E.164 is auto-converted to local format
#   .\call.ps1 0740525967 -TransferTo 0112345678
#
# Requires: start-agent.ps1 running in another window.
param(
    [string]$Number,
    [string]$TransferTo
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# load credentials from .env.local so the lk CLI can authenticate
if (-not (Test-Path ".env.local")) { throw ".env.local not found" }
Get-Content ".env.local" | ForEach-Object {
    if ($_ -match '^\s*([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$') {
        Set-Item -Path "Env:$($matches[1])" -Value $matches[2]
    }
}

# build the dispatch metadata
$meta = @{}
if ($Number)     { $meta.phone_number = $Number }
if ($TransferTo) { $meta.transfer_to  = $TransferTo }

$lkArgs = @("dispatch", "create", "--new-room", "--agent-name", "outbound-caller")
if ($meta.Count -gt 0) {
    $lkArgs += "--metadata"
    $lkArgs += ($meta | ConvertTo-Json -Compress)
}

$target = if ($Number) { $Number } else { $env:OUTBOUND_PHONE_NUMBER }
Write-Host "Dispatching call to $target ..." -ForegroundColor Cyan
& lk @lkArgs
