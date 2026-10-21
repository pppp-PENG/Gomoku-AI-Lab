param([int]$Port = 8000, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (!(Test-Path '.venv/Scripts/python.exe')) { throw 'Run scripts/bootstrap.ps1 first' }
if (!(Test-Path 'frontend/dist/index.html')) {
    Push-Location frontend
    try { npm run build; if ($LASTEXITCODE) { throw 'Frontend build failed' } }
    finally { Pop-Location }
}
$browserTask = $null
if (!$NoBrowser) {
    $browserTask = Start-Job -ArgumentList "http://127.0.0.1:$Port" -ScriptBlock {
        param($address)
        for ($attempt = 0; $attempt -lt 30; $attempt++) {
            try {
                $response = Invoke-WebRequest -Uri "$address/api/health" -UseBasicParsing -TimeoutSec 1
                if ($response.StatusCode -eq 200) { Start-Process $address; break }
            } catch { Start-Sleep -Milliseconds 500 }
        }
    }
}
try { & .venv/Scripts/python.exe -m gomoku.cli serve --port $Port }
finally { if ($browserTask) { Stop-Job $browserTask; Remove-Job $browserTask } }
