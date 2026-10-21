param([switch]$CPU)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (!(Test-Path '.venv/Scripts/python.exe')) { python -m venv .venv }
if ($LASTEXITCODE) { throw 'Cannot create Python environment' }
if ($CPU) { & .venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu }
else { & .venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu126 }
if ($LASTEXITCODE) { throw 'PyTorch install failed' }
& .venv/Scripts/python.exe -m pip install -e '.[dev]'
if ($LASTEXITCODE) { throw 'Python dependency install failed' }
Push-Location frontend
try { npm ci; if ($LASTEXITCODE) { throw 'npm ci failed' }; npm run build; if ($LASTEXITCODE) { throw 'Frontend build failed' } }
finally { Pop-Location }
Write-Host 'Ready. Run: .\scripts\start.ps1'
