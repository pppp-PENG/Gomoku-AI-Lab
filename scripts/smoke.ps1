$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
& .venv/Scripts/python.exe -m pytest -q
if ($LASTEXITCODE) { throw 'Tests failed' }
& .venv/Scripts/python.exe -m gomoku.cli train --config configs/smoke-cnn.json
if ($LASTEXITCODE) { throw 'CNN smoke failed' }
& .venv/Scripts/python.exe -m gomoku.cli train --config configs/smoke-vit.json
if ($LASTEXITCODE) { throw 'ViT smoke failed' }
