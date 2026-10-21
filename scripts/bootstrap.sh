#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e '.[dev]'
(cd frontend && npm ci && npm run build)
echo 'Start: .venv/bin/python -m gomoku.cli serve'
