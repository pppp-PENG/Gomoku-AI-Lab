import os
import torch
from pathlib import Path
import uuid

SESSION_ROOT = Path(__file__).resolve().parent.parent / ".test-runs" / uuid.uuid4().hex
SESSION_ROOT.mkdir(parents=True)

# Importing the ASGI module must never create records in the user's data directory.
os.environ["GOMOKU_DATA"] = str(SESSION_ROOT / "global")
torch.set_num_threads(1)


def pytest_configure(config):
    # Avoid sharing Windows temp/cache directories across execution accounts.
    if config.option.basetemp is None:
        config.option.basetemp = str(SESSION_ROOT / "scratch")
