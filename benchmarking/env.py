from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
_DOTENV_LOADED = False


def ensure_dotenv_loaded(dotenv_path: Path | None = None) -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    load_dotenv(dotenv_path=dotenv_path or ROOT / ".env")
    _DOTENV_LOADED = True


def get_env(name: str, default: str | None = None) -> str | None:
    ensure_dotenv_loaded()
    return os.environ.get(name, default)
