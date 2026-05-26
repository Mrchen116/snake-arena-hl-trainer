"""Dynamic loader for the current experiment's policy.py.

Reads SNAKE_POLICY_PATH and exec_module()s that file. Replaces the old
`snake_hl/policy.py` global slot so multiple experiments can train
concurrently without sharing a single file.

Trainer flow:
    1. train.py sets os.environ["SNAKE_POLICY_PATH"] = "<exp>/policy.py"
       before importing snake_hl.eval / failure_report / replay.
    2. eval.POLICIES["current"] delegates to policy_runtime.choose_action,
       which lazily loads the file on first call.
    3. After the optimizer edits the file, trainer calls reload() to swap
       in the new module; POLICIES["current"] sees the new function on the
       next call (no re-import needed).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

from snake_hl.env import SnakeState

_ENV_VAR = "SNAKE_POLICY_PATH"
_module: ModuleType | None = None


def _load() -> ModuleType:
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        raise RuntimeError(
            f"{_ENV_VAR} is not set. Trainer / eval / replay must run with "
            f"this env var pointing at the experiment's policy.py file."
        )
    path = Path(raw).resolve()
    if not path.is_file():
        raise RuntimeError(f"{_ENV_VAR}={path} does not exist or is not a file.")
    spec = importlib.util.spec_from_file_location("snake_hl._loaded_policy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot build import spec for {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "choose_action"):
        raise RuntimeError(f"{path} does not define choose_action(state).")
    return module


def reload() -> None:
    """Re-exec SNAKE_POLICY_PATH to pick up edits made by the optimizer."""
    global _module
    _module = _load()


def choose_action(state: SnakeState) -> str:
    if _module is None:
        reload()
    return _module.choose_action(state)  # type: ignore[union-attr]
