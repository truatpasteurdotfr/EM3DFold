from __future__ import annotations

import builtins
from contextlib import contextmanager
from time import perf_counter

from em3dfit.log_utils import normalize_log_message as _base_normalize_log_message


def _normalize_log_message(message: str) -> str:
    return _base_normalize_log_message(message.replace(": ", " "))


def log_message(message: str, stage: str | None = None, module_name: str | None = None) -> None:
    normalized = _normalize_log_message(message)
    _ = stage
    module_prefix = f"[{module_name}] " if module_name else ""
    if normalized:
        builtins.print(f"{module_prefix}{normalized}")
    else:
        builtins.print(module_prefix.rstrip())


def normalize_device_spec(device: str) -> str:
    device_text = str(device).strip().lower()
    if device_text == "":
        return "auto"
    if device_text.isdigit():
        return f"cuda:{device_text}"
    return device_text


def is_cuda_device_spec(device: str) -> bool:
    device_text = normalize_device_spec(device)
    return device_text == "cuda" or device_text.startswith("cuda:")


@contextmanager
def stage_timer(label: str, stage: str | None = None, module_name: str | None = None):
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        log_message(f"{label} took {elapsed:.2f}s", stage=stage, module_name=module_name)
