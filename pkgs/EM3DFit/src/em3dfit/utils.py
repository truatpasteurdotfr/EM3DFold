from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


def _normalize_log_message(message: str) -> str:
    return message.replace(": ", " ")


def log_message(message: str, stage: str | None = None) -> None:
    normalized = _normalize_log_message(message)
    tag = "Core" if stage is None else stage
    print(f"[{tag}] {normalized}")


@contextmanager
def stage_timer(label: str, stage: str | None = None):
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        log_message(f"{label} took {elapsed:.2f}s", stage=stage)
