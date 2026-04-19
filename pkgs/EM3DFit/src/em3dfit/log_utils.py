from __future__ import annotations

import builtins
import inspect
import sys
import threading
from typing import Any, Iterable


_STATE_KEY = "_em3d_package_print_state"
def normalize_log_message(message: str) -> str:
    if message.startswith("# "):
        return message[2:]
    if message.startswith("#"):
        return message[1:].lstrip()
    return message


def format_log_message(module_name: str, message: str) -> str:
    normalized = normalize_log_message(message)
    prefix = f"[{module_name}]"
    if normalized:
        return f"{prefix} {normalized}"
    return prefix


def _module_is_target(module_name: str, state: dict[str, Any]) -> bool:
    return (
        any(module_name.startswith(prefix) for prefix in state["target_prefixes"])
        and not any(module_name.startswith(prefix) for prefix in state["excluded_prefixes"])
    )


def _resolve_caller_module(state: dict[str, Any]) -> str | None:
    frame = inspect.currentframe()
    if frame is None:
        return None

    frame = frame.f_back
    matched_modules: list[str] = []
    while frame is not None:
        module_name = frame.f_globals.get("__name__")
        if isinstance(module_name, str) and _module_is_target(module_name, state):
            matched_modules.append(module_name)
        frame = frame.f_back

    for module_name in matched_modules:
        if module_name not in state["helper_modules"]:
            return module_name
    return None


def _wrapped_print(*args: Any, **kwargs: Any) -> None:
    state = _get_state()
    module_name = _resolve_caller_module(state)
    if module_name is None:
        state["original_print"](*args, **kwargs)
        return

    sep = kwargs.get("sep", " ")
    message = sep.join(str(arg) for arg in args)
    state["original_print"](format_log_message(module_name, message), **kwargs)


def _get_state() -> dict[str, Any]:
    state = getattr(builtins, _STATE_KEY, None)
    if state is None:
        original_print = getattr(builtins.print, "__em3d_original_print__", builtins.print)
        state = {
            "original_print": original_print,
            "target_prefixes": set(),
            "excluded_prefixes": set(),
            "helper_modules": set(),
            "lock": threading.RLock(),
        }
        _wrapped_print.__em3d_original_print__ = original_print
        state["wrapper"] = _wrapped_print
        setattr(builtins, _STATE_KEY, state)
    return state


def flush_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except Exception:
            pass


def install_exception_flush() -> None:
    original_excepthook = getattr(sys.excepthook, "__em3d_original_excepthook__", sys.excepthook)

    def _wrapped_excepthook(exc_type, exc_value, exc_traceback) -> None:
        flush_stdio()
        try:
            original_excepthook(exc_type, exc_value, exc_traceback)
        finally:
            flush_stdio()

    _wrapped_excepthook.__em3d_original_excepthook__ = original_excepthook
    if sys.excepthook is not _wrapped_excepthook:
        sys.excepthook = _wrapped_excepthook

    thread_hook = getattr(threading, "excepthook", None)
    if thread_hook is None:
        return

    original_thread_hook = getattr(thread_hook, "__em3d_original_threading_excepthook__", thread_hook)

    def _wrapped_thread_hook(args) -> None:
        flush_stdio()
        try:
            original_thread_hook(args)
        finally:
            flush_stdio()

    _wrapped_thread_hook.__em3d_original_threading_excepthook__ = original_thread_hook
    if threading.excepthook is not _wrapped_thread_hook:
        threading.excepthook = _wrapped_thread_hook


def install_package_print(
    target_prefixes: Iterable[str],
    *,
    excluded_prefixes: Iterable[str] = (),
    helper_modules: Iterable[str] = (),
) -> None:
    state = _get_state()
    with state["lock"]:
        state["target_prefixes"].update(target_prefixes)
        state["excluded_prefixes"].update(excluded_prefixes)
        state["helper_modules"].update(helper_modules)
        if builtins.print is not state["wrapper"]:
            builtins.print = state["wrapper"]


def make_module_print(module_name: str):
    state = _get_state()

    def _print(*args: Any, **kwargs: Any) -> None:
        sep = kwargs.get("sep", " ")
        message = sep.join(str(arg) for arg in args)
        state["original_print"](format_log_message(module_name, message), **kwargs)

    return _print
