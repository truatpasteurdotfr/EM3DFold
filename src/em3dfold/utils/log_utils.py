from __future__ import annotations

import builtins
import inspect
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Iterable


PRINT_STATE_KEY = "_em3d_package_print_state"
LOGGING_STATE_KEY = "_em3d_logging_runtime_state"
PROGRESS_LEVEL = 25
logging.addLevelName(PROGRESS_LEVEL, "PROGRESS")


def _progress(self, message, *args, **kwargs):
    if self.isEnabledFor(PROGRESS_LEVEL):
        self._log(PROGRESS_LEVEL, message, args, **kwargs)


logging.Logger.progress = _progress


class _AllowPackageLogsFilter(logging.Filter):
    def __init__(self, package_prefixes: Iterable[str], progress_logger_name: str):
        super().__init__()
        self.package_prefixes = tuple(package_prefixes)
        self.progress_logger_name = progress_logger_name

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name == self.progress_logger_name:
            return True
        return any(record.name.startswith(prefix) for prefix in self.package_prefixes)


class _ExactLevelFilter(logging.Filter):
    def __init__(self, level: int):
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.level


def normalize_log_message(message: str) -> str:
    message = message.lstrip("\r")
    if message.startswith("# "):
        return message[2:]
    if message.startswith("#"):
        return message[1:].lstrip()
    return message


def _get_print_state() -> dict[str, Any]:
    state = getattr(builtins, PRINT_STATE_KEY, None)
    if state is None:
        original_print = getattr(builtins.print, "__em3d_original_print__", builtins.print)
        state = {
            "original_print": original_print,
            "target_prefixes": set(),
            "excluded_prefixes": set(),
            "helper_modules": set(),
            "lock": threading.RLock(),
            "configured": False,
        }
        _wrapped_print.__em3d_original_print__ = original_print
        state["wrapper"] = _wrapped_print
        setattr(builtins, PRINT_STATE_KEY, state)
    return state


def _get_logging_state() -> dict[str, Any]:
    state = getattr(builtins, LOGGING_STATE_KEY, None)
    if state is None:
        state = {
            "file_handler": None,
            "stdout_handler": None,
            "log_path": None,
            "progress_logger_name": "em3d.progress",
            "package_prefixes": tuple(),
            "lock": threading.RLock(),
        }
        setattr(builtins, LOGGING_STATE_KEY, state)
    return state


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
    state = _get_print_state()
    if not state["configured"]:
        state["original_print"](*args, **kwargs)
        return

    file_obj = kwargs.get("file")
    if file_obj not in (None, sys.stdout, sys.stderr):
        state["original_print"](*args, **kwargs)
        return

    module_name = _resolve_caller_module(state)
    if module_name is None:
        state["original_print"](*args, **kwargs)
        return

    sep = kwargs.get("sep", " ")
    message = sep.join(str(arg) for arg in args)
    logging.getLogger(module_name).info(normalize_log_message(message))
    if kwargs.get("flush"):
        flush_stdio()


def install_package_print(
    target_prefixes: Iterable[str],
    *,
    excluded_prefixes: Iterable[str] = (),
    helper_modules: Iterable[str] = (),
) -> None:
    state = _get_print_state()
    with state["lock"]:
        state["target_prefixes"].update(target_prefixes)
        state["excluded_prefixes"].update(excluded_prefixes)
        state["helper_modules"].update(helper_modules)
        state["configured"] = True
        if builtins.print is not state["wrapper"]:
            builtins.print = state["wrapper"]


def make_module_print(module_name: str):
    def _print(*args: Any, **kwargs: Any) -> None:
        sep = kwargs.get("sep", " ")
        message = sep.join(str(arg) for arg in args)
        logging.getLogger(module_name).info(normalize_log_message(message))
        if kwargs.get("flush"):
            flush_stdio()

    return _print


def flush_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None:
                stream.flush()
        except Exception:
            pass


def install_exception_flush(package_name: str = "em3dfold") -> None:
    if getattr(sys.excepthook, "__em3d_managed__", False):
        return
    original_excepthook = getattr(sys.excepthook, "__em3d_original_excepthook__", sys.excepthook)

    def _wrapped_excepthook(exc_type, exc_value, exc_traceback) -> None:
        flush_stdio()
        try:
            logging.getLogger(f"{package_name}.unhandled").exception(
                "Unhandled exception",
                exc_info=(exc_type, exc_value, exc_traceback),
            )
            original_excepthook(exc_type, exc_value, exc_traceback)
        finally:
            flush_stdio()

    _wrapped_excepthook.__em3d_original_excepthook__ = original_excepthook
    _wrapped_excepthook.__em3d_managed__ = True
    if sys.excepthook is not _wrapped_excepthook:
        sys.excepthook = _wrapped_excepthook

    thread_hook = getattr(threading, "excepthook", None)
    if thread_hook is None:
        return
    original_thread_hook = getattr(thread_hook, "__em3d_original_threading_excepthook__", thread_hook)
    if getattr(thread_hook, "__em3d_managed__", False):
        return

    def _wrapped_thread_hook(args) -> None:
        flush_stdio()
        try:
            logging.getLogger(f"{package_name}.unhandled").exception(
                "Unhandled thread exception",
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            original_thread_hook(args)
        finally:
            flush_stdio()

    _wrapped_thread_hook.__em3d_original_threading_excepthook__ = original_thread_hook
    _wrapped_thread_hook.__em3d_managed__ = True
    if threading.excepthook is not _wrapped_thread_hook:
        threading.excepthook = _wrapped_thread_hook


def configure_runtime_logging(
    output_dir: str | Path,
    *,
    package_prefixes: Iterable[str],
    progress_logger_name: str,
    excluded_prefixes: Iterable[str] = (),
    helper_modules: Iterable[str] = (),
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"

    state = _get_logging_state()
    root_logger = logging.getLogger()

    with state["lock"]:
        file_handler = state.get("file_handler")
        if file_handler is not None:
            root_logger.removeHandler(file_handler)
            file_handler.close()

        stdout_handler = state.get("stdout_handler")
        if stdout_handler is not None:
            root_logger.removeHandler(stdout_handler)
            stdout_handler.close()

        root_logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.addFilter(_AllowPackageLogsFilter(package_prefixes, progress_logger_name))
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s")
        )

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(PROGRESS_LEVEL)
        stdout_handler.addFilter(_ExactLevelFilter(PROGRESS_LEVEL))
        stdout_handler.setFormatter(logging.Formatter("%(message)s"))

        root_logger.addHandler(file_handler)
        root_logger.addHandler(stdout_handler)

        state["file_handler"] = file_handler
        state["stdout_handler"] = stdout_handler
        state["log_path"] = log_path
        state["progress_logger_name"] = progress_logger_name
        state["package_prefixes"] = tuple(package_prefixes)

    install_package_print(
        target_prefixes=package_prefixes,
        excluded_prefixes=excluded_prefixes,
        helper_modules=helper_modules,
    )
    return log_path


def progress(message: str, *, logger_name: str | None = None) -> None:
    state = _get_logging_state()
    logger = logging.getLogger(logger_name or state["progress_logger_name"])
    logger.progress(normalize_log_message(message))
    flush_stdio()
