from __future__ import annotations

from numba import get_num_threads, set_num_threads


NUMBA_MAX_THREADS = 4


def configure_numba_threads(max_threads: int = NUMBA_MAX_THREADS) -> None:
    set_num_threads(min(int(max_threads), get_num_threads()))
