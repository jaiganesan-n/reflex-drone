"""Process-pool helpers shared by the planner, collection, DAgger, calibration and evaluation.

Always the ``spawn`` start method (the macOS default, and safe with torch). Workers run with one
math thread each so N workers use N cores.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _init_worker() -> None:
    for k in _THREAD_ENV:
        os.environ[k] = "1"
    if "torch" in sys.modules:          # imported by the spawn main-module fixup
        import torch
        torch.set_num_threads(1)


def make_executor(workers: int) -> ProcessPoolExecutor:
    # Children read these at import time (the parent's own libraries are already initialised).
    for k in _THREAD_ENV:
        os.environ.setdefault(k, "1")
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=_init_worker)


def parallel_map(fn, items, workers: int):
    """Yield ``(index, result)`` as tasks finish. ``workers <= 1`` runs inline, in order."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        for i, it in enumerate(items):
            yield i, fn(it)
        return
    with make_executor(min(workers, len(items))) as ex:
        pending = {ex.submit(fn, it): i for i, it in enumerate(items)}
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in done:
                yield pending.pop(f), f.result()
