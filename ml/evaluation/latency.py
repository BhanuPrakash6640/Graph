"""Latency benchmarking helpers for inference timing."""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def measure_latency(
    predict_fn: Callable[[Any], Any],
    X: Any,
    repeats: int = 20,
    warmup: int = 3,
) -> dict[str, float | int]:
    """Measure p50 and p95 inference latency per batch and per node.

    Synchronizes CUDA if torch and CUDA are active to ensure exact timing.
    """
    def _sync():
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()

    batch_size = len(X) if hasattr(X, "__len__") else 1

    # Warmup runs
    for _ in range(warmup):
        _sync()
        _ = predict_fn(X)
        _sync()

    times_ms = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        _ = predict_fn(X)
        _sync()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_arr = np.array(times_ms, dtype=np.float64)
    batch_p50 = float(np.percentile(times_arr, 50))
    batch_p95 = float(np.percentile(times_arr, 95))

    node_p50 = float(batch_p50 / batch_size) if batch_size > 0 else 0.0
    node_p95 = float(batch_p95 / batch_size) if batch_size > 0 else 0.0

    return {
        "batch_p50_ms": batch_p50,
        "batch_p95_ms": batch_p95,
        "node_p50_ms": node_p50,
        "node_p95_ms": node_p95,
        "repeats": repeats,
        "batch_size": batch_size,
    }

