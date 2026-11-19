"""Run recording and experiment tracking.

Creates experiments/results/<experiment>/<YYYYmmdd-HHMMSS>_<model>_<split>_<featset>_s<seed>/
and writes config.yaml, env.json, metrics.json, timing.json, and predictions.parquet.
Never overwrites a run directory.
"""
from __future__ import annotations

import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import pandas as pd
import psutil
import yaml

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def get_env_info(seed: int | None = None) -> dict[str, Any]:
    """Capture environment metadata: git hash, package versions, python version, device."""
    git_hash = "nogit"
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        git_hash = res.stdout.strip()
    except Exception:
        git_hash = "nogit"

    packages = ["numpy", "scikit-learn", "xgboost", "psutil", "pandas", "pyarrow", "torch"]
    package_versions = {}
    for pkg in packages:
        try:
            package_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            package_versions[pkg] = "not_installed"

    device = "cpu"
    device_name = "cpu"
    if HAS_TORCH and torch.cuda.is_available():
        device = "cuda:0"
        device_name = torch.cuda.get_device_name(0)

    return {
        "git_hash": git_hash,
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "seed": seed,
        "device": device,
        "device_name": device_name,
    }


class RunRecorder:
    """Context manager for an individual experiment run."""

    def __init__(
        self,
        experiment: str,
        config: dict[str, Any],
        base_dir: str | Path = "experiments/results",
        timestamp: str | None = None,
    ):
        self.experiment = experiment
        self.config = dict(config)
        self.base_dir = Path(base_dir)

        model = str(self.config.get("model", "model"))
        split = str(self.config.get("split", "split"))
        featset = str(self.config.get("feature_set", self.config.get("featset", "all")))
        seed = self.config.get("seed", 0)

        ts = timestamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_name = f"{ts}_{model}_{split}_{featset}_s{seed}"

        exp_dir = self.base_dir / experiment
        self.run_dir = exp_dir / dir_name
        counter = 1
        while self.run_dir.exists():
            self.run_dir = exp_dir / f"{dir_name}_{counter:02d}"
            counter += 1

        self.metrics: dict[str, Any] = {}
        self.timing: dict[str, Any] = {}
        self.predictions: pd.DataFrame | None = None

        self._start_time: float | None = None
        self._process = psutil.Process()
        self._initial_rss = 0.0

    @property
    def output_dir(self) -> Path:
        return self.run_dir

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self.metrics.update(metrics)

    def log_timing(self, timing: dict[str, Any]) -> None:
        self.timing.update(timing)

    def log_predictions(self, df: pd.DataFrame) -> None:
        required_cols = {"node_idx", "txId", "time_step", "y", "score", "part"}
        assert required_cols.issubset(df.columns), (
            f"Predictions missing required columns: {required_cols - set(df.columns)}"
        )
        self.predictions = df

    def __enter__(self) -> RunRecorder:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._start_time = time.perf_counter()
        self._initial_rss = self._process.memory_info().rss / (1024 * 1024)

        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Write config.yaml
        with open(self.run_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, sort_keys=False)

        # Write env.json
        env_info = get_env_info(seed=self.config.get("seed"))
        with open(self.run_dir / "env.json", "w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        total_time_s = time.perf_counter() - (self._start_time or time.perf_counter())

        rss_mb = self._process.memory_info().rss / (1024 * 1024)
        cuda_peak_mb = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if (HAS_TORCH and torch.cuda.is_available())
            else 0.0
        )
        peak_memory_mb = float(rss_mb + cuda_peak_mb)

        if "total_time_s" not in self.timing:
            self.timing["total_time_s"] = total_time_s
        self.timing["rss_peak_mb"] = float(rss_mb)
        self.timing["cuda_peak_mb"] = float(cuda_peak_mb)
        self.timing["peak_memory_mb"] = peak_memory_mb

        # Write metrics.json
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)

        # Write timing.json
        with open(self.run_dir / "timing.json", "w", encoding="utf-8") as f:
            json.dump(self.timing, f, indent=2)

        # Write predictions.parquet
        if self.predictions is not None:
            self.predictions.to_parquet(self.run_dir / "predictions.parquet", index=False)

