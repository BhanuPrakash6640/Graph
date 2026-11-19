"""Script to download precomputed model checkpoints and explanation artifacts.

Downloads released assets from GitHub Releases or validates local cached artifacts.
Note: The raw Elliptic dataset is not redistributed; download from Kaggle per README.md.
"""
from __future__ import annotations

import argparse
import logging
import os
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scripts.fetch_artifacts")

RELEASE_TAG = "v1.0.0"
BASE_URL = f"https://github.com/graphguard/graphguard/releases/download/{RELEASE_TAG}"

REQUIRED_ARTIFACTS = [
    "data/processed/nodes.parquet",
    "data/processed/edge_index.npy",
    "data/processed/splits.json",
    "checkpoints/xgb_paper_refit_all_s0.joblib",
    "checkpoints/20261117-150701_gcn_paper_refit_all_s0.pt",
]


def check_or_fetch(dest_dir: Path, dry_run: bool = False) -> None:
    logger.info("Checking required GraphGuard artifacts in %s...", dest_dir)
    missing = []
    for rel_path in REQUIRED_ARTIFACTS:
        full_path = dest_dir / rel_path
        if not full_path.exists():
            missing.append(rel_path)
        else:
            logger.info("Found: %s (size: %d bytes)", rel_path, full_path.stat().st_size)

    if not missing:
        logger.info("All essential artifacts are present locally.")
        return

    logger.warning("Missing %d artifacts: %s", len(missing), missing)
    if dry_run:
        logger.info("Dry run mode: skipped remote download.")
        return

    for item in missing:
        url = f"{BASE_URL}/{Path(item).name}"
        target = dest_dir / item
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Downloading %s from %s...", item, url)
            urllib.request.urlretrieve(url, str(target))
            logger.info("Downloaded %s successfully.", item)
        except Exception as e:
            logger.error("Could not download %s: %s. Build locally via Phase 2/3 scripts.", item, e)


def main():
    parser = argparse.ArgumentParser(description="Fetch GraphGuard release artifacts")
    parser.add_argument("--dest", default=".", help="Root repository directory")
    parser.add_argument("--dry-run", action="store_true", help="Check existence without downloading")
    args = parser.parse_args()

    check_or_fetch(Path(args.dest), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

