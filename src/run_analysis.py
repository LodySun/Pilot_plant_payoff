"""ROI discovery and ROI timeseries loading for the viewing analysis.

Used by src.run_pseudo_dataset_null / src.run_viewing_pseudo_null. The
original viewing-analysis entry point is not part of this repository.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_io  # noqa: E402
from src.common import stable_seed  # noqa: E402

SMOKE_VOXELS = 250


def resolve_rois(cfg: dict, smoke: bool) -> dict[str, dict]:
    if smoke:
        rois = {r["name"]: {"family": r["family"], "backend": "synthetic",
                            "description": "SYNTHETIC"} for r in cfg["rois"]}
        rois["pmc_chen"] = {"family": "pmn", "backend": "synthetic",
                            "description": "SYNTHETIC"}
        for info in rois.values():
            info["subjects"] = [f"{i:02d}" for i in range(1, 13)]
        return rois
    return data_io.available_rois(cfg)


def get_timeseries(cfg: dict, roi: str, subject: str, smoke: bool,
                   n_trs: int, runs: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Z-scored (within run) voxel x TR matrix for one subject and ROI."""
    if smoke:
        rng = np.random.default_rng(stable_seed(roi, subject, int(cfg["seed"])))
        data = rng.standard_normal((SMOKE_VOXELS, n_trs)).astype(np.float32)
    else:
        data = data_io.load_roi(cfg, roi, subject)
    return data_io.zscore_time(data, runs)
