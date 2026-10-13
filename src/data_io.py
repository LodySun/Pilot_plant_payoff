"""Unified access to ROI voxel timeseries, whichever release they came from.

Two backends exist, and the analysis stages are deliberately agnostic
between them:

  chen_mat        The Chen et al. release redistributed with the BrainIAK
                  tutorial: two pre-extracted ROIs (posterior medial cortex,
                  early auditory) for 17 subjects, already z-scored per voxel.
                  Kept under their own names (`pmc_chen`, `aud_early_chen`)
                  rather than folded into the atlas ROI names, because they
                  are different definitions of nearby tissue.

  wholebrain_npy  Atlas ROIs extracted from the fmriprep derivatives by
                  src/stream_wholebrain.py. Names come from config.yaml.

Every loader returns float32 (n_voxels, n_trs) with time along axis 1.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_movie_timeline, resolve  # noqa: E402

CHEN_ROOT_REL = "Sherlock_processed/data_mat"
CHEN_ROIS = {
    "pmc_chen": ("pmc_nn", "posterior medial cortex (Chen et al. release)", "pmn"),
    "aud_early_chen": ("aud_early", "early auditory cortex (Chen et al. release)",
                       "control_sensory"),
}


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def chen_root(cfg: dict) -> Path:
    return resolve(cfg, "raw") / CHEN_ROOT_REL


def chen_subjects(cfg: dict, roi: str) -> list[str]:
    stem = CHEN_ROIS[roi][0]
    root = chen_root(cfg)
    if not root.exists():
        return []
    found = []
    for p in root.glob(f"s*/sherlock_movie/{stem}_sherlock_movie_s*.mat"):
        m = re.search(r"_s(\d+)\.mat$", p.name)
        if m:
            found.append(m.group(1))
    return sorted(set(found), key=int)


def wholebrain_subjects(cfg: dict, roi: str) -> list[str]:
    d = resolve(cfg, "derived") / "wholebrain" / roi
    if not d.exists():
        return []
    return sorted({p.stem.replace("sub-", "") for p in d.glob("sub-*.npy")})


def available_rois(cfg: dict) -> dict[str, dict]:
    """Every ROI that has usable data on disk, with its subject list."""
    out: dict[str, dict] = {}
    for roi, (_stem, desc, family) in CHEN_ROIS.items():
        subs = chen_subjects(cfg, roi)
        if subs:
            out[roi] = {"backend": "chen_mat", "subjects": subs,
                        "family": family, "description": desc}
    for roi in cfg["rois"]:
        subs = wholebrain_subjects(cfg, roi["name"])
        if subs:
            out[roi["name"]] = {"backend": "wholebrain_npy", "subjects": subs,
                                "family": roi["family"],
                                "description": f"Harvard-Oxford: {', '.join(roi['labels'])}"}
    return out


def common_subjects(cfg: dict, rois: list[str]) -> list[str]:
    """Subjects present for every requested ROI."""
    avail = available_rois(cfg)
    sets = [set(avail[r]["subjects"]) for r in rois if r in avail]
    if not sets:
        return []
    return sorted(set.intersection(*sets), key=int)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_roi(cfg: dict, roi: str, subject: str) -> np.ndarray:
    """Return (n_voxels, n_trs) float32 for one subject and ROI."""
    avail = available_rois(cfg)
    if roi not in avail:
        raise KeyError(f"ROI {roi!r} has no data on disk. Available: {sorted(avail)}")
    backend = avail[roi]["backend"]

    if backend == "chen_mat":
        import scipy.io as sio
        stem = CHEN_ROIS[roi][0]
        path = (chen_root(cfg) / f"s{subject}" / "sherlock_movie"
                / f"{stem}_sherlock_movie_s{subject}.mat")
        mat = sio.loadmat(path)
        data = np.asarray(mat["rdata"], dtype=np.float32)
        if "keptvox" in mat:
            kept = np.asarray(mat["keptvox"]).ravel().astype(bool)
            if kept.size == data.shape[0] and not kept.all():
                data = data[kept, :]
        return data

    path = resolve(cfg, "derived") / "wholebrain" / roi / f"sub-{subject}.npy"
    return np.load(path).astype(np.float32, copy=False)


def _zscore_block(block: np.ndarray) -> np.ndarray:
    """Z-score each voxel over time within one contiguous block.

    Voxels with zero variance are returned as zeros rather than NaN, so a
    constant voxel contributes nothing to a correlation instead of poisoning it.
    """
    mu = block.mean(axis=1, keepdims=True)
    sd = block.std(axis=1, keepdims=True)
    out = np.zeros_like(block, dtype=np.float32)
    good = sd.ravel() > 0
    out[good] = ((block[good] - mu[good]) / sd[good]).astype(np.float32)
    return out


def zscore_time(data: np.ndarray, runs: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Z-score each voxel over time, WITHIN RUN.

    The two movie parts are separate scans. Standardising across the seam
    would let a between-run mean or scale difference -- scanner drift, a
    different baseline after the subject was repositioned -- leak into every
    pattern, and would do so differently for moments near the seam than for
    moments far from it. That is exactly the kind of artifact a plant/payoff
    contrast spanning the two runs would pick up.

    `runs` is a list of half-open [start, end) TR bounds. Passing None
    standardises over the whole series, which is correct only for
    single-run data.
    """
    if runs is None:
        return _zscore_block(data)
    out = np.zeros_like(data, dtype=np.float32)
    covered = 0
    for lo, hi in runs:
        lo, hi = max(int(lo), 0), min(int(hi), data.shape[1])
        if hi <= lo:
            continue
        out[:, lo:hi] = _zscore_block(data[:, lo:hi])
        covered += hi - lo
    if covered != data.shape[1]:
        raise ValueError(
            f"run bounds cover {covered} of {data.shape[1]} TRs; "
            "every timepoint must belong to exactly one run"
        )
    return out


def expected_n_trs(cfg: dict) -> int | None:
    tl = load_movie_timeline(cfg)
    return int(tl["total_trs"]) if tl else None
