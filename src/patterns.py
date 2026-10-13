"""Pattern extraction and similarity.

A "pattern" is the voxel vector for one moment: the z-scored BOLD averaged
over that moment's HRF-shifted TR window. Similarity between two moments is
the Pearson correlation of their patterns across voxels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import fixed_window_trs, seconds_to_tr, window_trs  # noqa: E402


def moment_window(start_s: float, end_s: float, cfg: dict,
                  fixed_trs: int | None = None) -> tuple[int, int]:
    """TR window for a moment, honouring the config clamps.

    `fixed_trs` forces a length, for the Stage-4 window-length sweep.
    """
    tr = float(cfg["acquisition"]["tr_seconds"])
    shift = int(cfg["acquisition"]["hrf_shift_trs"])
    if fixed_trs is not None:
        return fixed_window_trs(start_s, tr, shift, fixed_trs)
    w = cfg["windows"]
    return window_trs(start_s, end_s, tr, shift, w["min_trs"], w["max_trs"])


def extract(data_z: np.ndarray, lo: int, hi: int) -> np.ndarray | None:
    """Mean pattern over TRs [lo, hi). None if the window leaves the series."""
    n_trs = data_z.shape[1]
    if lo < 0 or hi > n_trs or hi <= lo:
        return None
    return data_z[:, lo:hi].mean(axis=1)


def correlate(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Pearson r across voxels; NaN if either pattern is missing or flat."""
    if a is None or b is None:
        return np.nan
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def percentile_within_null(real: float, null: np.ndarray) -> float:
    """Fraction of the null distribution the real value exceeds.

    Ties count half, so a real value identical to every null sits at 0.5
    rather than 0 or 1.
    """
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if not np.isfinite(real) or null.size == 0:
        return np.nan
    return float((np.sum(null < real) + 0.5 * np.sum(null == real)) / null.size)


class WindowMaker:
    """One definition of a moment's TR window, applied to real and control alike.

    Two modes, selected by `windows.mode` in config.yaml:

      fixed    ceil(duration / TR), clamped to [min_trs, max_trs].
      segment  the enclosing fine segment's own TR span, clamped to
               [segment_min_trs, segment_max_trs]. Segment boundaries mark
               where the stimulus actually changes, so the window follows the
               event instead of a fixed stopwatch.

    Real and control moments must go through the same instance. If real
    moments used segment windows while controls used fixed ones, the two
    would differ in length and in how well they align to stimulus changes,
    and the null would no longer be comparable to the measurement.
    """

    def __init__(self, cfg: dict, segment_bounds: "np.ndarray | None" = None,
                 force_fixed_trs: int | None = None):
        w = cfg["windows"]
        self.tr = float(cfg["acquisition"]["tr_seconds"])
        self.shift = int(cfg["acquisition"]["hrf_shift_trs"])
        self.min_trs, self.max_trs = int(w["min_trs"]), int(w["max_trs"])
        self.seg_min = int(w.get("segment_min_trs", 2))
        self.seg_max = int(w.get("segment_max_trs", 6))
        self.force_fixed_trs = force_fixed_trs
        self.segment_bounds = segment_bounds
        requested = str(w.get("mode", "fixed")).lower()
        if force_fixed_trs is not None:
            self.mode = "forced"
        elif requested == "segment" and segment_bounds is not None:
            self.mode = "segment"
        else:
            self.mode = "fixed"

    def describe(self) -> str:
        if self.mode == "forced":
            return f"fixed {self.force_fixed_trs} TRs"
        if self.mode == "segment":
            return f"enclosing segment, clamped to [{self.seg_min}, {self.seg_max}] TRs"
        return f"ceil(duration/TR), clamped to [{self.min_trs}, {self.max_trs}] TRs"

    def __call__(self, start_s: float, end_s: float) -> tuple[int, int]:
        if self.mode == "forced":
            return fixed_window_trs(start_s, self.tr, self.shift, self.force_fixed_trs)
        if self.mode == "segment":
            onset = seconds_to_tr(start_s, self.tr, self.shift)
            row = self._segment_for(onset)
            if row is not None:
                lo, hi = int(row[0]) + self.shift, int(row[1]) + 1 + self.shift
                n = min(max(hi - lo, self.seg_min), self.seg_max)
                return lo, lo + n
            # No enclosing segment (e.g. the 2 segments with no TR): fall back.
        return window_trs(start_s, end_s, self.tr, self.shift,
                          self.min_trs, self.max_trs)

    def _segment_for(self, onset_tr: int):
        """[start_tr, end_tr] of the segment covering this (unshifted) TR."""
        import numpy as _np
        stim_tr = onset_tr - self.shift
        b = self.segment_bounds
        if b is None or b.size == 0:
            return None
        hit = _np.flatnonzero((b[:, 0] <= stim_tr) & (stim_tr <= b[:, 1]))
        return b[hit[0]] if hit.size else None


def segment_bounds_from_scenes(scenes) -> "np.ndarray":
    """(n, 2) array of [start_tr, end_tr] for segments that have TRs."""
    import numpy as _np
    ok = scenes["start_tr"].notna() & scenes["end_tr"].notna()
    return _np.column_stack([
        scenes.loc[ok, "start_tr"].to_numpy(dtype=int),
        scenes.loc[ok, "end_tr"].to_numpy(dtype=int),
    ])
