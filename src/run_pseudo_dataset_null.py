"""Matched pseudo-narrative datasets for the viewing analysis.

This is a design-level re-expression of Family A, not an independent test.
Each draw chooses one matched control pair for every locked human pair and uses
that same stimulus-level choice for all participants. A separately labelled
fixed-window sensitivity requires exact endpoint lengths and exact discrete TR
gap while retaining all locked pairs.

The functions here are driven by src.run_viewing_pseudo_null, which fixes
source-aligned timing and a Family-A-only control pool.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import run_analysis as RA  # noqa: E402
from src.build_controls import (  # noqa: E402
    build_exclusion_mask, enumerate_candidates, window_is_valid,
)
from src.common import run_bounds, stable_seed  # noqa: E402
from src.patterns import (  # noqa: E402
    WindowMaker, correlate, extract, moment_window, percentile_within_null,
)


Window = tuple[int, int]
PairWindows = tuple[Window, Window]


def control_percentiles(correlations: np.ndarray) -> np.ndarray:
    """Rank every control inside its subject x pair control pool.

    The selected control remains in the pool. Average ranks therefore have an
    exact mean of .5 even with ties, matching ``percentile_within_null``.
    """
    x = np.asarray(correlations, dtype=float)
    out = np.full_like(x, np.nan)
    for s in range(x.shape[0]):
        ok = np.isfinite(x[s])
        n = int(ok.sum())
        if n:
            out[s, ok] = (rankdata(x[s, ok], method="average") - 0.5) / n
    return out


def aggregate_real(real_percentiles: np.ndarray) -> tuple[float, np.ndarray]:
    """Pairs within participant, then participants with equal weight."""
    x = np.asarray(real_percentiles, dtype=float)
    counts = np.isfinite(x).sum(axis=1)
    subject = np.divide(
        np.nansum(x, axis=1), counts,
        out=np.full(x.shape[0], np.nan), where=counts > 0,
    )
    return float(np.nanmean(subject)), subject


def draw_pseudo_datasets(
    percentile_by_pair: dict[str, np.ndarray], repetitions: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw complete pseudo stories with one shared control choice per pair."""
    if not percentile_by_pair:
        return np.full(repetitions, np.nan)
    n_subjects = next(iter(percentile_by_pair.values())).shape[0]
    sums = np.zeros((n_subjects, repetitions), dtype=float)
    counts = np.zeros((n_subjects, repetitions), dtype=np.int16)
    for matrix in percentile_by_pair.values():
        if matrix.shape[0] != n_subjects or matrix.shape[1] == 0:
            raise ValueError("incompatible or empty control-percentile matrix")
        selected = rng.integers(0, matrix.shape[1], size=repetitions)
        values = matrix[:, selected]
        ok = np.isfinite(values)
        sums += np.where(ok, values, 0.0)
        counts += ok
    subject_means = np.divide(
        sums, counts, out=np.full_like(sums, np.nan), where=counts > 0,
    )
    valid_subjects = np.isfinite(subject_means).sum(axis=0)
    return np.divide(
        np.nansum(subject_means, axis=0), valid_subjects,
        out=np.full(repetitions, np.nan), where=valid_subjects > 0,
    )


def pseudo_stats(observed: float, draws: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(draws, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size or not np.isfinite(observed):
        return {"observed": observed, "pseudo_mean": np.nan,
                "pseudo_sd": np.nan, "pseudo_p_greater": np.nan,
                "pseudo_p_two_sided": np.nan, "observed_percentile": np.nan,
                "n_draws": int(x.size)}
    center = float(x.mean())
    return {
        "observed": observed,
        "pseudo_mean": center,
        "pseudo_sd": float(x.std(ddof=1)),
        "pseudo_p_greater": float((1 + np.sum(x >= observed)) / (x.size + 1)),
        "pseudo_p_two_sided": float(
            (1 + np.sum(np.abs(x - center) >= abs(observed - center)))
            / (x.size + 1)
        ),
        "observed_percentile": float(
            (np.sum(x < observed) + 0.5 * np.sum(x == observed)) / x.size
        ),
        "n_draws": int(x.size),
    }


def _run_id(window: Window, runs: list[tuple[int, int]]) -> int | None:
    lo, hi = window
    for i, (a, b) in enumerate(runs):
        if lo >= a and hi <= b:
            return i
    return None


def _crosses(pair: PairWindows, runs: list[tuple[int, int]]) -> int:
    return int(_run_id(pair[0], runs) != _run_id(pair[1], runs))


def segment_exact_count(
    real: PairWindows, valid_windows: list[Window], runs: list[tuple[int, int]],
) -> int:
    """Number of segment-window controls matching lengths and exact TR gap."""
    len_a = real[0][1] - real[0][0]
    len_b = real[1][1] - real[1][0]
    gap = real[1][0] - real[0][1]
    span = _crosses(real, runs)
    b_lookup = {w for w in valid_windows if w[1] - w[0] == len_b}
    count = 0
    for a in valid_windows:
        if a[1] - a[0] != len_a:
            continue
        b = (a[1] + gap, a[1] + gap + len_b)
        if b in b_lookup and a[1] <= b[0] and _crosses((a, b), runs) == span:
            count += 1
    return count


def build_definitions(
    cfg: dict, pairs: pd.DataFrame, controls: pd.DataFrame,
    timeline: dict, window: WindowMaker, segment_bounds: np.ndarray,
) -> tuple[dict[str, dict], dict[str, dict], pd.DataFrame]:
    """Build primary and exact-gap window definitions without touching BOLD."""
    runs = run_bounds(timeline)
    excluded = build_exclusion_mask(pairs, cfg, timeline)

    primary: dict[str, dict] = {}
    for _, r in pairs.iterrows():
        pid = str(r["pair_id"])
        g = controls.loc[controls["pair_id"].astype(str) == pid]
        primary[pid] = {
            "real": (
                window(float(r["plant_start_s"]), float(r["plant_end_s"])),
                window(float(r["payoff_start_s"]), float(r["payoff_end_s"])),
            ),
            # Deliberately retain row multiplicity: this exactly mirrors the
            # current Analysis-1 Family-A-compatible pool.
            "controls": [
                (
                    window(float(c["a_start_s"]), float(c["a_end_s"])),
                    window(float(c["b_start_s"]), float(c["b_end_s"])),
                )
                for _, c in g.iterrows()
            ],
        }

    # Legal production segment windows, used only for the feasibility audit.
    shift = int(cfg["acquisition"]["hrf_shift_trs"])
    seg_min = int(cfg["windows"].get("segment_min_trs", 2))
    seg_max = int(cfg["windows"].get("segment_max_trs", 6))
    valid_segments: set[Window] = set()
    for start, end in np.asarray(segment_bounds, dtype=int):
        lo = int(start) + shift
        n = min(max(int(end - start + 1), seg_min), seg_max)
        candidate = (lo, lo + n)
        if window_is_valid(*candidate, runs, excluded):
            valid_segments.add(candidate)
    valid_segment_list = sorted(valid_segments)

    exact: dict[str, dict] = {}
    feasibility: list[dict] = []
    target = int(cfg["pseudo_dataset_null"]["exact_controls_per_pair"])
    for _, r in pairs.iterrows():
        pid = str(r["pair_id"])
        real_fixed = (
            moment_window(float(r["plant_start_s"]), float(r["plant_end_s"]), cfg),
            moment_window(float(r["payoff_start_s"]), float(r["payoff_end_s"]), cfg),
        )
        plant_dur = float(r["plant_end_s"] - r["plant_start_s"])
        payoff_dur = float(r["payoff_end_s"] - r["payoff_start_s"])
        gap_s = float(r["payoff_start_s"] - r["plant_end_s"])
        span = _crosses(real_fixed, runs)
        space = enumerate_candidates(
            plant_dur, payoff_dur, gap_s, cfg, timeline, excluded, runs,
            crosses_run=span,
        )
        exact_gap = int(real_fixed[1][0] - real_fixed[0][1])
        if len(space):
            space = space[(space[:, 2] - space[:, 1]) == exact_gap]
        n_available = int(len(space))
        if n_available < target:
            raise RuntimeError(
                f"{pid}: exact-gap sensitivity has {n_available} controls; "
                f"needs {target}. Change the frozen plan rather than silently "
                "dropping a pair."
            )
        rng = np.random.default_rng(stable_seed(cfg["seed"], "exact-gap", pid))
        take = np.sort(rng.choice(n_available, size=target, replace=False))
        chosen = space[take]
        exact[pid] = {
            "real": real_fixed,
            "controls": [
                ((int(a), int(b)), (int(c), int(d))) for a, b, c, d in chosen
            ],
        }
        real_segment = primary[pid]["real"]
        feasibility.append({
            "pair_id": pid,
            "segment_plant_trs": real_segment[0][1] - real_segment[0][0],
            "segment_payoff_trs": real_segment[1][1] - real_segment[1][0],
            "segment_gap_trs": real_segment[1][0] - real_segment[0][1],
            "segment_exact_candidates": segment_exact_count(
                real_segment, valid_segment_list, runs
            ),
            "fixed_plant_trs": real_fixed[0][1] - real_fixed[0][0],
            "fixed_payoff_trs": real_fixed[1][1] - real_fixed[1][0],
            "fixed_gap_trs": exact_gap,
            "fixed_exact_candidates": n_available,
            "fixed_exact_used": target,
            "crosses_run": span,
        })
    return primary, exact, pd.DataFrame(feasibility)


def score_roi(
    cfg: dict, roi: str, subjects: list[str], definitions: dict[str, dict],
    n_trs: int, runs: list[tuple[int, int]], invalid_table: dict, smoke: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pair_ids = list(definitions)
    real = np.full((len(subjects), len(pair_ids)), np.nan)
    controls = {
        pid: np.full((len(subjects), len(definitions[pid]["controls"])), np.nan)
        for pid in pair_ids
    }
    for s, subject in enumerate(subjects):
        ts = RA.get_timeseries(cfg, roi, subject, smoke, n_trs, runs)
        bad = None
        if invalid_table:
            from src.data_quality import invalid_mask
            bad = invalid_mask(cfg, roi, subject, ts.shape[1], invalid_table)
        cache: dict[Window, np.ndarray | None] = {}

        def pattern(w: Window) -> np.ndarray | None:
            if w not in cache:
                lo, hi = w
                if bad is not None and 0 <= lo < hi <= bad.size and bad[lo:hi].any():
                    cache[w] = None
                else:
                    cache[w] = extract(ts, lo, hi)
            return cache[w]

        for p, pid in enumerate(pair_ids):
            d = definitions[pid]
            real[s, p] = correlate(pattern(d["real"][0]), pattern(d["real"][1]))
            for c, pair in enumerate(d["controls"]):
                controls[pid][s, c] = correlate(pattern(pair[0]), pattern(pair[1]))
    return real, controls


def analyse_branch(
    cfg: dict, roi: str, subjects: list[str], definitions: dict[str, dict],
    n_trs: int, runs: list[tuple[int, int]], invalid_table: dict, smoke: bool,
    repetitions: int, seed_label: str,
) -> tuple[dict, np.ndarray, list[dict], np.ndarray]:
    real_r, control_r = score_roi(
        cfg, roi, subjects, definitions, n_trs, runs, invalid_table, smoke
    )
    pair_ids = list(definitions)
    real_pct = np.full_like(real_r, np.nan)
    control_pct: dict[str, np.ndarray] = {}
    for p, pid in enumerate(pair_ids):
        ctrl = control_r[pid]
        control_pct[pid] = control_percentiles(ctrl)
        for s in range(len(subjects)):
            real_pct[s, p] = percentile_within_null(real_r[s, p], ctrl[s])
    observed, subject_scores = aggregate_real(real_pct)
    draws = draw_pseudo_datasets(
        control_pct, repetitions,
        np.random.default_rng(stable_seed(cfg["seed"], "pseudo-null", roi, seed_label)),
    )
    stats = pseudo_stats(observed, draws)
    stats.update({"roi": roi, "n_subjects": int(np.isfinite(subject_scores).sum()),
                  "n_pairs": len(pair_ids)})
    cells = []
    for s, subject in enumerate(subjects):
        for p, pid in enumerate(pair_ids):
            cells.append({"roi": roi, "subject": subject, "pair_id": pid,
                          "r_real": real_r[s, p],
                          "real_percentile": real_pct[s, p],
                          "n_controls": int(np.isfinite(control_r[pid][s]).sum())})
    return stats, draws, cells, subject_scores


def make_plot(draws: np.ndarray, row: pd.Series, path: Path, exact: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = draws[np.isfinite(draws)]
    observed = float(row["observed"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(x, bins=38, color="#b8c9d8", edgecolor="white", linewidth=0.6)
    ax.axvline(float(row["pseudo_mean"]), color="#546a7b", ls="--", lw=1.5,
               label=f"pseudo mean = {row['pseudo_mean']:.3f}")
    ax.axvline(observed, color="#b33a3a", lw=3,
               label=f"locked pairs = {observed:.3f}")
    title = ("Exact-gap fixed-duration sensitivity" if exact else
             "Matched pseudo-narrative datasets")
    subtitle = ("Exact endpoint lengths + exact discrete TR gap" if exact else
                "Visualisation of the existing Family A control structure")
    ax.set_title(f"{title}\n{subtitle}", loc="left", fontweight="bold")
    ax.set_xlabel("Mean matched-null percentile across participants")
    ax.set_ylabel("Pseudo datasets")
    ax.legend(frameon=False)
    ax.text(0.99, 0.96,
            f"PMC · {int(row['n_pairs'])} pairs · {int(row['n_subjects'])} participants\n"
            f"descriptive upper-tail p = {row['pseudo_p_greater']:.4f}",
            ha="right", va="top", transform=ax.transAxes, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
