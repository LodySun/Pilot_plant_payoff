"""Onset-aligned plant-identity evidence during scanned free recall.

This exploratory follow-up keeps movie plant patterns fixed and moves a short,
full-width recall window around each released payoff-scene onset.  The reported
x axis removes the configured HRF shift, but the curve remains a slow BOLD
profile and cannot establish a fine-grained cognitive sequence.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_io  # noqa: E402
from src.build_controls import load_pairs  # noqa: E402
from src.common import (  # noqa: E402
    REPO_ROOT, ensure_dir, load_config, rel, require_lock_or_exit, resolve,
    stable_seed,
)
from src.patterns import correlate, percentile_within_null  # noqa: E402
from src.run_recall_bold import (  # noqa: E402
    load_chen_phase, load_event_release, one_based_blocks_pattern,
    order_valid_label_assignments, pair_scene_table, partial_correlate,
    subject_scene_blocks,
)
from src.stats_utils import fdr, fmt_p  # noqa: E402


METRICS = {
    "identity_delta_r": "candidate_delta_r",
    "partial_identity_delta_r": "partial_candidate_delta_r",
}


def timecourse_settings(cfg: dict, smoke: bool = False) -> dict:
    section = cfg["chen_recall_timecourse"]
    minimum = int(section["lag_min_trs"])
    maximum = int(section["lag_max_trs"])
    step = int(section["lag_step_trs"])
    if step <= 0 or maximum < minimum:
        raise ValueError("invalid recall timecourse lag range")
    lags = list(range(minimum, maximum + 1, step))
    if 0 not in lags:
        raise ValueError("recall timecourse must sample scene onset (lag 0)")
    key = "smoke_permutations" if smoke else "label_permutations"
    return {
        "lags": lags,
        "window_trs": int(section["window_trs"]),
        "hrf_shift_trs": int(section["hrf_shift_trs"]),
        "minimum_subject_pairs": int(section["minimum_subject_pairs"]),
        "permutations": int(section[key]),
    }


def relative_window_pattern(
    data: np.ndarray,
    blocks: list[tuple[int, int, int, int]],
    relative_tr: int,
    hrf_shift_trs: int,
    window_trs: int,
) -> tuple[np.ndarray | None, float]:
    """Return a full-width pattern and mean overlap with the labelled scene.

    Released blocks are one-based and inclusive. ``relative_tr`` is expressed
    on the cognitive/speech axis; the BOLD samples receive the HRF shift.
    Windows that would cross the scan edge are omitted rather than clipped.
    Repeated blocks for one scene are pooled at the same relative offset.
    """
    pieces, overlaps = [], []
    for start, end, offset, run_end in blocks:
        onset = int(offset) + int(start) - 1
        cognitive_lo = onset + int(relative_tr)
        bold_lo = cognitive_lo + int(hrf_shift_trs)
        bold_hi = bold_lo + int(window_trs)
        limit = min(int(run_end), data.shape[1])
        if bold_lo < int(offset) or bold_hi > limit:
            continue
        pieces.append(data[:, bold_lo:bold_hi])
        scene_lo = onset
        scene_hi = int(offset) + int(end)
        overlap = max(0, min(cognitive_lo + int(window_trs), scene_hi)
                      - max(cognitive_lo, scene_lo))
        overlaps.append(float(overlap / int(window_trs)))
    if not pieces:
        return None, np.nan
    return np.concatenate(pieces, axis=1).mean(axis=1), float(np.mean(overlaps))


def _candidate_scores(values: dict[int, float]) -> dict[int, float]:
    finite = {int(k): float(v) for k, v in values.items() if np.isfinite(v)}
    out = {}
    for candidate, value in finite.items():
        other = np.asarray([v for key, v in finite.items() if key != candidate])
        out[candidate] = percentile_within_null(value, other)
    return out


def _candidate_deltas(values: dict[int, float]) -> dict[int, float]:
    """Candidate similarity minus the mean of all other finite candidates."""
    finite = {int(k): float(v) for k, v in values.items() if np.isfinite(v)}
    out = {}
    for candidate, value in finite.items():
        other = np.asarray([v for key, v in finite.items() if key != candidate])
        out[candidate] = float(value - np.mean(other)) if other.size else np.nan
    return out


def score_timecourse_subject(
    subject: str,
    roi: str,
    mapping: pd.DataFrame,
    movie_patterns: dict[int, np.ndarray | None],
    recall: np.ndarray,
    recall_blocks: dict[int, list[tuple[int, int, int, int]]],
    lags: list[int],
    hrf_shift_trs: int,
    window_trs: int,
) -> tuple[list[dict], list[dict]]:
    """Score correct plants and every shuffled-label candidate at each lag."""
    plant_scenes = sorted(mapping["plant_scene"].astype(int).unique())
    cells, comparisons = [], []
    for row in mapping.itertuples(index=False):
        payoff_blocks = recall_blocks.get(int(row.payoff_scene), [])
        if not payoff_blocks:
            continue
        payoff_movie = movie_patterns.get(int(row.payoff_scene))
        for lag in lags:
            recall_pattern, overlap = relative_window_pattern(
                recall, payoff_blocks, lag, hrf_shift_trs, window_trs
            )
            if recall_pattern is None:
                continue
            raw = {
                scene: correlate(movie_patterns.get(scene), recall_pattern)
                for scene in plant_scenes
            }
            partial = {
                scene: partial_correlate(
                    movie_patterns.get(scene), recall_pattern, payoff_movie
                )
                for scene in plant_scenes
            }
            raw_pct = _candidate_scores(raw)
            partial_pct = _candidate_scores(partial)
            raw_delta = _candidate_deltas(raw)
            partial_delta = _candidate_deltas(partial)
            correct = int(row.plant_scene)
            cells.append({
                "subject": str(subject), "roi": str(roi),
                "pair_index": int(row.pair_index), "pair_id": str(row.pair_id),
                "plant_scene": correct, "payoff_scene": int(row.payoff_scene),
                "relative_tr": int(lag), "target_scene_overlap": overlap,
                "identity_percentile": raw_pct.get(correct, np.nan),
                "partial_identity_percentile": partial_pct.get(correct, np.nan),
                "identity_delta_r": raw_delta.get(correct, np.nan),
                "partial_identity_delta_r": partial_delta.get(correct, np.nan),
            })
            for candidate in plant_scenes:
                comparisons.append({
                    "subject": str(subject), "roi": str(roi),
                    "pair_index": int(row.pair_index),
                    "relative_tr": int(lag),
                    "candidate_plant_scene": int(candidate),
                    "candidate_percentile": raw_pct.get(candidate, np.nan),
                    "partial_candidate_percentile": partial_pct.get(
                        candidate, np.nan
                    ),
                    "candidate_delta_r": raw_delta.get(candidate, np.nan),
                    "partial_candidate_delta_r": partial_delta.get(
                        candidate, np.nan
                    ),
                })
    return cells, comparisons


def _assignment_curves(
    frame: pd.DataFrame,
    mapping: pd.DataFrame,
    assignments: list[tuple[int, ...]],
    value_column: str,
    minimum_pairs: int,
    chunk_size: int = 250,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Vectorised participant-first curves for observed and shuffled labels."""
    subjects = sorted(frame["subject"].astype(str).unique(), key=int)
    lags = np.sort(frame["relative_tr"].astype(int).unique())
    candidates = sorted(mapping["plant_scene"].astype(int).unique())
    subject_i = {value: index for index, value in enumerate(subjects)}
    lag_i = {value: index for index, value in enumerate(lags)}
    candidate_i = {value: index for index, value in enumerate(candidates)}
    n_pairs = len(mapping)
    values = np.full(
        (len(subjects), n_pairs, len(candidates), len(lags)), np.nan,
        dtype=np.float32,
    )
    for row in frame.itertuples(index=False):
        values[
            subject_i[str(row.subject)], int(row.pair_index),
            candidate_i[int(row.candidate_plant_scene)],
            lag_i[int(row.relative_tr)],
        ] = float(getattr(row, value_column))

    def curves(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        selected = np.full(
            (len(batch), len(subjects), n_pairs, len(lags)), np.nan,
            dtype=np.float32,
        )
        for pair_index in range(n_pairs):
            selected[:, :, pair_index, :] = np.transpose(
                values[:, pair_index, batch[:, pair_index], :], (1, 0, 2)
            )
        counts = np.sum(np.isfinite(selected), axis=2)
        sums = np.nansum(selected, axis=2)
        participant = np.divide(
            sums, counts, out=np.full_like(sums, np.nan),
            where=counts >= int(minimum_pairs),
        )
        group = np.nanmean(participant, axis=1)
        return group, participant

    def indices(labels: list[tuple[int, ...]] | tuple[tuple[int, ...], ...]) -> np.ndarray:
        return np.asarray([
            [candidate_i[int(scene)] for scene in assignment]
            for assignment in labels
        ], dtype=int)

    observed_labels = tuple(mapping["plant_scene"].astype(int))
    observed, participant = curves(indices([observed_labels]))
    null_parts = []
    assignment_indices = indices(assignments)
    for start in range(0, len(assignments), int(chunk_size)):
        null_parts.append(curves(assignment_indices[start:start + chunk_size])[0])
    return observed[0], np.vstack(null_parts), subjects, participant[0]


def timecourse_label_permutation(
    comparisons: pd.DataFrame,
    mapping: pd.DataFrame,
    assignments: list[tuple[int, ...]],
    minimum_pairs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lag curves with pointwise p and one-sided max-stat lag correction."""
    curve_rows, summary_rows, subject_rows = [], [], []
    for (roi, metric), value_column in [
        ((roi, metric), column)
        for roi in sorted(comparisons["roi"].unique())
        for metric, column in METRICS.items()
    ]:
        frame = comparisons[comparisons.roi == roi]
        observed, null, subjects, participant = _assignment_curves(
            frame, mapping, assignments, value_column, minimum_pairs
        )
        lags = np.sort(frame["relative_tr"].astype(int).unique())
        null_mean = np.nanmean(null, axis=0)
        null_low, null_high = np.nanquantile(null, [0.05, 0.95], axis=0)
        excess = observed - null_mean
        centered = null - null_mean[None, :]
        max_null = np.nanmax(centered, axis=1)
        global_offset = float(np.nanquantile(max_null, 0.95))
        p_point = (1 + np.sum(null >= observed[None, :], axis=0)) / (len(null) + 1)
        p_fwer = (1 + np.sum(max_null[:, None] >= excess[None, :], axis=0)) / (len(null) + 1)
        peak_i = int(np.nanargmax(excess))
        summary_rows.append({
            "roi": roi, "metric": metric,
            "peak_relative_tr": int(lags[peak_i]),
            "peak_observed": float(observed[peak_i]),
            "peak_null_mean": float(null_mean[peak_i]),
            "peak_excess": float(excess[peak_i]),
            "peak_p_fwer_lags": float(p_fwer[peak_i]),
            "n_permutations": int(len(null)),
        })
        for index, lag in enumerate(lags):
            curve_rows.append({
                "roi": roi, "metric": metric, "relative_tr": int(lag),
                "observed": float(observed[index]),
                "null_mean": float(null_mean[index]),
                "null_p05": float(null_low[index]),
                "null_p95": float(null_high[index]),
                "maxstat_upper_95": float(null_mean[index] + global_offset),
                "excess": float(excess[index]),
                "p_point_greater": float(p_point[index]),
                "p_fwer_lags": float(p_fwer[index]),
                "n_subjects": int(np.sum(np.isfinite(participant[:, index]))),
            })
        for subject_i, subject in enumerate(subjects):
            for lag_i, lag in enumerate(lags):
                subject_rows.append({
                    "subject": subject, "roi": roi, "metric": metric,
                    "relative_tr": int(lag),
                    "value": float(participant[subject_i, lag_i]),
                })
    summary = pd.DataFrame(summary_rows)
    reject, adjusted = fdr(summary["peak_p_fwer_lags"])
    summary["peak_q_four_curves"] = adjusted
    summary["peak_significant_q"] = reject
    return pd.DataFrame(curve_rows), summary, pd.DataFrame(subject_rows)


def write_figure(path: Path, curve: pd.DataFrame, cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    section = cfg["chen_recall_timecourse"]
    tr = float(cfg["acquisition"]["tr_seconds"])
    rois = [section["primary_roi"], section["control_roi"]]
    metrics = list(METRICS)
    labels = {
        "identity_delta_r": "Raw plant evidence",
        "partial_identity_delta_r": "Payoff-controlled plant evidence",
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
    for row_i, metric in enumerate(metrics):
        for col_i, roi in enumerate(rois):
            ax = axes[row_i, col_i]
            frame = curve[(curve.roi == roi) & (curve.metric == metric)].sort_values("relative_tr")
            x = frame.relative_tr.to_numpy() * tr
            ax.fill_between(x, frame.null_p05, frame.null_p95, color="0.8", alpha=.5,
                            label="pointwise 5–95% label null")
            ax.plot(x, frame.maxstat_upper_95, color="0.45", ls=":", lw=1,
                    label="one-sided 95% max-stat bound")
            ax.plot(x, frame.null_mean, color="0.45", ls="--", lw=1)
            ax.plot(x, frame.observed, color="#2457a6", lw=2, marker="o", ms=3)
            sig = frame.p_fwer_lags < .05
            ax.scatter(x[sig], frame.loc[sig, "observed"], marker="*", s=80,
                       color="#b2182b", zorder=5)
            ax.axvline(0, color="black", lw=1)
            ax.set_title(f"{roi}: {labels[metric]}")
            ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Seconds from labelled payoff-recall onset (HRF-corrected)")
    axes[1, 1].set_xlabel("Seconds from labelled payoff-recall onset (HRF-corrected)")
    axes[0, 0].set_ylabel("Correct minus mean-foil similarity")
    axes[1, 0].set_ylabel("Correct minus mean-foil similarity")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Exploratory recall-onset plant-evidence profile")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(path: Path, cfg: dict, meta: dict, summary: pd.DataFrame,
                  curve: pd.DataFrame, cells: pd.DataFrame) -> None:
    tr = float(cfg["acquisition"]["tr_seconds"])
    lines = [
        "# Time-resolved plant identity during scanned recall", "",
        "> **Exploratory relative BOLD profile, not a precise cognitive clock.**",
        "> The locked relations were read unchanged. No relation or keyword was",
        "> generated, edited, or inferred by this analysis.", "",
        "Each released payoff-recall scene onset was treated as time zero. At every",
        f"lag, a full {meta['window_trs']}-TR ({meta['window_trs'] * tr:.1f}-s) recall",
        f"window was sampled after the fixed +{meta['hrf_shift_trs']}-TR haemodynamic",
        "shift. The plotted x axis removes that shift, so zero denotes the labelled",
        "content onset rather than the BOLD sample time. Repeated recall blocks were",
        "pooled at the same relative lag; scan-edge windows were omitted, never clipped.",
        "", "Both raw plant identity and identity after linearly removing the matching",
        "movie-payoff pattern are shown. Correct plant labels were compared with",
        f"{meta['n_permutations']:,} order-valid reassignments. Peak p values use a",
        "one-sided max-statistic across every sampled lag within a curve; the four",
        "curve-level peak p values are then FDR-corrected.", "",
        "| ROI | metric | peak time (s) | observed | null mean | excess | lag-FWER p | four-curve q |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values("peak_p_fwer_lags").itertuples():
        lines.append(
            f"| `{row.roi}` | `{row.metric}` | {row.peak_relative_tr * tr:+.1f} | "
            f"{row.peak_observed:.3f} | {row.peak_null_mean:.3f} | {row.peak_excess:+.3f} | "
            f"{fmt_p(row.peak_p_fwer_lags)} | {fmt_p(row.peak_q_four_curves)} |"
        )
    primary = summary[
        (summary.roi == cfg["chen_recall_timecourse"]["primary_roi"])
        & (summary.metric == "partial_identity_delta_r")
    ].iloc[0]
    primary_curve = curve[
        (curve.roi == cfg["chen_recall_timecourse"]["primary_roi"])
        & (curve.metric == "partial_identity_delta_r")
    ].sort_values("relative_tr")
    significant = primary_curve.loc[
        primary_curve.p_fwer_lags < .05, "relative_tr"
    ].astype(int).tolist()
    significant_seconds = ", ".join(f"{lag * tr:+.1f}" for lag in significant)
    purely_pre = primary_curve[
        primary_curve.relative_tr + int(meta["window_trs"]) <= 0
    ]
    any_pre = bool((purely_pre.p_fwer_lags < .05).any())
    pre_sentence = (
        "At least one full-width window confined before onset was significant."
        if any_pre else
        "No full-width window confined entirely before onset was significant."
    )
    lines += ["", "## Primary interpretive curve", "",
              f"The payoff-controlled PMC curve peaked at **{primary.peak_relative_tr * tr:+.1f} s** ",
              f"(observed {primary.peak_observed:.3f}, label-null mean {primary.peak_null_mean:.3f}, ",
              f"lag-maxT p={fmt_p(primary.peak_p_fwer_lags)}, four-curve q={fmt_p(primary.peak_q_four_curves)}).",
              f"The lag-maxT-significant window starts were **{significant_seconds or 'none'} s**.",
              pre_sentence,
              "The -1.5-s window already spans -1.5 to +3.0 s because every point",
              "averages 4.5 s. The result therefore localises the strongest evidence",
              "to windows containing the labelled onset and the first seconds after it;",
              "it is not evidence that plant reinstatement preceded payoff narration.",
              "A significant point would localise evidence only at BOLD resolution; it",
              "would not by itself show that plant retrieval preceded speech or caused",
              "the payoff narration.", "", "## Coverage and interpretation limits", ""]
    overlap = cells.groupby("relative_tr")["target_scene_overlap"].mean()
    lines += [
        f"- The labelled-scene overlap of the cognitive window rises from {overlap.iloc[0]:.2f}",
        f"  at {overlap.index[0] * tr:+.1f} s to {overlap.loc[0]:.2f} at onset and",
        f"  is {overlap.iloc[-1]:.2f} at {overlap.index[-1] * tr:+.1f} s.",
        "- Adjacent points overlap heavily and must not be treated as independent tests.",
        "- Scene boundaries come from the released coarse recall annotation, not from",
        "  a new plant/payoff annotation. Time zero marks labelled content onset, not",
        "  necessarily the participant's internal retrieval onset.",
        "- Spoken production, auditory feedback, and neighbouring recalled scenes can",
        "  affect the curve. The early-auditory ROI is a control for, not a complete",
        "  removal of, those processes.",
        "- This analysis was specified after the scene-averaged recall result was known",
        "  and is therefore exploratory.", "",
        "Outputs: `curve.csv`, `curve_summary.csv`, `subject_curves.csv`,",
        "`pair_lag_cells.csv`, `window_coverage.csv`, `timecourse.png`, and `meta.json`.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run_smoke(cfg: dict) -> int:
    settings = timecourse_settings(cfg, smoke=True)
    rng = np.random.default_rng(728)
    mapping = pd.DataFrame({
        "pair_index": np.arange(4), "pair_id": [f"SMOKE_{i}" for i in range(4)],
        "plant_scene": np.arange(1, 5), "payoff_scene": np.arange(10, 14),
    })
    comparisons = []
    injected_lag = 2
    for subject in range(1, 11):
        movie = {scene: rng.standard_normal(120) for scene in range(1, 14)}
        recall = rng.standard_normal((120, 80)) * .1
        blocks = {}
        for plant, payoff in zip(mapping.plant_scene, mapping.payoff_scene):
            start = 21 + (int(payoff) - 10) * 10
            blocks[int(payoff)] = [(start, start + 7, 0, 80)]
            lo = start - 1 + settings["hrf_shift_trs"] + injected_lag
            recall[:, lo:lo + settings["window_trs"]] += 4 * movie[int(plant)][:, None]
        _, rows = score_timecourse_subject(
            str(subject), "pmc_chen", mapping, movie, recall, blocks,
            settings["lags"], settings["hrf_shift_trs"], settings["window_trs"],
        )
        comparisons.extend(rows)
    assignments = order_valid_label_assignments(
        mapping, settings["permutations"], 77
    )
    _, summary, _ = timecourse_label_permutation(
        pd.DataFrame(comparisons), mapping, assignments,
        settings["minimum_subject_pairs"],
    )
    peak = summary[summary.metric == "partial_identity_delta_r"].iloc[0]
    if int(peak.peak_relative_tr) != injected_lag or peak.peak_p_fwer_lags >= .05:
        raise AssertionError(
            "smoke recall timecourse did not recover injected lag: "
            f"peak={peak.peak_relative_tr}, p={peak.peak_p_fwer_lags}"
        )
    out = ensure_dir(resolve(cfg, "results_smoke") / "chen_recall_timecourse")
    summary.to_csv(out / "curve_summary.csv", index=False)
    (out / "summary.md").write_text(
        "# SMOKE TEST — SYNTHETIC DATA, NOT A BRAIN RESULT\n\n"
        f"Recovered injected lag {injected_lag} TR; p={peak.peak_p_fwer_lags:.6f}.\n"
    )
    print(f"Wrote {rel(out)} (synthetic timecourse check)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Recall-onset plant identity profile")
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--timing-corrected", action="store_true",
        help="use 0 additional TR for the already-shifted Chen MATs and a separate output",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.timing_corrected and not args.smoke_test:
        cfg = copy.deepcopy(cfg)
        section = cfg["chen_recall_timecourse"]
        section["hrf_shift_trs"] = int(
            cfg["chen_timing_corrected"]["additional_shift_trs"]
        )
        section["results_dir"] = str(Path(section["results_dir"]) / "timing_corrected")
    require_lock_or_exit(cfg, smoke=args.smoke_test)
    if args.smoke_test:
        return run_smoke(cfg)

    settings = timecourse_settings(cfg)
    section = cfg["chen_recall_timecourse"]
    pairs = load_pairs(cfg, False)
    scenes = pd.read_csv(resolve(cfg, "derived") / "scenes.csv")
    mapping = pair_scene_table(pairs, scenes)
    event_release = load_event_release(cfg)
    rois = [str(section["primary_roi"]), str(section["control_roi"])]
    subjects = data_io.common_subjects(cfg, rois)
    timeline = json.loads((resolve(cfg, "derived") / "movie_timeline.json").read_text())
    part1_trs = int(timeline["part1_trs"])
    minimum_scene_trs = int(cfg["chen_recall_bold"]["minimum_scene_trs"])
    from src.data_quality import invalid_mask, load as load_invalid
    invalid_table = load_invalid(cfg)

    all_cells, all_comparisons = [], []
    for roi in rois:
        for subject in subjects:
            movie = data_io.zscore_time(
                load_chen_phase(cfg, roi, subject, "movie"),
                [(0, part1_trs), (part1_trs, int(timeline["total_trs"]))],
            )
            recall = data_io.zscore_time(load_chen_phase(cfg, roi, subject, "recall"))
            bad_movie = invalid_mask(cfg, roi, subject, movie.shape[1], invalid_table)
            movie_blocks, recall_blocks = subject_scene_blocks(
                event_release[f"s{subject}"], part1_trs, recall.shape[1]
            )
            movie_patterns = {
                scene: one_based_blocks_pattern(
                    movie, blocks, settings["hrf_shift_trs"],
                    minimum_scene_trs, bad_movie,
                )
                for scene, blocks in movie_blocks.items()
            }
            cells, comparisons = score_timecourse_subject(
                subject, roi, mapping, movie_patterns, recall, recall_blocks,
                settings["lags"], settings["hrf_shift_trs"], settings["window_trs"],
            )
            all_cells.extend(cells)
            all_comparisons.extend(comparisons)

    cells = pd.DataFrame(all_cells)
    comparisons = pd.DataFrame(all_comparisons)
    assignments = order_valid_label_assignments(
        mapping, settings["permutations"],
        stable_seed("chen_recall_timecourse_labels", int(cfg["seed"])),
    )
    curve, summary, subject_curves = timecourse_label_permutation(
        comparisons, mapping, assignments, settings["minimum_subject_pairs"]
    )
    tr = float(cfg["acquisition"]["tr_seconds"])
    for table in (cells, comparisons, curve, subject_curves):
        table["relative_seconds"] = table["relative_tr"] * tr
    coverage = cells.groupby(["roi", "relative_tr"], as_index=False).agg(
        n_cells=("identity_percentile", "count"),
        n_subjects=("subject", "nunique"),
        mean_target_scene_overlap=("target_scene_overlap", "mean"),
    )
    coverage["relative_seconds"] = coverage["relative_tr"] * tr
    meta = {
        "analysis": "exploratory payoff-recall-onset plant identity timecourse",
        "subjects": subjects, "n_subjects": len(subjects), "rois": rois,
        "n_pairs": len(mapping), "lags_trs": settings["lags"],
        "window_trs": settings["window_trs"],
        "hrf_shift_trs": settings["hrf_shift_trs"],
        "n_permutations": len(assignments), "annotation_lock_unchanged": True,
        "timing_correction_audit": bool(args.timing_corrected),
    }
    out = ensure_dir(REPO_ROOT / section["results_dir"])
    cells.to_csv(out / "pair_lag_cells.csv", index=False)
    curve.to_csv(out / "curve.csv", index=False)
    summary.to_csv(out / "curve_summary.csv", index=False)
    subject_curves.to_csv(out / "subject_curves.csv", index=False)
    coverage.to_csv(out / "window_coverage.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_figure(out / "timecourse.png", curve, cfg)
    write_summary(out / "summary.md", cfg, meta, summary, curve, cells)
    print(f"Wrote {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
