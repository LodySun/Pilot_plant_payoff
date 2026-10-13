"""Within-participant plant-context reinstatement in Chen spoken-recall BOLD.

The locked plant/payoff table supplies relations only. The released Chen
``sherlock_allsubs_events.mat`` file supplies the independent 50-scene movie
segmentation and each participant's spoken-recall scene intervals.

This module holds the scoring, permutation and loading functions used by
src.run_chen_timing_corrected and src.run_recall_timecourse. The original
stand-alone analysis (with an extra 3-TR shift, superseded by the timing
correction) is not part of this repository.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_io  # noqa: E402
from src.common import rel  # noqa: E402
from src.patterns import correlate, percentile_within_null  # noqa: E402
from src.recall_text import scene_of_time  # noqa: E402
from src.stats_utils import directional_wilcoxon, fdr  # noqa: E402


def load_chen_phase(cfg: dict, roi: str, subject: str, phase: str) -> np.ndarray:
    """Load a Chen movie or recall ROI matrix as voxels x TRs."""
    if phase not in {"movie", "recall"}:
        raise ValueError(f"unknown Chen phase: {phase}")
    stem = data_io.CHEN_ROIS[roi][0]
    task = f"sherlock_{phase}"
    path = (
        data_io.chen_root(cfg) / f"s{subject}" / task
        / f"{stem}_{task}_s{subject}.mat"
    )
    if not path.exists():
        raise FileNotFoundError(rel(path))
    from scipy.io import loadmat

    mat = loadmat(path)
    data = np.asarray(mat["rdata"], dtype=np.float32)
    if "keptvox" in mat:
        kept = np.asarray(mat["keptvox"]).ravel().astype(bool)
        if kept.size == data.shape[0] and not kept.all():
            data = data[kept]
    return data


def load_event_release(cfg: dict) -> dict:
    """Load the release's subject-specific recall labels as ordinary dicts."""
    from scipy.io import loadmat

    path = data_io.chen_root(cfg) / "sherlock_allsubs_events.mat"
    if not path.exists():
        raise FileNotFoundError(rel(path))
    return loadmat(path, simplify_cells=True)["ev"]


def one_based_blocks_pattern(
    data: np.ndarray,
    blocks: list[tuple[int, int, int, int]],
    shift: int,
    minimum_trs: int,
    invalid: np.ndarray | None = None,
) -> np.ndarray | None:
    """Average one-based inclusive blocks after a shift.

    Blocks are ``(start, end, offset, run_end)``. Start/end are local to the
    run, while offset places them in a concatenated array. Shifting is clipped
    at the same run boundary and repeated recall blocks are pooled.
    """
    pieces = []
    for start, end, offset, run_end in blocks:
        lo = max(int(offset), int(offset) + int(start) - 1 + int(shift))
        hi = min(int(run_end), int(offset) + int(end) + int(shift), data.shape[1])
        if hi > lo:
            if invalid is not None and invalid[lo:hi].any():
                return None
            pieces.append(data[:, lo:hi])
    if not pieces or sum(piece.shape[1] for piece in pieces) < int(minimum_trs):
        return None
    return np.concatenate(pieces, axis=1).mean(axis=1)


def subject_scene_blocks(
    subject_events: dict, movie_part1_trs: int, recall_n_trs: int
) -> tuple[dict[int, list[tuple[int, int, int, int]]],
           dict[int, list[tuple[int, int, int, int]]]]:
    """Return movie and recall blocks keyed by the release's 1..50 scene IDs."""
    scene_ids = np.asarray(subject_events["SL_event_num"], dtype=int).ravel()
    part1 = np.atleast_2d(np.asarray(subject_events["fullSL1"], dtype=int))
    part2 = np.atleast_2d(np.asarray(subject_events["fullSL2"], dtype=int))
    if scene_ids.size != len(part1) + len(part2):
        raise ValueError("movie scene labels do not match fullSL1/fullSL2")
    movie: dict[int, list[tuple[int, int, int, int]]] = {}
    for scene, bounds in zip(scene_ids[:len(part1)], part1):
        movie[int(scene)] = [(int(bounds[0]), int(bounds[1]), 0, movie_part1_trs)]
    movie_total = movie_part1_trs + int(part2[:, 1].max())
    for scene, bounds in zip(scene_ids[len(part1):], part2):
        movie[int(scene)] = [(
            int(bounds[0]), int(bounds[1]), movie_part1_trs, movie_total
        )]

    event = subject_events["events"]
    recall_bounds = np.atleast_2d(np.asarray(event["freerecall"], dtype=int))
    recall_labels = np.asarray(event["freerecallnum"], dtype=int).ravel()
    if len(recall_bounds) != len(recall_labels):
        raise ValueError("recall intervals and labels differ in length")
    recall: dict[int, list[tuple[int, int, int, int]]] = {}
    for scene, bounds in zip(recall_labels, recall_bounds):
        recall.setdefault(int(scene), []).append(
            (int(bounds[0]), int(bounds[1]), 0, int(recall_n_trs))
        )
    return movie, recall


def partial_correlate(a: np.ndarray | None, b: np.ndarray | None,
                      covariate: np.ndarray | None) -> float:
    """Voxel-pattern partial r(a,b | covariate), with intercept removed.

    A correlation is undefined when either residual has effectively zero
    norm. Checking this explicitly prevents floating-point crumbs from a
    vector regressed on itself from being treated as a real pattern.
    """
    if a is None or b is None or covariate is None:
        return np.nan
    c = np.asarray(covariate, dtype=float)
    c = c - c.mean()
    denom = float(np.dot(c, c))
    if denom <= 0:
        return np.nan

    def residual(vector: np.ndarray) -> np.ndarray | None:
        centered = np.asarray(vector, dtype=float) - float(np.mean(vector))
        result = centered - c * (float(np.dot(centered, c)) / denom)
        scale = max(float(np.linalg.norm(centered)), 1.0)
        if (not np.isfinite(result).all()
                or float(np.linalg.norm(result)) <= 1e-10 * scale):
            return None
        return result

    a_residual, b_residual = residual(a), residual(b)
    if a_residual is None or b_residual is None:
        return np.nan
    return correlate(a_residual, b_residual)


def identity_values(correct: int, candidates: list[int],
                    value_by_scene: dict[int, float]) -> dict:
    """Correct-vs-other-scene rank and descriptives."""
    real = float(value_by_scene.get(int(correct), np.nan))
    wrong = np.asarray([
        value_by_scene.get(int(scene), np.nan)
        for scene in candidates if int(scene) != int(correct)
    ], dtype=float)
    wrong = wrong[np.isfinite(wrong)]
    return {
        "r_correct": real,
        "wrong_mean_r": float(np.mean(wrong)) if wrong.size else np.nan,
        "delta_r": float(real - np.mean(wrong)) if np.isfinite(real) and wrong.size else np.nan,
        "percentile": percentile_within_null(real, wrong),
        "n_distractors": int(wrong.size),
    }


def pair_scene_table(pairs: pd.DataFrame, scenes: pd.DataFrame) -> pd.DataFrame:
    """Map existing pair endpoints to the independent Chen 50-scene units."""
    rows = []
    for _, pair in pairs.iterrows():
        plant = scene_of_time(scenes, float(pair["plant_start_s"]))
        payoff = scene_of_time(scenes, float(pair["payoff_start_s"]))
        if plant is None or payoff is None or plant >= payoff:
            raise ValueError(f"{pair['pair_id']}: invalid plant/payoff scene mapping")
        rows.append({
            "pair_index": len(rows),
            "pair_id": str(pair["pair_id"]),
            "plant_scene": int(plant),
            "payoff_scene": int(payoff),
        })
    return pd.DataFrame(rows)


def score_subject(
    subject: str,
    roi: str,
    mapping: pd.DataFrame,
    movie_patterns: dict[int, np.ndarray | None],
    recall_patterns: dict[int, np.ndarray | None],
    shift: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Score plant->recall-payoff cells, all candidate plants, and validity."""
    plant_scenes = sorted(mapping["plant_scene"].astype(int).unique())
    pair_rows, comparisons, validity_rows = [], [], []

    recalled = sorted(scene for scene, pattern in recall_patterns.items() if pattern is not None)
    for scene in recalled:
        values = {
            other: correlate(movie_patterns.get(other), recall_patterns.get(scene))
            for other in recalled
        }
        score = identity_values(scene, recalled, values)
        validity_rows.append({
            "subject": subject, "roi": roi, "shift_trs": int(shift),
            "scene": int(scene), "same_scene_r": score["r_correct"],
            "same_scene_delta_r": score["delta_r"],
            "same_scene_percentile": score["percentile"],
            "n_distractors": score["n_distractors"],
        })

    for _, pair in mapping.iterrows():
        plant = int(pair["plant_scene"])
        payoff = int(pair["payoff_scene"])
        recall_payoff = recall_patterns.get(payoff)
        if recall_payoff is None:
            continue
        payoff_movie = movie_patterns.get(payoff)
        raw = {
            scene: correlate(movie_patterns.get(scene), recall_payoff)
            for scene in plant_scenes
        }
        # If a scene is both another relation's plant and the current payoff,
        # controlling the payoff from that same candidate leaves a zero vector.
        # It is therefore not a defined payoff-controlled distractor.
        partial_candidates = [scene for scene in plant_scenes if scene != payoff]
        partial = {
            scene: partial_correlate(
                movie_patterns.get(scene), recall_payoff, payoff_movie
            )
            for scene in partial_candidates
        }
        raw_score = identity_values(plant, plant_scenes, raw)
        partial_score = identity_values(plant, partial_candidates, partial)

        # A payoff scene can itself serve as the plant of a different locked
        # relation. Exclude it here: correlating a recall pattern with itself
        # would be a tautological distractor in this secondary analysis.
        recalled_plants = [
            scene for scene in plant_scenes
            if scene != payoff and recall_patterns.get(scene) is not None
        ]
        recall_values = {
            scene: correlate(recall_patterns.get(scene), recall_payoff)
            for scene in recalled_plants
        }
        recall_score = identity_values(plant, recalled_plants, recall_values)
        pair_rows.append({
            "subject": subject,
            "roi": roi,
            "shift_trs": int(shift),
            "pair_index": int(pair["pair_index"]),
            "pair_id": str(pair["pair_id"]),
            "plant_scene": plant,
            "payoff_scene": payoff,
            "r_correct": raw_score["r_correct"],
            "wrong_plant_mean_r": raw_score["wrong_mean_r"],
            "identity_delta_r": raw_score["delta_r"],
            "identity_percentile": raw_score["percentile"],
            "n_wrong_plants": raw_score["n_distractors"],
            "partial_r_correct": partial_score["r_correct"],
            "partial_wrong_mean_r": partial_score["wrong_mean_r"],
            "partial_identity_delta_r": partial_score["delta_r"],
            "partial_identity_percentile": partial_score["percentile"],
            "payoff_movie_recall_r": correlate(payoff_movie, recall_payoff),
            "plant_recalled": int(recall_patterns.get(plant) is not None),
            "recall_identity_delta_r": recall_score["delta_r"],
            "recall_identity_percentile": recall_score["percentile"],
            "n_recalled_wrong_plants": recall_score["n_distractors"],
        })
        finite_raw = {k: v for k, v in raw.items() if np.isfinite(v)}
        finite_partial = {k: v for k, v in partial.items() if np.isfinite(v)}
        for candidate, value in finite_raw.items():
            other = np.asarray([v for k, v in finite_raw.items() if k != candidate])
            partial_value = float(finite_partial.get(candidate, np.nan))
            partial_other = np.asarray([
                v for k, v in finite_partial.items() if k != candidate
            ])
            comparisons.append({
                "subject": subject,
                "roi": roi,
                "shift_trs": int(shift),
                "pair_index": int(pair["pair_index"]),
                "pair_id": str(pair["pair_id"]),
                "candidate_plant_scene": int(candidate),
                "r": float(value),
                "candidate_percentile": percentile_within_null(float(value), other),
                "partial_r": partial_value,
                "partial_candidate_percentile": percentile_within_null(
                    partial_value, partial_other
                ),
            })
    return pair_rows, comparisons, validity_rows


def subject_scores(cells: pd.DataFrame, minimum_pairs: int) -> pd.DataFrame:
    metrics = [
        "identity_percentile",
        "partial_identity_percentile",
        "recall_identity_percentile",
    ]
    rows = []
    for (roi, shift, subject), frame in cells.groupby(["roi", "shift_trs", "subject"]):
        for metric in metrics:
            values = frame[metric].dropna()
            if len(values) >= int(minimum_pairs):
                rows.append({
                    "roi": roi, "shift_trs": int(shift), "subject": subject,
                    "metric": metric, "n_pairs": int(len(values)),
                    "value": float(values.mean()),
                })
    return pd.DataFrame(rows)


def order_valid_label_assignments(mapping: pd.DataFrame, n: int,
                                  seed: int) -> list[tuple[int, ...]]:
    """Sample unique multiset label assignments that preserve temporal order."""
    plants = mapping["plant_scene"].to_numpy(dtype=int)
    payoffs = mapping["payoff_scene"].to_numpy(dtype=int)
    observed = tuple(int(v) for v in plants)
    rng = np.random.default_rng(int(seed))
    found: set[tuple[int, ...]] = set()
    attempts = 0
    limit = max(100_000, int(n) * 2_000)
    while len(found) < int(n) and attempts < limit:
        attempts += 1
        assignment = tuple(int(v) for v in rng.permutation(plants))
        if assignment == observed or assignment in found:
            continue
        if all(plant < payoff for plant, payoff in zip(assignment, payoffs)):
            found.add(assignment)
    if len(found) < int(n):
        raise RuntimeError(
            f"only {len(found)} unique time-valid label permutations found after {attempts} attempts"
        )
    return sorted(found)


def label_permutation_test(comparisons: pd.DataFrame, mapping: pd.DataFrame,
                           assignments: list[tuple[int, ...]],
                           minimum_pairs: int,
                           value_column: str = "candidate_percentile",
                           metric: str = "identity_percentile") -> pd.DataFrame:
    """Group mean subject-rank under time-direction-preserving pair labels."""
    if value_column not in comparisons:
        raise ValueError(f"missing permutation value column: {value_column}")
    rows = []
    observed = tuple(mapping["plant_scene"].astype(int))
    for roi, frame in comparisons.groupby("roi"):
        lookup = {
            (str(row.subject), int(row.pair_index), int(row.candidate_plant_scene)):
                float(getattr(row, value_column))
            for row in frame.itertuples()
        }
        subjects = sorted(frame["subject"].astype(str).unique(), key=int)

        def statistic(assignment: tuple[int, ...]) -> float:
            subject_values = []
            for subject in subjects:
                values = [
                    lookup.get((subject, pair_index, int(plant)), np.nan)
                    for pair_index, plant in enumerate(assignment)
                ]
                values = np.asarray(values, dtype=float)
                values = values[np.isfinite(values)]
                if values.size >= int(minimum_pairs):
                    subject_values.append(float(values.mean()))
            return float(np.mean(subject_values)) if subject_values else np.nan

        observed_value = statistic(observed)
        null = np.asarray([statistic(a) for a in assignments], dtype=float)
        null = null[np.isfinite(null)]
        p = float((1 + np.sum(null >= observed_value)) / (null.size + 1))
        rows.append({
            "metric": str(metric), "roi": roi, "n_subjects": len(subjects),
            "n_permutations": int(null.size), "observed": observed_value,
            "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
            "p_greater": p,
        })
    out = pd.DataFrame(rows)
    reject, adjusted = fdr(out["p_greater"])
    out["p_fdr_rois"] = adjusted
    out["significant_fdr_rois"] = reject
    return out


def validity_stats(validity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject = (
        validity.groupby(["roi", "shift_trs", "subject"])
        .agg(n_scenes=("same_scene_percentile", "count"),
             mean_percentile=("same_scene_percentile", "mean"),
             mean_delta_r=("same_scene_delta_r", "mean"))
        .reset_index()
    )
    rows = []
    for (roi, shift), frame in subject.groupby(["roi", "shift_trs"]):
        rows.append({
            "roi": roi, "shift_trs": int(shift),
            **directional_wilcoxon(frame["mean_percentile"], 0.5, "greater"),
            "mean_delta_r": float(frame["mean_delta_r"].mean()),
        })
    return subject, pd.DataFrame(rows)
