"""Source-aligned audit of compact-Chen movie-to-recall plant identity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src import data_io
from src.build_controls import load_pairs
from src.common import (
    REPO_ROOT, ensure_dir, load_config, rel, require_lock_or_exit, resolve,
    stable_seed,
)
from src.run_recall_bold import (
    label_permutation_test, order_valid_label_assignments, pair_scene_table,
    score_subject, subject_scores, validity_stats,
)
from src.chen_scene_patterns import extract
from src.stats_utils import directional_wilcoxon, fdr, fmt_p


METRICS = ("identity_percentile", "partial_identity_percentile")


def correction_stats(scores: pd.DataFrame, alpha: float = .05) -> pd.DataFrame:
    rows = []
    for (metric, roi), frame in scores.groupby(["metric", "roi"]):
        if metric not in METRICS:
            continue
        rows.append({"metric": metric, "roi": roi,
                     **directional_wilcoxon(frame.value, .5, "greater")})
    out = pd.DataFrame(rows)
    out["q_within_metric"] = np.nan
    out["significant_within_metric"] = False
    for _, indices in out.groupby("metric").groups.items():
        reject, q = fdr(out.loc[indices, "p_value"], alpha=alpha)
        out.loc[indices, "q_within_metric"] = q
        out.loc[indices, "significant_within_metric"] = reject
    reject, q = fdr(out.p_value, alpha=alpha)
    out["q_all_four"] = q
    out["significant_all_four"] = reject
    return out


def deletion_checks(cells: pd.DataFrame, minimum_pairs: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = cells[cells.roi == "pmc_chen"].copy()
    subject_rows, pair_rows = [], []
    for metric in METRICS:
        for omitted in sorted(primary.subject.astype(str).unique(), key=int):
            kept = primary[primary.subject.astype(str) != omitted]
            values = (kept.groupby("subject")[metric].agg(["count", "mean"])
                      .query("count >= @minimum_pairs")["mean"])
            subject_rows.append({"metric": metric, "left_out_subject": omitted,
                                 **directional_wilcoxon(values, .5, "greater")})
        for omitted in sorted(primary.pair_id.unique()):
            kept = primary[primary.pair_id != omitted]
            values = (kept.groupby("subject")[metric].agg(["count", "mean"])
                      .query("count >= @minimum_pairs")["mean"])
            pair_rows.append({"metric": metric, "left_out_pair": omitted,
                              **directional_wilcoxon(values, .5, "greater")})
    return pd.DataFrame(subject_rows), pd.DataFrame(pair_rows)


def ceiling_and_scene_checks(cells: pd.DataFrame, validity: pd.DataFrame,
                             mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Payoff same-scene benchmark and scene-level selection diagnostics."""
    payoff_scenes = set(mapping.payoff_scene.astype(int))
    same = validity[
        (validity.roi == "pmc_chen") & validity.scene.isin(payoff_scenes)
    ][["subject", "scene", "same_scene_percentile"]].drop_duplicates(
        ["subject", "scene"]
    )
    plant = cells[cells.roi == "pmc_chen"][
        ["subject", "payoff_scene", "identity_percentile",
         "partial_identity_percentile"]
    ].groupby(["subject", "payoff_scene"], as_index=False).mean(numeric_only=True)
    joined = plant.merge(
        same, left_on=["subject", "payoff_scene"],
        right_on=["subject", "scene"], validate="one_to_one",
    )
    # Use exactly the intersecting unique subject x payoff-scene cells for both
    # quantities.  At present the sets are identical (115/115), but keeping the
    # comparison paired prevents future missingness from changing the ratio.
    raw_subject = joined.groupby("subject").identity_percentile.mean()
    same_subject = joined.groupby("subject").same_scene_percentile.mean()
    raw_mean = float(raw_subject.mean())
    same_mean = float(same_subject.mean())
    ceiling = pd.DataFrame([{
        "benchmark": "same payoff scene movie-to-recall identity",
        "n_subjects": int(same_subject.size),
        "n_unique_subject_payoff_scenes": int(len(joined)),
        "raw_plant_identity_mean": raw_mean,
        "same_scene_identity_mean": same_mean,
        "raw_fraction_of_chance_to_same_scene_lift": (
            (raw_mean - .5) / (same_mean - .5)
        ),
        "note": "empirical within-scene benchmark, not a mathematical noise ceiling",
    }])
    scene = joined.groupby("payoff_scene", as_index=False).agg(
        same_scene_identity=("same_scene_percentile", "mean"),
        plant_identity=("identity_percentile", "mean"),
        partial_plant_identity=("partial_identity_percentile", "mean"),
        recall_coverage=("subject", "size"),
    )
    raw_s = spearmanr(scene.same_scene_identity, scene.plant_identity)
    partial_s = spearmanr(scene.same_scene_identity, scene.partial_plant_identity)
    coverage_s = spearmanr(scene.recall_coverage, scene.plant_identity)
    raw_p = pearsonr(scene.same_scene_identity, scene.plant_identity)
    scene.attrs["diagnostics"] = {
        "n_payoff_scenes": int(len(scene)),
        "same_scene_vs_raw_plant_spearman_rho": float(raw_s.statistic),
        "same_scene_vs_raw_plant_spearman_p": float(raw_s.pvalue),
        "same_scene_vs_raw_plant_pearson_r": float(raw_p.statistic),
        "same_scene_vs_raw_plant_pearson_p": float(raw_p.pvalue),
        "same_scene_vs_partial_plant_spearman_rho": float(partial_s.statistic),
        "same_scene_vs_partial_plant_spearman_p": float(partial_s.pvalue),
        "coverage_vs_raw_plant_spearman_rho": float(coverage_s.statistic),
        "coverage_vs_raw_plant_spearman_p": float(coverage_s.pvalue),
    }
    return ceiling, scene


def write_summary(path: Path, tests: pd.DataFrame, permutations: pd.DataFrame,
                  subject_loso: pd.DataFrame, pair_lopo: pd.DataFrame,
                  cells: pd.DataFrame, ceiling: pd.DataFrame,
                  scene_diagnostics: dict) -> None:
    lines = [
        "# Source-aligned Chen recall audit", "",
        "> **Timing-correction analysis, not a new prospective confirmation.**",
        "> The compact Chen movie and recall MAT files already contain the",
        "> author's 3-TR haemodynamic shift. These results add 0 further TRs.", "",
        "The locked 11 plant-payoff relations are unchanged. Raw identity retains",
        "the historical candidate set. Payoff-controlled identity excludes the",
        "current payoff when it is also another relation's plant, because its",
        "residual after controlling itself is undefined.", "",
        "## Participant-first tests", "",
        "| metric | ROI | n | mean percentile | median | p (greater) | q, 2 ROIs | q, all 4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in tests.sort_values(["metric", "roi"]).itertuples():
        lines.append(
            f"| `{row.metric}` | `{row.roi}` | {row.n} | {row.mean:.3f} | "
            f"{row.median:.3f} | {fmt_p(row.p_value)} | "
            f"{fmt_p(row.q_within_metric)} | {fmt_p(row.q_all_four)} |"
        )
    lines += ["", "## Order-valid plant-label permutations", "",
              "Each payoff and each participant's recall coverage remain fixed; only",
              "time-valid plant labels are reassigned.", "",
              "| metric | ROI | observed | null mean | p | q, 2 ROIs | q, all 4 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in permutations.sort_values(["metric", "roi"]).itertuples():
        lines.append(
            f"| `{row.metric}` | `{row.roi}` | {row.observed:.3f} | "
            f"{row.null_mean:.3f} | {fmt_p(row.p_greater)} | "
            f"{fmt_p(row.p_fdr_rois)} | {fmt_p(row.q_all_four)} |"
        )
    lines += ["", "## Influence diagnostics", ""]
    for metric in METRICS:
        s = subject_loso[subject_loso.metric == metric]
        p = pair_lopo[pair_lopo.metric == metric]
        lines.append(
            f"- `{metric}`: leave-one-subject-out p range "
            f"{s.p_value.min():.4f}-{s.p_value.max():.4f}; leave-one-pair-out "
            f"p range {p.p_value.min():.4f}-{p.p_value.max():.4f}."
        )
    pmc = cells[(cells.roi == "pmc_chen") & cells.identity_percentile.notna()]
    per_subject = pmc.groupby("subject").size()
    benchmark = ceiling.iloc[0]
    lines += ["", "## Coverage", "",
              f"PMC contributes **{len(pmc)}** subject x relation cells from "
              f"**{pmc.subject.nunique()}** participants. Cells per participant "
              f"range from **{per_subject.min()}** to **{per_subject.max()}** "
              f"(median {per_subject.median():.0f}).", "",
              "## Empirical same-scene benchmark", "",
              f"The source-aligned payoff same-scene movie-to-recall benchmark is "
              f"**{benchmark.same_scene_identity_mean:.3f}** across "
              f"{int(benchmark.n_unique_subject_payoff_scenes)} matched, unique "
              "subject x payoff-scene cells. The raw plant mean on that "
              f"deduplicated set is **{benchmark.raw_plant_identity_mean:.3f}**, reaching "
              f"**{benchmark.raw_fraction_of_chance_to_same_scene_lift:.1%}** "
              "of the lift from chance (.5) to that benchmark. This is an empirical "
              "within-scene reference, not a mathematical ceiling.", "",
              f"Across the 10 payoff scenes, same-scene quality and raw plant identity "
              f"were negatively associated (Spearman rho="
              f"{scene_diagnostics['same_scene_vs_raw_plant_spearman_rho']:+.3f}, "
              f"p={scene_diagnostics['same_scene_vs_raw_plant_spearman_p']:.3f}); "
              "recall coverage did not significantly track raw plant identity "
              f"(rho={scene_diagnostics['coverage_vs_raw_plant_spearman_rho']:+.3f}, "
              f"p={scene_diagnostics['coverage_vs_raw_plant_spearman_p']:.3f}). "
              "These n=10 diagnostics are descriptive.", "",
              "Historical +3-TR outputs remain in `results/chen_recall_bold/` and",
              "must not be described as the source-required HRF correction."]
    path.write_text("\n".join(lines) + "\n")


def run(cfg: dict) -> Path:
    section = cfg["chen_timing_corrected"]
    base = cfg["chen_recall_bold"]
    shift = int(section["additional_shift_trs"])
    if shift != 0:
        raise ValueError("source-aligned compact-Chen audit must add zero TRs")
    scenes = pd.read_csv(resolve(cfg, "derived") / "scenes.csv")
    mapping = pair_scene_table(load_pairs(cfg, False), scenes)
    scene_ids = sorted(scenes.loc[~scenes.is_cartoon, "scene_id"].astype(int).unique())
    rois = [str(base["primary_roi"]), str(base["control_roi"])]
    subjects = data_io.common_subjects(cfg, rois)
    all_cells, all_comparisons, all_validity, audit = [], [], [], []
    for roi in rois:
        movie, recall, extracted = extract(
            cfg, roi, subjects, scene_ids, shift, normalize=False
        )
        audit.extend(extracted)
        for i, subject in enumerate(subjects):
            movie_patterns = {
                scene: (movie[i, j] if np.isfinite(movie[i, j]).all() else None)
                for j, scene in enumerate(scene_ids)
            }
            recall_patterns = {
                scene: (recall[i, j] if np.isfinite(recall[i, j]).all() else None)
                for j, scene in enumerate(scene_ids)
            }
            scored, comparisons, validity_rows = score_subject(
                subject, roi, mapping, movie_patterns, recall_patterns, shift
            )
            all_cells.extend(scored)
            all_comparisons.extend(comparisons)
            all_validity.extend(validity_rows)
    cells = pd.DataFrame(all_cells)
    comparisons = pd.DataFrame(all_comparisons)
    validity = pd.DataFrame(all_validity)
    minimum = int(base["minimum_subject_pairs"])
    scores = subject_scores(cells, minimum)
    tests = correction_stats(scores, float(cfg["analysis"]["fdr_alpha"]))
    assignments = order_valid_label_assignments(
        mapping, int(section["label_permutations"]),
        stable_seed("chen_recall_bold_labels", int(cfg["seed"])),
    )
    permutations = pd.concat([
        label_permutation_test(
            comparisons, mapping, assignments, minimum,
            value_column="candidate_percentile", metric="identity_percentile",
        ),
        label_permutation_test(
            comparisons, mapping, assignments, minimum,
            value_column="partial_candidate_percentile",
            metric="partial_identity_percentile",
        ),
    ], ignore_index=True)
    permutations["q_all_four"] = fdr(
        permutations.p_greater, alpha=float(cfg["analysis"]["fdr_alpha"])
    )[1]
    subject_loso, pair_lopo = deletion_checks(cells, minimum)
    validity_subjects, validity_group = validity_stats(validity)
    ceiling, scene_table = ceiling_and_scene_checks(cells, validity, mapping)
    scene_diagnostics = scene_table.attrs["diagnostics"]

    out = ensure_dir(REPO_ROOT / str(section["results_dir"]))
    cells.to_csv(out / "pair_cells.csv", index=False)
    comparisons.to_csv(out / "candidate_comparisons.csv", index=False)
    scores.to_csv(out / "subject_scores.csv", index=False)
    tests.to_csv(out / "group_tests.csv", index=False)
    permutations.to_csv(out / "order_valid_permutation.csv", index=False)
    subject_loso.to_csv(out / "leave_one_subject_out.csv", index=False)
    pair_lopo.to_csv(out / "leave_one_pair_out.csv", index=False)
    validity.to_csv(out / "same_scene_validity_cells.csv", index=False)
    validity_subjects.to_csv(out / "same_scene_validity_subjects.csv", index=False)
    validity_group.to_csv(out / "same_scene_validity_group.csv", index=False)
    ceiling.to_csv(out / "empirical_ceiling.csv", index=False)
    scene_table.to_csv(out / "payoff_scene_diagnostics.csv", index=False)
    (out / "payoff_scene_diagnostics.json").write_text(
        json.dumps(scene_diagnostics, indent=2) + "\n"
    )
    mapping.to_csv(out / "pair_scene_mapping.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "extraction_audit.csv", index=False)
    meta = {
        "analysis": "source-aligned correction audit of compact Chen recall RSA",
        "frozen_plan": "docs/chen_timing_correction_plan.md",
        "additional_shift_trs": shift,
        "historical_additional_shift_trs": int(base["hrf_shift_trs"]),
        "partial_payoff_candidate_excluded": True,
        "subjects": subjects,
        "n_pairs": int(len(mapping)),
        "label_permutations": int(section["label_permutations"]),
        "annotation_lock_unchanged": True,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    write_summary(out / "summary.md", tests, permutations, subject_loso,
                  pair_lopo, cells, ceiling, scene_diagnostics)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    require_lock_or_exit(cfg, smoke=False)
    out = run(cfg)
    print(f"Wrote {rel(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
