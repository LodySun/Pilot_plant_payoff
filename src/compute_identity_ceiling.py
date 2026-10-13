"""Reliability (noise) ceiling for the recall plant-identity percentile.

Question: in the same ROI, the same source-aligned windows, the same 11-pair
structure, the same 9 plant candidates and the same percentile function, how
high could ANY fixed (subject-invariant) plant assignment score, given how
reproducible the candidate ranking is across participants?

This is the Nili et al. (2014) noise-ceiling logic applied to this retrieval
problem. The locked annotations are one fixed hypothesis shared by every
subject, so no fixed hypothesis can predict subject-specific noise:

  lower bound (leave-one-subject-out): for subject s and pair k, pick the
      candidate the OTHER 16 subjects rank highest on average; score subject
      s's own percentile for that candidate.
  upper bound (all subjects): the same, but the pick also uses subject s
      (slightly overfit, as in Nili's upper bound).

Per-cell percentiles are then averaged within subject (>= 3 pairs, as in the
primary analysis) and across subjects -- exactly how identity_percentile is
aggregated. Ties in the pick are averaged.

Hypothesis space: "order-valid" candidates only (plant scene earlier than the
payoff scene), which is the space the true plants and the order-valid
permutation null live in. "All 9 candidates" is reported as a looser variant.

Input: results/chen_timing_corrected/candidate_comparisons.csv (every subject x
pair x candidate percentile, written by src.run_chen_timing_corrected). No
fMRI data is re-read.

Run:  python -m src.compute_identity_ceiling
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from src.common import REPO_ROOT  # noqa: E402

RES = REPO_ROOT / "results/chen_timing_corrected"
OUT = REPO_ROOT / "results/identity_ceiling"
MIN_PAIRS = 3
METRICS = {"identity_percentile": ("candidate_percentile",),
           "partial_identity_percentile": ("partial_candidate_percentile",)}


def aggregate(cell_values: pd.DataFrame, column: str) -> float:
    per_subject = (cell_values.dropna(subset=[column]).groupby("subject")[column]
                   .agg(["count", "mean"]).query("count >= @MIN_PAIRS")["mean"])
    return float(per_subject.mean()), int(per_subject.size)


def ceiling(comp: pd.DataFrame, column: str, order_valid: bool, leave_out: bool) -> tuple:
    comp = comp.dropna(subset=[column])
    if order_valid:
        comp = comp[comp.candidate_plant_scene < comp.payoff_scene]
    rows, hits = [], []
    for pair, frame in comp.groupby("pair_id"):
        table = frame.pivot(index="subject", columns="candidate_plant_scene", values=column)
        for subject in table.index:
            pool = table.drop(index=subject) if leave_out else table
            mean_rank = pool.mean(axis=0, skipna=True)
            best = mean_rank[mean_rank == mean_rank.max()].index
            own = table.loc[subject, best].dropna()
            if own.empty:
                continue
            rows.append({"subject": subject, "pair_id": pair, "value": float(own.mean())})
            hits.append(int(frame.plant_scene.iloc[0] in set(best)))
    value, n = aggregate(pd.DataFrame(rows), "value")
    return value, n, float(np.mean(hits))


def main() -> int:
    comp = pd.read_csv(RES / "candidate_comparisons.csv")
    comp = comp[comp.shift_trs == 0]
    mapping = pd.read_csv(RES / "pair_scene_mapping.csv").set_index("pair_id")
    comp = comp.assign(plant_scene=comp.pair_id.map(mapping.plant_scene),
                       payoff_scene=comp.pair_id.map(mapping.payoff_scene))
    perm = pd.read_csv(RES / "order_valid_permutation.csv").set_index(["roi", "metric"])
    bench = pd.read_csv(RES / "empirical_ceiling.csv").same_scene_identity_mean.iloc[0]

    rows = []
    for roi, roi_comp in comp.groupby("roi"):
        for metric, (column,) in METRICS.items():
            # sanity: the true-plant cells must reproduce the published score
            true = roi_comp[roi_comp.candidate_plant_scene == roi_comp.plant_scene]
            observed, n_obs = aggregate(true, column)
            published = float(perm.loc[(roi, metric), "observed"])
            if abs(observed - published) > 1e-9:
                raise RuntimeError(f"{roi} {metric}: {observed} != published {published}")
            row = {"roi": roi, "metric": metric, "n_subjects": n_obs,
                   "observed": observed,
                   "order_valid_null_mean": float(perm.loc[(roi, metric), "null_mean"]),
                   "order_valid_null_sd": float(perm.loc[(roi, metric), "null_sd"])}
            for space, order_valid in (("ordervalid", True), ("all9", False)):
                for bound, leave_out in (("lower_loso", True), ("upper_all", False)):
                    value, _, hit = ceiling(roi_comp, column, order_valid, leave_out)
                    row[f"ceiling_{space}_{bound}"] = value
                    row[f"pick_is_true_plant_{space}_{bound}"] = hit
            rows.append(row)
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "identity_noise_ceiling.csv", index=False)

    L = ["# Reliability ceiling for the recall plant-identity percentile", "",
         "Nili-style noise ceiling for this exact retrieval problem: same ROI, same",
         "source-aligned windows, same 11 pairs, same 9 plant candidates, same percentile",
         "and subject aggregation. Lower bound = leave-one-subject-out pick of the most",
         "consistent candidate; upper bound = pick using all subjects.", "",
         "| ROI | metric | observed | order-valid null | ceiling (order-valid) lower–upper | "
         "ceiling (all 9) lower–upper | observed as % of null→ceiling(lower) |",
         "|---|---|---:|---:|---:|---:|---:|"]
    for r in table.itertuples():
        span = r.ceiling_ordervalid_lower_loso - r.order_valid_null_mean
        # undefined when the ceiling does not rise above the null (no reliable ranking)
        frac = (r.observed - r.order_valid_null_mean) / span if span > 0 else None
        L.append(f"| {r.roi} | {r.metric} | {r.observed:.3f} | "
                 f"{r.order_valid_null_mean:.3f} ± {r.order_valid_null_sd:.3f} | "
                 f"{r.ceiling_ordervalid_lower_loso:.3f}–{r.ceiling_ordervalid_upper_all:.3f} | "
                 f"{r.ceiling_all9_lower_loso:.3f}–{r.ceiling_all9_upper_all:.3f} | "
                 f"{'n/a (ceiling ≤ null)' if frac is None else format(frac, '.0%')} |")
    n_valid = (mapping.reset_index().apply(
        lambda p: sum(s < p.payoff_scene for s in set(mapping.plant_scene)), axis=1))
    L += ["", f"Chance that a random order-valid pick is the annotated plant: "
          f"{float((1 / n_valid).mean()):.0%} (mean over pairs); all 9: {1/9:.0%}."]
    L += ["", f"For comparison, the same-scene benchmark (a different retrieval problem) is {bench:.3f}.",
          "", "Share of cells whose consistency pick is the annotated plant:", ""]
    for r in table.itertuples():
        L.append(f"- {r.roi} / {r.metric}: order-valid LOSO {r.pick_is_true_plant_ordervalid_lower_loso:.0%}, "
                 f"all-9 LOSO {r.pick_is_true_plant_all9_lower_loso:.0%}")
    (OUT / "identity_noise_ceiling.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
