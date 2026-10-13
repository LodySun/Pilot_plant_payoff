"""Export the order-valid label-permutation null for recall identity (Figure 3).

src.run_chen_timing_corrected tests raw plant identity against 10,000
order-valid relabelings (every plant stays before its payoff) but keeps only
the null's mean and SD. Figure 3 draws the null's central 95%, so this script
regenerates the same 10,000 draws -- same assignments, same seed, same
statistic as label_permutation_test -- and checks that their mean and SD
reproduce results/chen_timing_corrected/order_valid_permutation.csv exactly.

Run:  python -m src.export_order_valid_null
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common import REPO_ROOT, load_config, stable_seed
from src.run_recall_bold import order_valid_label_assignments

RES = REPO_ROOT / "results/chen_timing_corrected"
ROIS = ("aud_early_chen", "pmc_chen")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    minimum = int(cfg["chen_recall_bold"]["minimum_subject_pairs"])
    mapping = pd.read_csv(RES / "pair_scene_mapping.csv").sort_values("pair_index")
    assignments = np.asarray(order_valid_label_assignments(
        mapping, int(cfg["chen_timing_corrected"]["label_permutations"]),
        stable_seed("chen_recall_bold_labels", int(cfg["seed"]))))
    comp = pd.read_csv(RES / "candidate_comparisons.csv")
    comp = comp[comp.shift_trs == 0]

    rows = []
    for roi in ROIS:
        frame = comp[comp.roi == roi]
        subjects = sorted(frame.subject.astype(str).unique(), key=int)
        scenes = sorted(frame.candidate_plant_scene.unique())
        cube = np.full((len(subjects), len(mapping), len(scenes)), np.nan)
        s_idx = {s: i for i, s in enumerate(subjects)}
        c_idx = {c: i for i, c in enumerate(scenes)}
        for r in frame.itertuples():
            cube[s_idx[str(r.subject)], int(r.pair_index),
                 c_idx[int(r.candidate_plant_scene)]] = r.candidate_percentile
        picks = np.vectorize(c_idx.get)(assignments)               # (n_perm, n_pairs)
        vals = cube[:, np.arange(len(mapping))[None, :], picks]    # (subj, n_perm, n_pairs)
        count = np.isfinite(vals).sum(axis=2)
        means = np.where(count >= minimum, np.nanmean(vals, axis=2), np.nan)
        draws = np.nanmean(means, axis=0)
        rows += [{"roi": roi, "draw": i, "value": v} for i, v in enumerate(draws)]
    out = pd.DataFrame(rows)

    ref = pd.read_csv(RES / "order_valid_permutation.csv")
    ref = ref[ref.metric == "identity_percentile"].set_index("roi")
    for roi in ROIS:
        x = out[out.roi == roi].value
        if (abs(x.mean() - ref.loc[roi, "null_mean"]) > 1e-9
                or abs(x.std(ddof=0) - ref.loc[roi, "null_sd"]) > 1e-9):
            raise RuntimeError(f"{roi}: regenerated null does not match order_valid_permutation.csv")
    out.to_csv(RES / "order_valid_null_draws.csv", index=False)
    print(f"wrote {RES / 'order_valid_null_draws.csv'} (mean/SD match the permutation test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
