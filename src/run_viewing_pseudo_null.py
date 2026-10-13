"""Matched pseudo-narrative viewing null for PMC (poster Figure 2).

For each participant and locked pair, the viewing-pattern similarity between
plant and payoff in PMC is ranked against distance-matched control moment
pairs (Family A: same temporal gap), giving a matched-null percentile. The
locked pairs' mean percentile is then compared with 10,000 pseudo-narratives,
each of which replaces every locked pair by one of its matched controls.

Two settings matter and are fixed here:

1. Source-aligned timing. The Chen compact-release MAT files already carry
   the authors' 3-TR haemodynamic shift, so no further shift is added.
2. A Family-A-only control pool. The control table also holds rows that only
   qualify for Family B (feature-matched); those are excluded here.

The resulting p is descriptive -- a re-presentation of the matched-control
comparison, not an independent test.

Run:  python -m src.run_viewing_pseudo_null
"""
from __future__ import annotations

import argparse
import copy

import pandas as pd

from src import run_analysis as RA
from src.build_controls import build, load_pairs
from src.common import REPO_ROOT, load_config, load_movie_timeline, require_lock_or_exit, run_bounds
from src.data_quality import load as load_invalid
from src.patterns import WindowMaker, segment_bounds_from_scenes
from src.run_pseudo_dataset_null import analyse_branch, build_definitions, make_plot

OUT = REPO_ROOT / "results/viewing_pseudo_null"
ROI = "pmc_chen"
GZIP = {"method": "gzip", "mtime": 0}     # no timestamp: reruns are byte-identical


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    require_lock_or_exit(cfg, smoke=False)
    cfg0 = copy.deepcopy(cfg)
    cfg0["acquisition"]["hrf_shift_trs"] = 0          # source-aligned for Chen

    timeline = load_movie_timeline(cfg)
    n_trs, runs = int(timeline["total_trs"]), run_bounds(timeline)
    pairs = load_pairs(cfg, False).reset_index(drop=True)
    bounds = segment_bounds_from_scenes(pd.read_csv(REPO_ROOT / "data/derived/scenes.csv"))
    window = WindowMaker(cfg0, bounds)
    invalid = load_invalid(cfg)
    reps = int(cfg["pseudo_dataset_null"]["permutations"])

    print("building control pairs at 0 extra TRs ...")
    controls = build(cfg0, pairs, verbose=False)[0]
    controls_a = controls[controls["family_a"]].reset_index(drop=True)
    print(f"  Family-A rows only: {len(controls_a)} (of {len(controls)} total)")

    primary, exact, feasibility = build_definitions(
        cfg0, pairs, controls_a, timeline, window, bounds)

    subjects = [str(s) for s in RA.resolve_rois(cfg, False)[ROI]["subjects"]]
    summary, draws, cells = [], [], []
    for null_type, defs in (("family_a_segment", primary),
                            ("exact_gap_fixed_duration", exact)):
        print(f"  {ROI} / {null_type} ...", flush=True)
        stats, d, c, _ = analyse_branch(cfg0, ROI, subjects, defs, n_trs, runs,
                                        invalid, False, reps, null_type)
        stats["null_type"] = null_type
        summary.append(stats)
        draws += [{"roi": ROI, "null_type": null_type, "draw": i,
                   "dataset_score": v} for i, v in enumerate(d)]
        for row in c:
            row["null_type"] = null_type
        cells += c
    summary = pd.DataFrame(summary)
    draws = pd.DataFrame(draws)
    cells = pd.DataFrame(cells)

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "roi_summary.csv", index=False)
    draws.to_csv(OUT / "pseudo_draws.csv.gz", index=False, compression=GZIP)
    cells.to_csv(OUT / "cell_scores.csv.gz", index=False, compression=GZIP)
    feasibility.to_csv(OUT / "exact_gap_feasibility.csv", index=False)
    for null_type, name, flag in (("family_a_segment", "pseudo_null_primary", False),
                                  ("exact_gap_fixed_duration", "exact_gap_sensitivity", True)):
        row = summary[summary.null_type == null_type].iloc[0]
        x = draws[draws.null_type == null_type].dataset_score.to_numpy()
        make_plot(x, row, OUT / name, flag)

    lines = ["# Pseudo-narrative viewing null -- PMC", "",
             "Source-aligned timing (0 extra TRs; the Chen files already carry a 3-TR",
             f"shift) and a Family-A-only control pool; {reps:,} pseudo-narratives.",
             "Descriptive: a re-presentation of the matched-control comparison,",
             "not an independent test.", "",
             "| null | observed | pseudo mean (SD) | descriptive upper-tail p |",
             "|---|---:|---:|---:|"]
    for _, r in summary.iterrows():
        lines.append(f"| `{r.null_type}` | {r.observed:.4f} | {r.pseudo_mean:.4f} "
                     f"({r.pseudo_sd:.4f}) | {r.pseudo_p_greater:.3f} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[7:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
