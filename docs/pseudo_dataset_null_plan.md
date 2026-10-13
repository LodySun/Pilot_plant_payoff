# Matched pseudo-narrative viewing null — frozen plan

This analysis is a design-level visualisation of the existing viewing Family A
control structure. It is **not an independent validation** and does not create,
edit, or infer any plant–payoff annotation.

## Primary pseudo-dataset null

The primary branch uses the same segment-based moment windows, exclusion mask,
run-span rule, and distance-matched control pool already used by Analysis 1.
For each pseudo dataset, exactly one matched control pair is selected for each
of the 11 locked human pairs. A control selection is stimulus-level, so the
same selected control is used for every participant rather than drawing a
different pseudo story for every brain.

Within each participant and locked pair, the selected control correlation is
ranked against that pair's full Family A-compatible control pool, with ties
counted as one half. Scores are averaged over available pairs within participant
and then equally over participants. Ten thousand complete pseudo datasets form
the displayed null distribution. The real vertical line is the identically
aggregated percentile of the locked plant–payoff pairs. The primary poster plot
uses the pre-existing Chen PMC ROI; every available viewing ROI is retained in
the output tables.

The plot title must say that this is a visualisation of Family A. Its p value is
descriptive of the same matched-control evidence and must never be presented as
a new independent test.

## Exact discrete-TR-gap sensitivity

The production segment windows cannot support an exact endpoint-length and
exact discrete-gap control pool for all 11 locked pairs: two pairs have zero
legal candidates. The primary analysis therefore remains unchanged.

A separately labelled sensitivity analysis uses the pipeline's already-defined
pair-specific fixed-duration windows (annotated duration divided by TR and
clamped to 2–4 TR). A valid control must match both endpoint lengths, the exact
integer gap `b_lo - a_hi`, and the real pair's within-run/cross-run status, while
obeying the same exclusions. One hundred controls per pair are selected with a
fixed seed. This retains all 11 pairs and repeats the complete pseudo-dataset
procedure above.

The output includes a feasibility table reporting, for every locked pair, the
number of exact candidates under both the production segment-window definition
and the fixed-duration sensitivity definition. The two nulls are never pooled.

## Outputs

- `roi_summary.csv`: observed values and pseudo-null summaries for all ROIs.
- `pseudo_draws.csv.gz`: all complete-dataset draws.
- `cell_scores.csv.gz`: participant × pair real percentiles.
- `exact_gap_feasibility.csv`: candidate counts and window properties.
- `pseudo_null_primary.png/.pdf`: single-distribution PMC poster graphic.
- `exact_gap_sensitivity.png/.pdf`: separately labelled exact-gap graphic.
- `summary.md`: plain-language methodological interpretation.

Smoke-test output is synthetic and routed under `results/smoke/`.
