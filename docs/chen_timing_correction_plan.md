# Chen compact-MAT timing correction audit

Frozen 2026-10-07, after the provenance finding in
`results/chen_source_audit/summary.md` and before inspecting any corrected
payoff-controlled or downstream result.

## Reason for this audit

The public Chen tutorial states that both compact movie and recall ROI MAT
files were already shifted by three TRs relative to their event labels.  The
historical pipeline then added another three TRs.  For these compact MAT files,
source-aligned extraction therefore uses **zero additional TRs**.  Historical
+3-TR results are retained as a displaced sensitivity analysis and are not
overwritten.

The audit also corrects a numerical problem in the historical
payoff-controlled score.  When a candidate plant scene is the current payoff
scene, regressing the payoff pattern from that same candidate leaves a
theoretically zero vector.  That candidate is undefined for partial
correlation and will be excluded from the payoff-controlled candidate set.
Near-zero residual vectors will also be rejected explicitly.

## Frozen reconstruction rules

1. Use the locked `annotations/pairs.csv` unchanged.
2. Use the released 50-scene labels and subject-specific recall intervals
   unchanged.
3. Use zero additional shift for both movie and recall compact Chen MAT data.
4. Preserve the historical raw candidate set.  For payoff-controlled scores
   only, exclude a candidate when its scene is the current payoff scene.
5. Aggregate cells within participant first, requiring at least three usable
   pairs, exactly as in the historical analysis.
6. Report one-sided Wilcoxon tests against .5 and 10,000 order-valid plant-label
   permutations for raw and payoff-controlled identity in PMC and early
   auditory cortex.  Report BH-FDR both within each two-ROI metric family and
   across the four audit tests.  These are correction-audit tests, not newly
   prospective confirmations.
7. Recompute leave-one-participant-out and leave-one-pair-out stability from
   the corrected PMC cells.
8. Write all corrected products under `results/chen_timing_corrected/`; never
   overwrite `results/chen_recall_bold/`.

## Analyses that use the corrected outcome

The following analyses will be rerun with the source-aligned cell table and
written to a `timing_corrected/` subdirectory of their existing result folder:

- spoken plant-content association (`chen_recall_content`);
- plant-surprise association (`plant_surprise`);
- payoff-time pattern-change predictors (`payoff_formation` and the
  hippocampal follow-up);
- payoff activation, connectivity, and baseline sensitivity.

For analyses whose predictor is also drawn from a compact Chen movie MAT,
that predictor uses zero additional shift.  Predictors drawn from separate
OpenNeuro/fMRIPrep NIfTI use their existing +3-TR sampling.  In mixed
PMC--hippocampal analyses the clocks are handled source by source rather than
with one shared shift.

The recall onset time course will be regenerated at zero additional shift and
with the corrected partial candidate set under
`results/chen_recall_timecourse/timing_corrected/`.  Its curve is descriptive;
the original lag search cannot become a new confirmatory test after this
timing finding.

## Explicitly outside this correction

- `wholebrain_recall` uses separate OpenNeuro/fMRIPrep NIfTI and retains its
  specified HRF shift.
- The 39/78-second episode-to-movie conversion is unchanged.
- Zadbood replication and unrelated datasets are unchanged.
- Existing historical figures and tables remain on disk for provenance, but
  poster-facing code must not silently mix historical +3-TR results with
  source-aligned results.

## Interpretation rule

The audit asks whether conclusions survive correct source timing.  A weaker,
null, or reversed result will be reported without selecting an alternative lag
or metric.  Corrected post-hoc analyses remain post-hoc, regardless of their
p-values.
