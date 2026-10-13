# Time-resolved plant identity during scanned recall

> **Exploratory relative BOLD profile, not a precise cognitive clock.**
> The locked relations were read unchanged. No relation or keyword was
> generated, edited, or inferred by this analysis.

Each released payoff-recall scene onset was treated as time zero. At every
lag, a full 3-TR (4.5-s) recall
window was sampled after the fixed +0-TR haemodynamic
shift. The plotted x axis removes that shift, so zero denotes the labelled
content onset rather than the BOLD sample time. Repeated recall blocks were
pooled at the same relative lag; scan-edge windows were omitted, never clipped.

Both raw plant identity and identity after linearly removing the matching
movie-payoff pattern are shown. Correct plant labels were compared with
10,000 order-valid reassignments. Peak p values use a
one-sided max-statistic across every sampled lag within a curve; the four
curve-level peak p values are then FDR-corrected.

| ROI | metric | peak time (s) | observed | null mean | excess | lag-FWER p | four-curve q |
|---|---|---:|---:|---:|---:|---:|---:|
| `pmc_chen` | `partial_identity_delta_r` | +4.5 | 0.057 | -0.022 | +0.079 | 0.020 | 0.079 |
| `pmc_chen` | `identity_delta_r` | +4.5 | 0.044 | -0.032 | +0.076 | 0.051 | 0.103 |
| `aud_early_chen` | `identity_delta_r` | -10.5 | 0.025 | 0.002 | +0.023 | 0.644 | 0.689 |
| `aud_early_chen` | `partial_identity_delta_r` | -10.5 | 0.023 | 0.002 | +0.021 | 0.689 | 0.689 |

## Primary interpretive curve

The payoff-controlled PMC curve peaked at **+4.5 s** 
(observed 0.057, label-null mean -0.022, 
lag-maxT p=0.020, four-curve q=0.079).
The lag-maxT-significant window starts were **+3.0, +4.5 s**.
No full-width window confined entirely before onset was significant.
The -1.5-s window already spans -1.5 to +3.0 s because every point
averages 4.5 s. The result therefore localises the strongest evidence
to windows containing the labelled onset and the first seconds after it;
it is not evidence that plant reinstatement preceded payoff narration.
A significant point would localise evidence only at BOLD resolution; it
would not by itself show that plant retrieval preceded speech or caused
the payoff narration.

## Coverage and interpretation limits

- The labelled-scene overlap of the cognitive window rises from 0.00
  at -18.0 s to 0.99 at onset and
  is 0.72 at +21.0 s.
- Adjacent points overlap heavily and must not be treated as independent tests.
- Scene boundaries come from the released coarse recall annotation, not from
  a new plant/payoff annotation. Time zero marks labelled content onset, not
  necessarily the participant's internal retrieval onset.
- Spoken production, auditory feedback, and neighbouring recalled scenes can
  affect the curve. The early-auditory ROI is a control for, not a complete
  removal of, those processes.
- This analysis was specified after the scene-averaged recall result was known
  and is therefore exploratory.

Outputs: `curve.csv`, `curve_summary.csv`, `subject_curves.csv`,
`pair_lag_cells.csv`, `window_coverage.csv`, `timecourse.png`, and `meta.json`.
