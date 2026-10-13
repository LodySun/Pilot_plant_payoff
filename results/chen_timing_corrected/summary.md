# Source-aligned Chen recall audit

> **Timing-correction analysis, not a new prospective confirmation.**
> The compact Chen movie and recall MAT files already contain the
> author's 3-TR haemodynamic shift. These results add 0 further TRs.

The locked 11 plant-payoff relations are unchanged. Raw identity retains
the historical candidate set. Payoff-controlled identity excludes the
current payoff when it is also another relation's plant, because its
residual after controlling itself is undefined.

## Participant-first tests

| metric | ROI | n | mean percentile | median | p (greater) | q, 2 ROIs | q, all 4 |
|---|---|---:|---:|---:|---:|---:|---:|
| `identity_percentile` | `aud_early_chen` | 17 | 0.488 | 0.484 | 0.641 | 0.641 | 0.771 |
| `identity_percentile` | `pmc_chen` | 17 | 0.550 | 0.554 | 0.021 | 0.041 | 0.041 |
| `partial_identity_percentile` | `aud_early_chen` | 17 | 0.472 | 0.475 | 0.771 | 0.771 | 0.771 |
| `partial_identity_percentile` | `pmc_chen` | 17 | 0.574 | 0.558 | 0.012 | 0.023 | 0.041 |

## Order-valid plant-label permutations

Each payoff and each participant's recall coverage remain fixed; only
time-valid plant labels are reassigned.

| metric | ROI | observed | null mean | p | q, 2 ROIs | q, all 4 |
|---|---|---:|---:|---:|---:|---:|
| `identity_percentile` | `aud_early_chen` | 0.488 | 0.503 | 0.659 | 0.659 | 0.738 |
| `identity_percentile` | `pmc_chen` | 0.550 | 0.482 | 0.024 | 0.049 | 0.049 |
| `partial_identity_percentile` | `aud_early_chen` | 0.472 | 0.494 | 0.738 | 0.738 | 0.738 |
| `partial_identity_percentile` | `pmc_chen` | 0.574 | 0.495 | 0.017 | 0.035 | 0.049 |

## Influence diagnostics

- `identity_percentile`: leave-one-subject-out p range 0.0053-0.0367; leave-one-pair-out p range 0.0038-0.1883.
- `partial_identity_percentile`: leave-one-subject-out p range 0.0049-0.0218; leave-one-pair-out p range 0.0028-0.0601.

## Coverage

PMC contributes **132** subject x relation cells from **17** participants. Cells per participant range from **5** to **11** (median 8).

## Empirical same-scene benchmark

The source-aligned payoff same-scene movie-to-recall benchmark is **0.662** across 115 matched, unique subject x payoff-scene cells. The raw plant mean on that deduplicated set is **0.549**, reaching **29.9%** of the lift from chance (.5) to that benchmark. This is an empirical within-scene reference, not a mathematical ceiling.

Across the 10 payoff scenes, same-scene quality and raw plant identity were negatively associated (Spearman rho=-0.622, p=0.055); recall coverage did not significantly track raw plant identity (rho=+0.349, p=0.323). These n=10 diagnostics are descriptive.

Historical +3-TR outputs remain in `results/chen_recall_bold/` and
must not be described as the source-required HRF correction.
