# Does spoken plant content explain Chen recall-BOLD identity?

> **Exploratory explanatory analysis.** It does not alter the locked
> relations or keywords, and association cannot establish causal direction.

## Frozen primary content test

PMC plant-identity percentile was related to whether the subject actually
said a locked plant keyword while recalling that payoff scene. Subject and
pair fixed effects and log word count were removed; inference sign-flipped
whole-subject residual cross-products.

| cells | subjects | varying pairs | keyword hit rate | partial r (95% cluster CI) | p | adjusted hit−no-hit |
|---:|---:|---:|---:|---:|---:|---:|
| 132 | 17 | 3 | 23.5% | -0.239 (-0.411, -0.093) | 0.005 | -0.256 |

The last column is the adjusted difference in neural identity percentile
for keyword-hit versus no-hit cells.

## Continuous sensitivities in PMC

| content measure | cells | partial r (95% cluster CI) | p | secondary q |
|---|---:|---:|---:|---:|
| payoff speech → correct plant movie-context percentile | 132 | +0.033 (-0.208, +0.243) | 0.779 | 0.940 |
| recalled plant ↔ recalled payoff binding percentile | 113 | +0.094 (-0.099, +0.252) | 0.339 | 0.847 |

## All pre-specified associations

| ROI | content measure | cells | partial r | p | secondary q |
|---|---|---:|---:|---:|---:|
| `pmc_chen` | `keyword_hit` | 132 | -0.239 | 0.005 | — |
| `pmc_chen` | `semantic_percentile` | 132 | +0.033 | 0.779 | 0.940 |
| `pmc_chen` | `binding_percentile` | 113 | +0.094 | 0.339 | 0.847 |
| `aud_early_chen` | `keyword_hit` | 132 | -0.052 | 0.636 | 0.940 |
| `aud_early_chen` | `semantic_percentile` | 132 | -0.112 | 0.155 | 0.773 |
| `aud_early_chen` | `binding_percentile` | 113 | -0.010 | 0.940 | 0.940 |

## Interpretation

None of the three content measures showed a reliable positive PMC
association. The explicit keyword and plant-context embedding estimates
were slightly negative; recalled-scene binding was near zero. Therefore
the observed PMC plant identity is not explained by a tendency to become
stronger when plant content is overtly present in the payoff speech.

This supports the claim that the neural result is not *merely* current
spoken content, but it does not prove covert context reconstruction. In
particular, the semantic-content interval still permits r up to +0.243
and the recall-binding interval permits r up to +0.252.

## Measurement validity and limits

- A plant scene was recalled in 80.7% of subject × pair cells; conditional on that scene being recalled, its locked keyword occurred in 26.5%.
- The correct plant-context embedding had mean own-plant-scene validity percentile 0.814 (median 0.900).
- A null association does not prove covert reconstruction: the interval,
  sparse keyword variation, coarse scene labels, and 17 clusters determine
  how strongly that interpretation can be defended.
- A positive association is compatible with currently spoken content
  contributing to PMC identity, but cannot prove that speech caused it.
- NN01–NN17 map directly to Princeton DataSpace subjects 01–17 according
  to the published timestamp data paper (doi:10.1016/j.dib.2023.109490).
  The unrelated 16-subject mapping is not used.

Full outputs: `content_cells.csv`, `association_tests.csv`,
`transcript_scene_cells.csv`, `semantic_validity_cells.csv`,
`recall_binding_cells.csv`, and `meta.json`.
