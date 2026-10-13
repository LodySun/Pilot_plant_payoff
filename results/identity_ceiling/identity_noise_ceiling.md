# Reliability ceiling for the recall plant-identity percentile

Nili-style noise ceiling for this exact retrieval problem: same ROI, same
source-aligned windows, same 11 pairs, same 9 plant candidates, same percentile
and subject aggregation. Lower bound = leave-one-subject-out pick of the most
consistent candidate; upper bound = pick using all subjects.

| ROI | metric | observed | order-valid null | ceiling (order-valid) lower–upper | ceiling (all 9) lower–upper | observed as % of null→ceiling(lower) |
|---|---|---:|---:|---:|---:|---:|
| aud_early_chen | identity_percentile | 0.488 | 0.503 ± 0.034 | 0.474–0.641 | 0.467–0.648 | n/a (ceiling ≤ null) |
| aud_early_chen | partial_identity_percentile | 0.472 | 0.494 ± 0.034 | 0.463–0.624 | 0.451–0.667 | n/a (ceiling ≤ null) |
| pmc_chen | identity_percentile | 0.550 | 0.482 ± 0.035 | 0.599–0.650 | 0.633–0.717 | 59% |
| pmc_chen | partial_identity_percentile | 0.574 | 0.495 ± 0.037 | 0.608–0.665 | 0.563–0.691 | 70% |

Chance that a random order-valid pick is the annotated plant: 17% (mean over pairs); all 9: 11%.

For comparison, the same-scene benchmark (a different retrieval problem) is 0.662.

Share of cells whose consistency pick is the annotated plant:

- aud_early_chen / identity_percentile: order-valid LOSO 21%, all-9 LOSO 10%
- aud_early_chen / partial_identity_percentile: order-valid LOSO 21%, all-9 LOSO 6%
- pmc_chen / identity_percentile: order-valid LOSO 44%, all-9 LOSO 24%
- pmc_chen / partial_identity_percentile: order-valid LOSO 43%, all-9 LOSO 26%
