# When Payoff Returns — poster analyses

Code, annotations and result tables behind the four figures of the poster
*When Payoff Returns: Reinstatement of Earlier Narrative Context During Recall*
(Jianlun Sun, University of Helsinki). Data: *Sherlock* (Chen et al., 2017),
N = 17, movie viewing and free spoken recall, fMRI.

The question: a **plant** (an early story detail) and its **payoff** (the later
moment that gives it meaning) are linked across the narrative. Is that link
visible in brain activity patterns while watching, and when the payoff is
recalled?

This repository contains only the analyses the poster uses. It was extracted
from a larger pilot pipeline; exploratory analyses that are not on the poster
are not included.

## Figures and the analyses behind them

| Figure | What it shows | Command | Result tables |
|---|---|---|---|
| 1 | The 11 locked plant–payoff pairs on the movie timeline | `make figures` | `annotations/pairs.csv`, `data/derived/scenes.csv` |
| 2 | Viewing: locked pairs vs 10,000 matched pseudo-narratives (PMC) | `make viewing-null` | `results/viewing_pseudo_null/` |
| 3 | Recall: correct-plant identity in PMC and early auditory cortex, with the order-valid null and the reliability ceiling | `make recall`, `make ceiling` | `results/chen_timing_corrected/`, `results/identity_ceiling/` |
| 4A | Correct relation: raw and payoff-controlled identity vs order-valid null | `make recall` | `results/chen_timing_corrected/order_valid_permutation.csv` |
| 4B | Identity time course around payoff-recall onset (exploratory) | `make timecourse` | `results/chen_recall_timecourse/timing_corrected/` |
| 4C | Association of identity with what participants said | `make content` | `results/recall_content/` |

`make figures` redraws all four as true-size vector PDFs in `figures/`
(sizes in pt are set in `FRAMES` at the top of `src/make_poster_figures.py`;
place them at 100% so ticks print at 22 pt).

### Key numbers

| | value |
|---|---|
| Viewing, locked pairs vs pseudo-narratives (PMC) | 0.494 vs mean 0.500; descriptive p = 0.568 (exact-gap sensitivity 0.534, p = 0.148) |
| Recall identity, PMC | mean 0.550, 95% bootstrap CI 0.507–0.596; 12/17 above 0.50 |
| Order-valid null, PMC | mean 0.482 (SD 0.035); permutation p = 0.024, **q = 0.049** (FDR across ROIs) |
| Payoff-controlled identity, PMC | 0.574; q = 0.035 |
| Early auditory cortex | 0.488; q = 0.659 |
| Reliability ceiling, PMC | 0.599–0.650 (leave-one-participant-out to all-participant bound) |
| Leave-one-subject-out / leave-one-pair-out, PMC | p = 0.005–0.037 / p = 0.004–0.188 (5 of 11 ≥ 0.05) |
| Time course, PMC peak | +4.5 s; max-statistic across lags p = 0.020, q = 0.079 across four curves |
| Spoken keyword vs identity, PMC | partial r = −0.239, p = 0.005 |

## Reading the results correctly

- **Timing is source-aligned.** The Chen compact-release MAT files already
  contain the authors' 3-TR haemodynamic shift, so no further shift is added
  (`docs/chen_timing_correction_plan.md`).
- **The right baseline for recall identity is the order-valid null, not 0.50.**
  True plants always precede their payoff; earlier scenes score lower on
  average (0.486 in PMC) than the payoff's own scene or later scenes, so the
  relabeling null that keeps every plant before its payoff centres at 0.482.
- **Reliability ceiling** (`src/compute_identity_ceiling.py`): a Nili-style
  noise ceiling for this exact retrieval problem — same ROI, windows, pairs,
  candidates and percentile. For each participant and pair, the earlier
  candidate that the *other* participants rank highest is scored in the
  held-out participant (lower bound) or with everyone included (upper bound).
  In early auditory cortex the lower bound (0.474) is below the null (0.503):
  candidate rankings are not reproducible there, so a null result in that
  region is not evidence that the effect is specific to PMC.
  `results/chen_timing_corrected/empirical_ceiling.csv` (0.662) is a different
  quantity — how well a recalled payoff identifies its *own* movie scene — and
  is not this ceiling.
- **Figure 2 is descriptive.** The pseudo-narrative null re-presents the
  matched-control comparison (Family A: same temporal gap; Family-A rows only);
  it is not an independent test.
- **Figure 4B is exploratory.** The peak survives correction across lags
  within the curve but not FDR across the four curves (q = 0.079).
- **Figure 4C, spoken keyword.** The keyword varies across participants in only
  3 pairs, 2 of which name payoff details; saying it predicted *lower*
  identity. Semantic plant content and recall binding showed no association.
- **One annotator, one episode, 11 pairs.** The effect depends on individual
  pairs (leave-one-pair-out above).

## Data

Not committed (third-party; fetched or downloaded by hand):

| Input | Source | Where it goes |
|---|---|---|
| Chen et al. ROI release, 17 subjects × PMC / early auditory | Zenodo record 2598755, `Sherlock_processed.zip` (public) — `make data` | `data/raw/Sherlock_processed/` |
| Segment / scene labels `Sherlock_Segments_1000_NN_2017.xlsx` | GIN `ljchang/Sherlock` (public) — `make data` | `data/annotations_source/` |
| Word-timestamped recall transcripts NN01–NN17 | Zenodo 10.5281/zenodo.8208709 (manual; Figure 4C only) | `data/annotations_source/transcripts/` |

Committed: the locked annotations (`annotations/pairs.csv` and its lock),
the small derived tables the figures read (`data/derived/scenes.csv`,
`movie_timeline.json`, `invalid_trs.json`), and every result table above, so
`make figures` works without the raw data. The per-scene recall transcript
table is not committed.

**Annotation lock.** `annotations/pairs.csv` was annotated by hand and frozen
before any neural data were analysed; `annotations/pairs.csv.lock` holds its
SHA-256. Every analysis that touches real-pair neural data refuses to run if
the file no longer matches its lock.

## Reproduce

```bash
make setup      # .venv + requirements.txt (Python 3.12)
make data       # Chen release + segment labels; transcripts by hand (see above)
make all        # derived inputs, all analyses, figures (~2 minutes)
```

The sentence-embedding model (`sentence-transformers/all-MiniLM-L6-v2`) is
downloaded on first use.

### Verification

Run from scratch in this repository, every analysis reproduced the original
pipeline's outputs: all recall, time-course, pseudo-null, null-draw and
ceiling tables are byte-identical (except that numbers in one generated
summary now carry a leading zero), and the four figures matched pixel for
pixel before numbers on them were given leading zeros.
The spoken-content tables differ only in raw sentence-embedding cosine
similarities (≤ 4 × 10⁻⁷, float32 rounding); all percentiles, keyword hits and
test statistics agree to six decimals.

## Layout

```
annotations/   locked plant–payoff pairs (+ lock, template)
config.yaml    timing, windows, controls, permutations, paths
data/derived/  scene table, movie timeline, invalid-TR scan
docs/          frozen analysis plans (written for the original pipeline, so
               their command names refer to it); data notes from the pipeline
figures/       poster Figures 1–4 (true-size PDF + preview PNG)
results/       result tables for every figure
src/           analysis code (entry points listed in the Makefile)
```

## References

Chen J, et al. Shared memories reveal shared structure in neural activity
across individuals. *Nat Neurosci.* 2017;20:115–125.

Baldassano C, et al. Discovering event structure in continuous narrative
perception and memory. *Neuron.* 2017;95:709–721.

Nili H, et al. A toolbox for representational similarity analysis.
*PLoS Comput Biol.* 2014;10:e1003553.
