# Frozen plan: does spoken plant content explain recall-BOLD identity?

This interpretive follow-up was frozen before joining any Chen recall-BOLD
plant-identity score to transcript content. It never creates, edits, or infers a
plant/payoff relation, keyword, or description. Relations and keywords come only
from the locked human annotation file.

## Subject and time alignment

- Use the 17 Chen/Princeton DataSpace subjects only. The published timestamp
  data paper explicitly maps `NN01`–`NN17` to Princeton DataSpace fMRI subjects
  `01`–`17`, respectively (doi:10.1016/j.dib.2023.109490). This is distinct from
  the separate 16-subject recall-onset/OpenNeuro mapping used elsewhere.
- Slice each `NNxx` transcript with that subject's own one-based inclusive
  `freerecall` intervals in `sherlock_allsubs_events.mat`. Pool non-contiguous
  intervals carrying the same 50-scene label, exactly as in the recall-BOLD
  analysis.
- Join only the 132 subject × pair cells already entering the frozen +3 TR
  recall-BOLD analysis. No missing payoff is imputed.

## Content measures

1. **Frozen primary content measure:** whether any locked, human-written
   `plant_keywords` item appears in the subject's payoff-scene speech. Matching
   uses the existing conservative token/stem rules; keywords are not changed in
   response to the data.
2. **Semantic sensitivity:** embedding similarity between payoff-scene speech
   and the correct locked plant's released movie context, ranked against the
   other ten locked plant contexts. This is continuous but model-dependent.
3. **Recall-binding sensitivity:** when the plant was also recalled, reuse the
   within-subject semantic binding percentile between recalled plant and payoff
   scenes.

Report own-plant-scene keyword and embedding validity diagnostics. The
embedding model and context radius remain those already specified in
`config.yaml`.

## Association model

- Outcome: +3 TR plant-identity percentile from recall BOLD.
- Primary cell: PMC identity versus the binary keyword hit. Secondary cells are
  PMC versus the two continuous measures and the same three associations in
  early auditory cortex.
- For each association, residualise neural and content values for subject fixed
  effects, pair fixed effects, and log payoff-scene word count; report the
  residual partial correlation. For the binary primary measure, also report the
  adjusted hit-minus-no-hit neural difference.
- Inference uses whole-subject cluster sign flips of the residual cross-products,
  preserving every subject's set of recalled pairs and within-subject dependence.
  Report a subject-cluster bootstrap 95% interval. Repetition counts and seed
  come from `config.yaml`.
- The frozen primary PMC/keyword test is two-sided at alpha .05. Secondary tests
  are two-sided and BH-FDR corrected together. Report the number of pairs with
  within-pair variation for each content measure.

## Interpretation

- A positive association would be consistent with PMC identity tracking plant
  material actually spoken during payoff recall. It would weaken a claim of
  covert contextual reconstruction, but cannot establish that speech content
  caused the neural pattern.
- A null association only makes the stronger context-reconstruction reading
  more plausible if its interval excludes a substantively large association.
  Sparse keywords, noisy embeddings, coarse scenes, and only 17 participant
  clusters limit that conclusion.
- In either case this is an exploratory explanatory analysis, not a new test of
  whether the recall-BOLD effect exists.
