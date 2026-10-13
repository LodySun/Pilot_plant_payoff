# Frozen plan: Chen within-participant recall-BOLD analysis

This plan was written before inspecting any plant/payoff statistic computed
from the Chen spoken-recall BOLD data. It reuses only the locked, human-written
relations in `annotations/pairs.csv`; it does not generate, edit, or infer a
relation or its description.

## Data and time alignment

- Use the 17 Chen release subjects (`s1`–`s17`) with both movie and free-recall
  BOLD in the two distributed ROIs: posterior medial cortex (`pmc_chen`) and
  early auditory cortex (`aud_early_chen`). Never mix these IDs with the GIN
  fMRIPrep or Zadbood/OpenNeuro subject IDs.
- Read each subject's released `freerecall` intervals and `freerecallnum` scene
  labels from `sherlock_allsubs_events.mat`. These directly link recall BOLD to
  the original 50 Chen scenes; transcript-number inference is unnecessary.
- Convert released one-based inclusive intervals to zero-based half-open
  intervals. Pool non-contiguous repetitions of the same recalled scene.
- Average the full Chen scene after the fixed +3 TR (4.5 s) HRF shift, clipped
  within its own movie or recall run. A zero-shift version is a labelled timing
  sensitivity analysis.

## Frozen primary test

For every subject and locked pair whose payoff scene was recalled:

1. Correlate the subject's movie pattern for the correct plant scene with the
   subject's recall pattern for the payoff scene.
2. Rank that correlation against correlations between the same recall-payoff
   pattern and every other unique locked plant scene from the movie. Ties count
   half; chance identity percentile is 0.5.
3. Average available pair percentiles within subject. Subjects need at least
   three payoff pairs. Test the subject scores in `pmc_chen` against 0.5 with a
   one-sided Wilcoxon signed-rank test (`greater`, alpha .05).

This is one pre-specified ROI/metric and receives no multiplicity correction.
It asks whether a recalled payoff context contains relatively specific neural
geometry from its earlier plant context. It does not by itself prove causal
binding.

## Validity and secondary analyses

- **Alignment validity:** reproduce ordinary same-scene reinstatement by asking
  whether each recalled scene is closer to its matching movie scene than to
  other recalled movie scenes.
- **Payoff-controlled partial RSA:** residualize both each plant movie pattern
  and the recall-payoff pattern with respect to the matching payoff movie
  pattern before computing identity. This asks for plant information beyond
  ordinary reinstatement of the payoff's own scene.
- **Recall-to-recall identity:** when both endpoints were recalled, compare the
  recalled payoff pattern with its correct recalled plant pattern versus other
  recalled locked plant scenes.
- Repeat the primary metric in early auditory cortex as a sensory/speech
  control and at zero HRF shift as a timing sensitivity.
- Candidate-level tests are two-sided and FDR-corrected across both ROIs and all
  11 locked pairs. They remain exploratory.
- As a stricter, explicitly secondary null, permute the 11 existing plant labels
  across payoff labels while retaining only assignments in which the assigned
  plant scene precedes the payoff scene. Preserve repeated scene IDs and each
  subject's observed recall coverage. The number of accepted permutations comes
  from `config.yaml` and the RNG seed is stable.

## Interpretation guardrails

- Recall scenes are coarse, variable-duration spoken descriptions. A positive
  result is scene-level reinstatement, not moment-resolved proof that a plant
  caused a payoff response.
- Speaking, acoustic feedback, temporal compression, and ordinary semantic
  overlap can affect recall patterns. The auditory ROI, same-scene validity,
  partial RSA, and label permutation diagnose but cannot eliminate every such
  alternative.
- The distributed Chen package contains only PMC and early auditory ROI data.
  Claims about angular gyrus, hippocampus, or mPFC require separately acquired
  whole-brain recall data and are not inferred here.
