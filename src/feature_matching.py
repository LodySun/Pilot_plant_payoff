"""Family-B matching on the released segment labels.

The spec's preferred Family B matches control moments to real moments on
stimulus features rather than on text-embedding similarity. Two properties
of the released labels shape how this is done, both measured in Stage 0:

  * **Arousal** is a 1-5 rating from four raters whose baselines differ
    sharply, so the composite is the mean of per-rater z-scores, then
    re-standardised across segments so that a "±0.5 SD" tolerance means what
    it says.
  * **Valence is not numeric.** Each rater codes only `+` or `-`, so there is
    no standard deviation to take a tolerance on. It is summarised as the
    count of `+` across raters (0-4) and matched within an integer tolerance.

`Words on Screen` is deliberately not matched on: it is non-empty in only
37 of 1000 segments, too sparse to constrain anything.

When the tolerances cannot yield enough controls, the ladder relaxes in a
fixed, recorded order: numeric tolerances first, then exact constraints are
dropped one at a time from least to most theoretically important.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUIRED_COLUMNS = [
    "arousal_z", "valence_pos", "anyone_speaking", "indoor_outdoor",
    "sherlock_onscreen", "john_onscreen",
]


def available(cfg: dict, scenes: pd.DataFrame) -> bool:
    """True when the label file supports feature matching."""
    if not cfg["controls"].get("feature_matching", {}).get("enabled", False):
        return False
    return all(c in scenes.columns for c in REQUIRED_COLUMNS)


def prepare(scenes: pd.DataFrame) -> pd.DataFrame:
    """Segment-indexed matching table, with arousal on a true SD scale."""
    feats = scenes[REQUIRED_COLUMNS].copy()
    a = pd.to_numeric(feats["arousal_z"], errors="coerce")
    sd = float(a.std(ddof=0))
    feats["arousal_sd"] = (a - a.mean()) / sd if sd > 0 else 0.0
    feats["valence_pos"] = pd.to_numeric(feats["valence_pos"], errors="coerce")
    feats["indoor_outdoor"] = feats["indoor_outdoor"].astype(str).str.strip().str.casefold()
    for c in ("anyone_speaking", "sherlock_onscreen", "john_onscreen"):
        feats[c] = feats[c].astype(bool)
    return feats


@dataclass
class MatchResult:
    mask: np.ndarray
    arousal_tol: float
    valence_tol: int
    exact_fields: list[str]
    rung: int
    n: int
    notes: list[str] = field(default_factory=list)

    @property
    def relaxed(self) -> bool:
        return self.rung > 0


def _one_pass(feats: pd.DataFrame, seg_a: np.ndarray, seg_b: np.ndarray,
              real_a: int, real_b: int, exact_fields: list[str],
              arousal_tol: float, valence_tol: int) -> np.ndarray:
    """Which candidates match, at the given tolerances."""
    ok = (seg_a >= 0) & (seg_b >= 0)
    if not ok.any():
        return ok

    for side, seg, real in (("a", seg_a, real_a), ("b", seg_b, real_b)):
        if real < 0:
            continue
        idx = np.clip(seg, 0, len(feats) - 1)

        for col in exact_fields:
            want = feats[col].iloc[real]
            ok &= feats[col].to_numpy()[idx] == want

        want_a = feats["arousal_sd"].iloc[real]
        got_a = feats["arousal_sd"].to_numpy()[idx]
        ok &= np.abs(got_a - want_a) <= arousal_tol

        want_v = feats["valence_pos"].iloc[real]
        got_v = feats["valence_pos"].to_numpy()[idx]
        if np.isfinite(want_v):
            ok &= np.abs(got_v - want_v) <= valence_tol
    return ok


def match(cfg: dict, feats: pd.DataFrame, seg_a: np.ndarray, seg_b: np.ndarray,
          real_a: int, real_b: int) -> MatchResult:
    """Feature-match candidates to a real pair, relaxing until enough survive."""
    fc = cfg["controls"]["feature_matching"]
    min_n = int(fc["min_surviving"])
    exact = list(fc["exact_fields"])
    a_steps = [float(x) for x in fc["relax_arousal_steps"]]
    v_steps = [int(x) for x in fc["relax_valence_steps"]]
    drop_order = list(fc["drop_exact_order"])

    # The ladder must be MONOTONE: every rung strictly looser than the last.
    # Relaxing valence to its maximum and then re-tightening it while arousal
    # widens would make "rung 3" more restrictive than "rung 2", so the two
    # numeric tolerances are widened in lockstep, the shorter list holding at
    # its final value.
    n_rungs = max(len(a_steps), len(v_steps))

    def _step(steps: list, i: int):
        return steps[min(i, len(steps) - 1)]

    ladder: list[tuple[float, int, list[str]]] = [
        (float(_step(a_steps, i)), int(_step(v_steps, i)), list(exact))
        for i in range(n_rungs)
    ]
    # Only then start dropping exact constraints, at the loosest numeric setting.
    dropped: list[str] = []
    for fieldname in drop_order:
        dropped.append(fieldname)
        remaining = [c for c in exact if c not in dropped]
        ladder.append((float(a_steps[-1]), int(v_steps[-1]), remaining))

    notes: list[str] = []
    best: MatchResult | None = None
    for rung, (a_tol, v_tol, ex) in enumerate(ladder):
        mask = _one_pass(feats, seg_a, seg_b, real_a, real_b, ex, a_tol, v_tol)
        n = int(mask.sum())
        if best is None or n > best.n:
            best = MatchResult(mask, a_tol, v_tol, ex, rung, n, notes)
        if n >= min_n:
            if rung > 0:
                notes.append(
                    f"relaxed to rung {rung} (arousal ±{a_tol} SD, valence ±{v_tol}"
                    + (f", dropped {', '.join(c for c in exact if c not in ex)}"
                       if len(ex) < len(exact) else "")
                    + f") to reach {n} controls"
                )
            return MatchResult(mask, a_tol, v_tol, ex, rung, n, notes)

    assert best is not None
    best.notes = notes + [
        f"exhausted the relaxation ladder; only {best.n} control(s) matched "
        f"(target {min_n})"
    ]
    return best


def describe(cfg: dict, feats: pd.DataFrame, seg: int) -> str:
    """Compact human-readable feature signature for one segment."""
    if seg < 0:
        return "n/a"
    r = feats.iloc[seg]
    return (f"arousal {r['arousal_sd']:+.2f}SD, valence {int(r['valence_pos'])}/4, "
            f"{'speech' if r['anyone_speaking'] else 'silent'}, "
            f"{r['indoor_outdoor']}, "
            f"S={'Y' if r['sherlock_onscreen'] else 'N'}"
            f"/J={'Y' if r['john_onscreen'] else 'N'}")
