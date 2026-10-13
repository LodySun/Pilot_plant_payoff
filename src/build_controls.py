"""Stage 2 -- build matched control pairs.

Two null families are drawn for every real plant/payoff pair, from the movie
timeline with all real plant/payoff windows (plus a buffer) excluded:

  Family A -- distance-matched. Random moment-pairs separated by the same
    temporal gap as the real pair, within a tolerance. This is the null that
    matters most: without it, any plant/payoff similarity is confounded with
    how far apart in time the two moments are.

  Family B -- distance- AND semantic-matched. The subset of Family A whose
    two moments have a scene-description similarity close to the real pair's.
    This asks whether plant/payoff similarity survives after controlling for
    the moments simply being about similar things.

Both families reuse the real pair's own window durations, so pattern
extraction is identical between real and control moments.

Semantic similarity is computed the same way on both sides -- from the
ENCLOSING SEGMENT's description in the released scene file. Control moments
have no annotator-written description, so using the annotator's text for the
real pair and scene text for the controls would compare two different things.
The annotator-description similarity is recorded alongside, for reference.

Windows never cross the Part1/Part2 boundary: such a window would average
BOLD across two separate runs. A pair may span the boundary; a single moment
may not.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (  # noqa: E402  # noqa: E402
    REPO_ROOT,
    append_data_note,
    ensure_dir,
    load_config,
    load_movie_timeline,
    require_lock_or_exit,
    resolve,
    run_bounds,
    tr_to_seconds,
    window_trs,
)

MAX_ATTEMPTS_PER_CONTROL = 4000


# --------------------------------------------------------------------------
# Timeline bookkeeping
# --------------------------------------------------------------------------

def window_is_valid(lo: int, hi: int, runs: list[tuple[int, int]],
                    excluded: np.ndarray) -> bool:
    """Window must sit inside one run and touch no excluded TR."""
    if lo < 0 or hi > excluded.size or hi <= lo:
        return False
    if not any(lo >= a and hi <= b for a, b in runs):
        return False
    return not excluded[lo:hi].any()


def build_exclusion_mask(pairs: pd.DataFrame, cfg: dict, timeline: dict) -> np.ndarray:
    """TRs control moments may not be drawn from.

    Two sources: every real plant/payoff window (padded by a buffer), and the
    run-opening cartoon. The cartoon is excluded because the identical clip is
    shown at the start of BOTH runs, so a control pair straddling the two
    copies would compare literally the same stimulus to itself and inflate the
    null; it is also not narrative content, and it sits where BOLD is least
    stable after scan onset.
    """
    tr = float(cfg["acquisition"]["tr_seconds"])
    shift = int(cfg["acquisition"]["hrf_shift_trs"])
    w = cfg["windows"]
    buf = int(cfg["controls"]["exclusion_buffer_trs"])
    total = int(timeline["total_trs"])

    mask = np.zeros(total, dtype=bool)
    for _, r in pairs.iterrows():
        for kind in ("plant", "payoff"):
            lo, hi = window_trs(float(r[f"{kind}_start_s"]), float(r[f"{kind}_end_s"]),
                                tr, shift, w["min_trs"], w["max_trs"])
            mask[max(lo - buf, 0):min(hi + buf, total)] = True

    for span in timeline.get("cartoon_spans", []):
        lo = int(np.floor(float(span["start_s"]) / tr))
        hi = int(np.ceil(float(span["end_s"]) / tr)) + shift
        mask[max(lo, 0):min(hi, total)] = True
    return mask


# --------------------------------------------------------------------------
# Scene lookup
# --------------------------------------------------------------------------

def segment_index_for_tr(scenes: pd.DataFrame, timeline: dict) -> np.ndarray:
    """For every TR, the row index of the segment covering it (-1 if none)."""
    total = int(timeline["total_trs"])
    out = np.full(total, -1, dtype=np.int32)
    starts = scenes["start_tr"].to_numpy(dtype="float64")
    ends = scenes["end_tr"].to_numpy(dtype="float64")
    for i in range(len(scenes)):
        s, e = starts[i], ends[i]
        if np.isnan(s) or np.isnan(e):
            continue  # segments 481-482 carry no TR
        lo, hi = int(s), int(e) + 1  # source end_tr is inclusive
        out[max(lo, 0):min(hi, total)] = i
    return out


def _pair_semantic_similarity(seg_a: int, seg_b: int, emb: np.ndarray) -> float:
    if seg_a < 0 or seg_b < 0:
        return np.nan
    return float(np.dot(emb[seg_a], emb[seg_b]))


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def _run_of_tr(tr_index: int, runs: list[tuple[int, int]]) -> int | None:
    for k, (a, b) in enumerate(runs, start=1):
        if a <= tr_index < b:
            return k
    return None


def enumerate_candidates(
    plant_dur: float, payoff_dur: float, gap: float,
    cfg: dict, timeline: dict, excluded: np.ndarray,
    runs: list[tuple[int, int]], crosses_run: int | None = None,
) -> np.ndarray:
    """Every valid control pair for this gap, as an (n, 4) array of TR bounds.

    Enumerated rather than rejection-sampled. Rejection sampling was fine for
    Family A, which needs only 100 draws, but Family B selects a
    feature-matched subset and its survival rate is well under 1%: drawing
    100 candidates and hoping 30 match cannot work. Enumerating the whole
    space (5-8x10^4 pairs for a typical gap) lets Family B choose from all of
    it, and makes "how many exist" an exact answer rather than an estimate.

    Columns: lo1, hi1, lo2, hi2 (half-open TR windows, HRF-shifted).
    """
    tr = float(cfg["acquisition"]["tr_seconds"])
    shift = int(cfg["acquisition"]["hrf_shift_trs"])
    w = cfg["windows"]
    tol = float(cfg["controls"]["gap_tolerance_frac"])
    total = int(timeline["total_trs"])
    boundary = int(timeline["part1_trs"])

    len1 = window_trs(0.0, plant_dur, tr, shift, w["min_trs"], w["max_trs"])
    len1 = len1[1] - len1[0]
    len2 = window_trs(0.0, payoff_dur, tr, shift, w["min_trs"], w["max_trs"])
    len2 = len2[1] - len2[0]

    def valid_onsets(length: int) -> np.ndarray:
        """Boolean over onset TRs: window fits in one run and is unexcluded."""
        ok = np.zeros(total, dtype=bool)
        for a, b in runs:
            hi = b - length
            if hi <= a:
                continue
            ok[a:hi] = True
        # A window is excluded if ANY of its TRs is excluded.
        bad = np.zeros(total, dtype=bool)
        cum = np.concatenate([[0], np.cumsum(excluded.astype(np.int32))])
        starts = np.arange(total)
        ends = np.minimum(starts + length, total)
        bad[:] = (cum[ends] - cum[starts]) > 0
        return ok & ~bad

    ok1 = valid_onsets(len1)
    ok2 = valid_onsets(len2)

    # The realised gap is (lo2 - hi1) * tr = (delta - len1) * tr, where delta
    # is the offset between the two onsets. Note len1, the CLAMPED window
    # length in TRs -- not the annotated duration in seconds, which can differ
    # once the 2-4 TR clamp bites. Round inward so the realised gap never
    # exceeds the tolerance.
    d_lo = int(np.ceil(gap * (1 - tol) / tr)) + len1
    d_hi = int(np.floor(gap * (1 + tol) / tr)) + len1
    d_lo = max(d_lo, len1)  # the two windows must not overlap

    idx1 = np.flatnonzero(ok1)
    out: list[np.ndarray] = []
    for delta in range(d_lo, d_hi + 1):
        lo2 = idx1 + delta
        keep = lo2 < total
        lo1c, lo2c = idx1[keep], lo2[keep]
        keep2 = ok2[lo2c]
        lo1c, lo2c = lo1c[keep2], lo2c[keep2]
        if crosses_run is not None and lo1c.size:
            r1 = (lo1c >= boundary).astype(int)
            r2 = (lo2c >= boundary).astype(int)
            m = (r1 != r2).astype(int) == int(crosses_run)
            lo1c, lo2c = lo1c[m], lo2c[m]
        if lo1c.size:
            out.append(np.column_stack([lo1c, lo1c + len1, lo2c, lo2c + len2]))

    if not out:
        return np.zeros((0, 4), dtype=int)
    return np.unique(np.concatenate(out, axis=0), axis=0)


def sample_family_a(
    rng: np.random.Generator, candidates: np.ndarray, n_target: int,
) -> np.ndarray:
    """A plain distance-matched random subsample of the enumerated space."""
    if candidates.shape[0] <= n_target:
        return candidates
    pick = rng.choice(candidates.shape[0], size=n_target, replace=False)
    return candidates[np.sort(pick)]


def build(cfg: dict, pairs: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    from src.embeddings import scene_embeddings

    timeline = load_movie_timeline(cfg)
    if timeline is None:
        raise FileNotFoundError(
            "data/derived/movie_timeline.json missing -- run `make data` first."
        )
    ccfg = cfg["controls"]
    tr_s = float(cfg["acquisition"]["tr_seconds"])
    shift_trs = int(cfg["acquisition"]["hrf_shift_trs"])
    w = cfg["windows"]

    from src import feature_matching as fm

    scenes, emb = scene_embeddings(cfg, verbose=verbose)
    use_features = fm.available(cfg, scenes)
    feats = fm.prepare(scenes) if use_features else None
    b_rows: list[dict] = []
    seg_of_tr = segment_index_for_tr(scenes, timeline)
    runs = run_bounds(timeline)
    excluded = build_exclusion_mask(pairs, cfg, timeline)

    findings = [
        f"- Movie timeline: **{timeline['total_trs']} TRs** in "
        f"{len(runs)} runs {runs}.",
        f"- Excluded from control sampling: **{int(excluded.sum())} TRs** "
        f"({100 * excluded.mean():.1f}% of the movie) -- every real plant/payoff "
        f"window padded by ±{ccfg['exclusion_buffer_trs']} TR.",
        "- Control windows may not cross the Part1/Part2 boundary (that would "
        "average BOLD across two separate runs); a control *pair* may span it, "
        "exactly as a real pair does.",
        "- A control pair spans the run boundary **if and only if its real pair "
        "does**, so a cross-run real pair is never compared against a within-run "
        "null (or the reverse).",
        f"- The run-opening cartoon is excluded from sampling: "
        + ", ".join(f"{s['start_s']:.0f}-{s['end_s']:.0f}s"
                    for s in timeline.get("cartoon_spans", []))
        + ". The identical clip opens both runs, so a control pair drawn from "
        "the two copies would compare the same stimulus to itself.",
        "",
        "| pair | gap (s) | run span | valid space | Family A | Family B | B rung |",
        "|---|---|---|---|---|---|---|",
    ]

    seed = int(cfg["seed"])
    root = np.random.default_rng(seed)
    seeds = root.spawn(len(pairs)) if hasattr(root, "spawn") else [
        np.random.default_rng(seed + i) for i in range(len(pairs))
    ]

    rows: list[dict] = []
    shortfalls: list[str] = []

    for i, (_, r) in enumerate(pairs.iterrows()):
        pid = str(r["pair_id"])
        rng = seeds[i]
        plant_dur = float(r["plant_end_s"]) - float(r["plant_start_s"])
        payoff_dur = float(r["payoff_end_s"]) - float(r["payoff_start_s"])
        gap = float(r["payoff_start_s"]) - float(r["plant_end_s"])

        # --- the real pair's own enclosing segments ------------------------
        plo, _ = window_trs(float(r["plant_start_s"]), float(r["plant_end_s"]),
                            tr_s, shift_trs, w["min_trs"], w["max_trs"])
        olo, _ = window_trs(float(r["payoff_start_s"]), float(r["payoff_end_s"]),
                            tr_s, shift_trs, w["min_trs"], w["max_trs"])
        seg_p = int(seg_of_tr[plo]) if plo < len(seg_of_tr) else -1
        seg_o = int(seg_of_tr[olo]) if olo < len(seg_of_tr) else -1

        boundary = float(timeline["part1_seconds"])
        crosses = int((float(r["plant_start_s"]) < boundary)
                      != (float(r["payoff_start_s"]) < boundary))

        # The complete valid space for this gap. Family A is a random
        # subsample of it; Family B feature-matches across all of it, because
        # its survival rate is far below 1% and selecting from 100 draws
        # would find nothing.
        space = enumerate_candidates(plant_dur, payoff_dur, gap, cfg, timeline,
                                     excluded, runs, crosses_run=crosses)
        n_space = int(space.shape[0])
        chosen_a = sample_family_a(rng, space, int(ccfg["n_per_pair"]))
        if n_space < int(ccfg["n_per_pair"]):
            shortfalls.append(
                f"{pid}: only {n_space} distinct control pair(s) exist at gap "
                f"{gap:.0f}s (target {ccfg['n_per_pair']}); all were used"
            )

        seg_all_a = seg_of_tr[space[:, 0]] if n_space else np.zeros(0, dtype=int)
        seg_all_b = seg_of_tr[space[:, 2]] if n_space else np.zeros(0, dtype=int)

        # --- the real pair's own semantics and features --------------------
        real_sim = _pair_semantic_similarity(seg_p, seg_o, emb)

        # --- Family B ------------------------------------------------------
        b_sel = np.zeros(0, dtype=int)
        tol_used = float(ccfg["semantic_tolerance"])
        b_detail = ""
        b_rung = 0

        if use_features and n_space:
            res = fm.match(cfg, feats, seg_all_a, seg_all_b, seg_p, seg_o)
            hits = np.flatnonzero(res.mask)
            if hits.size > int(ccfg["n_per_pair"]):
                hits = np.sort(rng.choice(hits, size=int(ccfg["n_per_pair"]),
                                          replace=False))
            b_sel = hits
            tol_used, b_rung = res.arousal_tol, res.rung
            b_detail = (f"arousal ±{res.arousal_tol} SD, valence ±{res.valence_tol}, "
                        f"exact [{', '.join(res.exact_fields) or 'none'}]")
            for note in res.notes:
                shortfalls.append(f"{pid}: Family B {note}")
            b_rows.append({
                "pair_id": pid, "mode": "feature", "rung": res.rung,
                "n_matched": int(res.mask.sum()), "n_used": int(b_sel.size),
                "n_space": n_space,
                "arousal_tol": res.arousal_tol, "valence_tol": res.valence_tol,
                "exact_fields": ";".join(res.exact_fields),
                "plant_features": fm.describe(cfg, feats, seg_p),
                "payoff_features": fm.describe(cfg, feats, seg_o),
            })
        elif n_space:
            sims_all = np.array([
                _pair_semantic_similarity(int(a), int(b), emb)
                for a, b in zip(seg_all_a, seg_all_b)], dtype=float)
            for tol in (float(ccfg["semantic_tolerance"]),
                        float(ccfg["semantic_tolerance_relaxed"])):
                hits = np.flatnonzero(np.abs(sims_all - real_sim) <= tol)
                tol_used = tol
                if hits.size >= int(ccfg["semantic_min_surviving"]):
                    break
            if tol_used != float(ccfg["semantic_tolerance"]):
                shortfalls.append(f"{pid}: Family B relaxed to ±{tol_used} "
                                  f"({hits.size} surviving)")
            if hits.size > int(ccfg["n_per_pair"]):
                hits = np.sort(rng.choice(hits, size=int(ccfg["n_per_pair"]),
                                          replace=False))
            b_sel = hits
            b_detail = f"embedding similarity ±{tol_used}"
            b_rows.append({"pair_id": pid, "mode": "embedding", "rung": 0,
                           "n_matched": int(hits.size), "n_used": int(b_sel.size),
                           "n_space": n_space,
                           "arousal_tol": np.nan, "valence_tol": np.nan,
                           "exact_fields": "", "plant_features": "",
                           "payoff_features": ""})

        # --- emit rows: Family A subsample, plus the Family B selection ----
        emitted: dict[tuple, int] = {}
        def _emit(row4: np.ndarray, in_a: bool, in_b: bool) -> None:
            key = tuple(int(x) for x in row4)
            if key in emitted:
                rows[emitted[key]]["family_a"] |= in_a
                rows[emitted[key]]["family_b"] |= in_b
                return
            lo1, hi1, lo2, hi2 = (int(x) for x in row4)
            sa, sb = int(seg_of_tr[lo1]), int(seg_of_tr[lo2])
            emitted[key] = len(rows)
            rows.append({
                "pair_id": pid, "control_index": len(emitted) - 1,
                "a_start_s": tr_to_seconds(lo1, tr_s, shift_trs),
                "a_end_s": tr_to_seconds(hi1, tr_s, shift_trs),
                "b_start_s": tr_to_seconds(lo2, tr_s, shift_trs),
                "b_end_s": tr_to_seconds(hi2, tr_s, shift_trs),
                "a_tr_lo": lo1, "a_tr_hi": hi1, "b_tr_lo": lo2, "b_tr_hi": hi2,
                "gap_s": tr_to_seconds(lo2, tr_s, shift_trs)
                         - tr_to_seconds(hi1, tr_s, shift_trs),
                "segment_a": sa, "segment_b": sb,
                "semantic_sim": _pair_semantic_similarity(sa, sb, emb),
                "family_a": in_a, "family_b": in_b,
                "real_gap_s": gap, "real_semantic_sim": real_sim,
                "family_b_tolerance": tol_used, "family_b_rung": b_rung,
                "crosses_run": crosses, "n_space": n_space,
            })

        for row4 in chosen_a:
            _emit(row4, True, False)
        for i in b_sel:
            _emit(space[i], False, True)

        n_a = int(chosen_a.shape[0])
        n_b = int(b_sel.size)
        findings.append(
            f"| `{pid}` | {gap:.0f} | {'cross' if crosses else 'within'} | "
            f"{n_space} | {n_a} | {n_b} | {b_rung} |"
        )
        if verbose:
            print(f"  {pid:<18s} gap {gap:7.0f}s  "
                  f"{'cross-run' if crosses else 'within-run':10s}"
                  f"  space={n_space:6d}  A={n_a:4d}  B={n_b:4d}"
                  f"  [{b_detail}, rung {b_rung}]")

    controls = pd.DataFrame(rows)
    b_table = pd.DataFrame(b_rows)
    if not b_table.empty:
        findings += [
            "",
            "Family B, per pair:",
            "",
            "| pair | mode | rung | matched | used | arousal tol | valence tol "
            "| plant features | payoff features |",
            "|---|---|---|---|---|---|---|---|---|",
            *[f"| `{r['pair_id']}` | {r['mode']} | {int(r['rung'])} | "
              f"{int(r['n_matched'])} | {int(r['n_used'])} | "
              f"{r['arousal_tol'] if pd.notna(r['arousal_tol']) else 'n/a'} | "
              f"{r['valence_tol'] if pd.notna(r['valence_tol']) else 'n/a'} | "
              f"{r['plant_features']} | {r['payoff_features']} |"
              for _, r in b_table.iterrows()],
        ]
    if shortfalls:
        findings += ["", "Shortfalls recorded:"] + [f"- {s}" for s in shortfalls]
    findings += [
        "",
        f"- RNG seed: **{seed}** (from `config.yaml`), one independent stream per pair.",
        "- Semantic similarity uses the enclosing SEGMENT description from the "
        "released scene file, for real and control pairs alike, so the two are "
        "measured the same way.",
        "",
        (f"- **Family B matching mode: `feature`** -- matched on the released "
         f"segment labels (exact: {', '.join(cfg['controls']['feature_matching']['exact_fields'])}; "
         f"arousal within ±{cfg['controls']['feature_matching']['arousal_sd_tolerance']} SD; "
         f"valence within ±{cfg['controls']['feature_matching']['valence_tolerance']} "
         "on the 0-4 count of '+' across raters). Valence carries no SD because "
         "each rater codes only '+' or '-'. `Words on Screen` is not matched on: "
         "it is non-empty in 37/1000 segments."
         if use_features else
         "- **Family B matching mode: `embedding`** -- the segment label columns "
         "were unavailable, so Family B fell back to scene-description embedding "
         "similarity."),
    ]
    return controls, findings


# --------------------------------------------------------------------------
# Smoke-test support
# --------------------------------------------------------------------------

def synthetic_pairs(cfg: dict, timeline: dict, n: int = 10) -> pd.DataFrame:
    """Fake annotations for --smoke-test only. Never a scientific result."""
    rng = np.random.default_rng(int(cfg["seed"]))
    duration = float(timeline["movie_duration_seconds"])
    min_gap = float(cfg["annotations"]["min_gap_seconds"])
    rows = []
    for i in range(n):
        plant_start = float(rng.uniform(0, duration * 0.4))
        plant_end = plant_start + float(rng.uniform(3, 6))
        gap = float(rng.uniform(min_gap * 2, duration * 0.35))
        payoff_start = plant_end + gap
        payoff_end = min(payoff_start + float(rng.uniform(3, 6)), duration - 1)
        rows.append({
            "pair_id": f"SMOKE_{i + 1:02d}",
            "plant_start_s": plant_start, "plant_end_s": plant_end,
            "payoff_start_s": payoff_start, "payoff_end_s": payoff_end,
            "modality": ["visual", "auditory", "verbal", "mixed"][i % 4],
            "plant_desc": f"synthetic plant {i}", "payoff_desc": f"synthetic payoff {i}",
            "characters_plant": "A" if i % 2 else "B",
            "characters_payoff": "A" if i % 3 else "C",
            "notes": "SYNTHETIC -- smoke test only",
        })
    return pd.DataFrame(rows)


def load_pairs(cfg: dict, smoke: bool) -> pd.DataFrame | None:
    """Real annotations, or synthetic ones in smoke mode."""
    timeline = load_movie_timeline(cfg)
    if smoke:
        if timeline is None:
            raise FileNotFoundError("smoke test needs data/derived/movie_timeline.json "
                                    "-- run `make data` first.")
        return synthetic_pairs(cfg, timeline)

    from src.validate_annotations import apply_time_frame, parse_annotations, Report
    path = REPO_ROOT / cfg["annotations"]["path"]
    if not path.exists():
        return None
    rep = Report()
    df = parse_annotations(path, rep)
    if df is None or rep.errors:
        for e in rep.errors:
            print(f"  annotation error: {e}", file=sys.stderr)
        raise SystemExit("annotations are invalid; run `make validate`.")
    return apply_time_frame(df, cfg, rep)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 2: build matched control pairs.")
    p.add_argument("--config", default=None)
    p.add_argument("--smoke-test", action="store_true",
                   help="use synthetic annotations; write to results/smoke/")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    print("=" * 72)
    print("Stage 2 -- control pairs" + ("  [SMOKE TEST]" if args.smoke_test else ""))
    print("=" * 72)

    require_lock_or_exit(cfg, smoke=args.smoke_test)

    pairs = load_pairs(cfg, args.smoke_test)
    if pairs is None:
        print(
            "\nNOT RUN: annotations/pairs.csv does not exist.\n"
            "  Annotation is human work. Copy annotations/pairs_template.csv,\n"
            "  fill it in, then:\n"
            "    make validate\n"
            "    make lock-annotations\n"
            "    make controls\n",
            file=sys.stderr,
        )
        return 4

    controls, findings = build(cfg, pairs, verbose=True)

    out_dir = ensure_dir(resolve(cfg, "derived"))
    name = "control_pairs_smoke.parquet" if args.smoke_test else "control_pairs.parquet"
    out = out_dir / name
    controls.to_parquet(out, index=False)

    n_a = int(controls["family_a"].sum())
    n_b = int(controls["family_b"].sum())
    print(f"\n  Family A: {n_a} control pairs")
    print(f"  Family B: {n_b} control pairs")
    print(f"  -> {out.relative_to(REPO_ROOT)}")

    if not args.smoke_test:
        append_data_note(cfg, "Control pairs (Stage 2)", findings)
        print("  Recorded in docs/data_notes.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
