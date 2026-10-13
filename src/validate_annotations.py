"""Stage 1 -- validate a human-provided plant/payoff annotation file.

This script never writes, edits, or infers annotations. It only reads the
file the annotator produced and reports whether it satisfies the pilot's
design constraints.

Exit codes: 0 = valid (warnings allowed), 1 = invalid, 2 = file problem.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (  # noqa: E402
    REPO_ROOT,
    hhmmss_to_seconds,
    load_config,
    load_movie_timeline,
    resolve,
    seconds_to_hhmmss,
    window_trs,
)

REQUIRED_COLUMNS = [
    "pair_id", "plant_start_s", "plant_end_s", "payoff_start_s", "payoff_end_s",
    "characters_plant", "characters_payoff",
]
# Modality may be given as ONE column describing the pair, or as TWO columns
# describing each moment. Two is more informative -- it makes "cross-modal"
# an observation rather than an inference -- so both are accepted.
MODALITY_COLUMNS = ["modality", "modality_plant", "modality_payoff"]
# Descriptions are optional: they feed the embedding fallback for Family B and
# the wrong-time-frame check, neither of which is required when the segment
# label file is present. crosses_run is computed here, so a file may omit it.
OPTIONAL_COLUMNS = [
    "crosses_run", "plant_keywords", "plant_part", "payoff_part", "notes",
    "plant_desc", "payoff_desc",
    "plant_anchor_type", "payoff_anchor_type",
]
VALID_MODALITIES = {"visual", "auditory", "verbal", "mixed"}
TIME_COLUMNS = ["plant_start_s", "plant_end_s", "payoff_start_s", "payoff_end_s"]


class Report:
    """Collects errors and warnings so all problems surface in one pass."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    @property
    def ok(self) -> bool:
        return not self.errors


def _parse_time(value, where: str, rep: Report) -> float:
    """Accept seconds or hh:mm:ss; record a parse failure as NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        rep.error(f"{where}: missing time")
        return float("nan")
    s = str(value).strip()
    if not s:
        rep.error(f"{where}: missing time")
        return float("nan")
    try:
        return hhmmss_to_seconds(s) if ":" in s else float(s)
    except ValueError:
        rep.error(f"{where}: unparseable time {value!r}")
        return float("nan")


def parse_annotations(path: Path, rep: Report) -> pd.DataFrame | None:
    """Read the CSV, check its shape, and normalise times to seconds."""
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    except Exception as exc:  # noqa: BLE001
        rep.error(f"could not read {path}: {exc}")
        return None

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        rep.error(f"missing required column(s): {', '.join(missing)}")
        return None
    if not any(c in df.columns for c in MODALITY_COLUMNS):
        rep.error("no modality column: provide `modality`, or "
                  "`modality_plant` and `modality_payoff`")
        return None
    known = set(REQUIRED_COLUMNS) | set(OPTIONAL_COLUMNS) | set(MODALITY_COLUMNS)
    extra = [c for c in df.columns if c not in known]
    if extra:
        rep.note(f"extra column(s) kept but unused: {', '.join(extra)}")

    if len(df) == 0:
        rep.error("annotation file has no rows")
        return None

    for col in TIME_COLUMNS:
        df[col] = [
            _parse_time(v, f"row {i + 2} ({df.at[i, 'pair_id']}) {col}", rep)
            for i, v in enumerate(df[col])
        ]
    return df


def apply_time_frame(df: pd.DataFrame, cfg: dict, rep: Report) -> pd.DataFrame:
    """Convert annotated times to the pipeline's continuous clock.

    Under `cumulative` (the default) times are already on that clock. Under
    `part_local` the file must carry `plant_part`/`payoff_part` columns and
    Part 2 times are shifted by the length of Part 1.
    """
    frame = str(cfg["annotations"].get("time_frame", "cumulative")).lower()
    if frame == "cumulative":
        rep.note("time frame: `cumulative` -- times are seconds from the start of Part 1.")
        return df

    if frame == "episode":
        # Times as read off the episode itself. The movie clock runs ahead
        # because each run opens with a cartoon that is not episode content.
        timeline = load_movie_timeline(cfg)
        if timeline is None or not timeline.get("episode_offset_by_part"):
            rep.error("time_frame is 'episode' but data/derived/movie_timeline.json "
                      "has no episode offsets. Run `make data` first.")
            return df
        offs = timeline["episode_offset_by_part"]
        split = float(timeline["episode_split_s"])
        dur = float(timeline["episode_duration_s"])
        rep.note(f"time frame: `episode` -- episode timecodes, converted to movie "
                 f"time by +{offs['1']:.0f}s before episode {split:.0f}s "
                 f"and +{offs['2']:.0f}s after.")
        for kind in ("plant", "payoff"):
            for suffix in ("start_s", "end_s"):
                col = f"{kind}_{suffix}"
                vals = df[col].to_numpy(dtype=float)
                over = np.where(vals > dur)[0]
                for i in over:
                    rep.error(f"row {i + 2} ({df.at[i, 'pair_id']}) {col}: "
                              f"{vals[i]:.1f}s is past the end of the stimulus "
                              f"(the episode ends at {dur:.1f}s = "
                              f"{seconds_to_hhmmss(dur)})")
                df[col] = np.where(vals < split,
                                   vals + float(offs["1"]), vals + float(offs["2"]))
        return df

    if frame != "part_local":
        rep.error(f"config annotations.time_frame is {frame!r}; "
                  "expected 'cumulative', 'episode', or 'part_local'")
        return df

    timeline = load_movie_timeline(cfg)
    if timeline is None:
        rep.error("time_frame is 'part_local' but data/derived/movie_timeline.json "
                  "is absent, so Part 1's length is unknown. Run `make data` first.")
        return df
    offset = float(timeline["part1_seconds"])
    rep.note(f"time frame: `part_local` -- Part 2 times shifted by +{offset:.1f}s.")

    for kind in ("plant", "payoff"):
        col = f"{kind}_part"
        if col not in df.columns:
            rep.error(f"time_frame is 'part_local' but column {col!r} is missing")
            continue
        parts = pd.to_numeric(df[col], errors="coerce")
        bad = parts[~parts.isin([1, 2])]
        if len(bad):
            rep.error(f"{col} must be 1 or 2; got {sorted(set(bad.dropna().tolist()))}")
        shift = np.where(parts.to_numpy() == 2, offset, 0.0)
        for suffix in ("start_s", "end_s"):
            df[f"{kind}_{suffix}"] = df[f"{kind}_{suffix}"] + shift
    return df


def validate(df: pd.DataFrame | None, cfg: dict, rep: Report) -> pd.DataFrame:
    """Apply every design constraint; return a per-pair summary table."""
    if df is None or df.empty:
        return pd.DataFrame()
    acfg = cfg["annotations"]
    min_gap = float(acfg["min_gap_seconds"])
    min_win = float(acfg["min_window_seconds"])
    tr_s = float(cfg["acquisition"]["tr_seconds"])
    hrf_shift = int(cfg["acquisition"]["hrf_shift_trs"])
    wcfg = cfg["windows"]

    # --- movie duration: only checked if Stage 0 has recorded it -------------
    timeline = load_movie_timeline(cfg)
    duration = None
    if timeline and timeline.get("movie_duration_seconds"):
        duration = float(timeline["movie_duration_seconds"])
        rep.note(f"movie duration {duration:.1f}s "
                 f"({seconds_to_hhmmss(duration)}) read from {timeline.get('source', 'data')}")
    else:
        rep.warn(
            "movie duration unknown (Stage 0 has not recorded data/derived/movie_timeline.json) "
            "-- times are NOT bounds-checked against the end of the movie. Run `make data` first."
        )

    # --- identifiers ---------------------------------------------------------
    ids = df["pair_id"].astype(str).str.strip()
    if ids.eq("").any():
        rep.error("blank pair_id(s) present")
    dupes = ids[ids.duplicated()].unique().tolist()
    if dupes:
        rep.error(f"duplicate pair_id(s): {', '.join(dupes)}")

    example_rows = ids[ids.str.upper().str.startswith("EXAMPLE")].tolist()

    rows = []
    row_runs: dict[str, tuple] = {}
    missing_desc: set[str] = set()
    for i, r in df.iterrows():
        pid = str(r["pair_id"]).strip()
        where = f"row {i + 2} ({pid})"
        ps, pe = r["plant_start_s"], r["plant_end_s"]
        os_, oe = r["payoff_start_s"], r["payoff_end_s"]

        if any(np.isnan(x) for x in (ps, pe, os_, oe)):
            continue  # parse errors already recorded

        plant_dur, payoff_dur = pe - ps, oe - os_
        gap = os_ - pe

        # window ordering and duration
        for label, a, b in (("plant", ps, pe), ("payoff", os_, oe)):
            if b <= a:
                rep.error(f"{where}: {label}_end must be after {label}_start "
                          f"({a:.1f} -> {b:.1f})")
            elif b - a < min_win:
                rep.error(f"{where}: {label} window is {b - a:.2f}s, "
                          f"below the {min_win:.0f}s minimum")
            if a < 0:
                rep.error(f"{where}: negative {label}_start ({a:.1f}s)")
            if duration is not None and b > duration:
                rep.error(f"{where}: {label}_end {b:.1f}s "
                          f"({seconds_to_hhmmss(b)}) is past the end of the movie "
                          f"({duration:.1f}s)")

        # Which run each moment falls in, and whether it sits in the
        # run-opening cartoon (identical clip at the start of BOTH runs, so
        # a pair drawn from it would compare literally the same stimulus).
        if timeline is not None:
            runs = _run_of(timeline)
            plant_run, payoff_run = runs(ps), runs(os_)
            row_runs[pid] = (plant_run, payoff_run)
            for label, a, b in (("plant", ps, pe), ("payoff", os_, oe)):
                span = _cartoon_span(timeline, a, b)
                if span is not None:
                    rep.error(
                        f"{where}: the {label} window ({a:.1f}-{b:.1f}s) overlaps the "
                        f"run-opening cartoon ({span[0]:.1f}-{span[1]:.1f}s). That clip "
                        "is shown identically at the start of both runs and is not "
                        "narrative content; it cannot serve as a plant or a payoff."
                    )
            if plant_run is None or payoff_run is None:
                rep.error(f"{where}: a moment falls outside both runs")

        # The window that will actually be sampled must exist in the data.
        # Two ways it can fail that a plain time-bounds check misses:
        # the HRF shift can push the window past the final TR, and a window
        # can straddle the Part1/Part2 boundary, which would average BOLD
        # across two separate runs.
        if timeline is not None:
            total_trs = int(timeline["total_trs"])
            part1_trs = int(timeline["part1_trs"])
            for label, a, b in (("plant", ps, pe), ("payoff", os_, oe)):
                if b <= a:
                    continue
                lo, hi = window_trs(a, b, tr_s, hrf_shift,
                                    wcfg["min_trs"], wcfg["max_trs"])
                if hi > total_trs:
                    rep.error(
                        f"{where}: the {label} window lands on TRs [{lo}, {hi}) but "
                        f"the scan ends at TR {total_trs}. After the "
                        f"{hrf_shift}-TR haemodynamic shift there is no BOLD left "
                        f"to sample; move it at least "
                        f"{(hi - total_trs) * tr_s:.1f}s earlier."
                    )
                elif lo < part1_trs < hi:
                    rep.error(
                        f"{where}: the {label} window spans TRs [{lo}, {hi}), "
                        f"crossing the Part1/Part2 boundary at TR {part1_trs}. "
                        "Averaging BOLD across two separate runs is an artifact; "
                        "move the moment wholly inside one part."
                    )

        # plant precedes payoff, with no overlap and a real temporal gap
        if os_ < ps:
            rep.error(f"{where}: payoff starts before the plant "
                      f"({seconds_to_hhmmss(os_)} < {seconds_to_hhmmss(ps)})")
        elif os_ < pe:
            rep.error(f"{where}: plant and payoff windows overlap "
                      f"(plant ends {pe:.1f}s, payoff starts {os_:.1f}s)")
        elif gap < min_gap:
            rep.error(f"{where}: gap is {gap:.1f}s, below the {min_gap:.0f}s minimum "
                      "(a short gap confounds the effect with temporal autocorrelation)")

        # modality: accept one column or two, and record both moments.
        if "modality_plant" in df.columns and "modality_payoff" in df.columns:
            mp = str(r["modality_plant"]).strip().lower()
            mo = str(r["modality_payoff"]).strip().lower()
            for label, m in (("modality_plant", mp), ("modality_payoff", mo)):
                if m not in VALID_MODALITIES:
                    rep.error(f"{where}: {label} {m!r} is not one of "
                              f"{sorted(VALID_MODALITIES)}")
            mod = mp if mp == mo else "mixed"
        else:
            mod = str(r["modality"]).strip().lower()
            mp = mo = mod
            if mod not in VALID_MODALITIES:
                rep.error(f"{where}: modality {r['modality']!r} is not one of "
                          f"{sorted(VALID_MODALITIES)}")

        # Descriptions are optional. They are only needed for the embedding
        # fallback and the wrong-time-frame check; the feature-matched Family B
        # does not use them.
        for col in ("plant_desc", "payoff_desc"):
            if col in df.columns:
                val = str(r[col]).strip()
                if not val or val.lower() == "nan":
                    missing_desc.add(col)

        chars_p = _split_chars(r["characters_plant"])
        chars_o = _split_chars(r["characters_payoff"])
        rows.append({
            "pair_id": pid,
            "modality": mod,
            "modality_plant": mp,
            "modality_payoff": mo,
            "cross_modal": int(mp != mo),
            "plant_start_s": ps, "plant_end_s": pe, "plant_dur_s": plant_dur,
            "payoff_start_s": os_, "payoff_end_s": oe, "payoff_dur_s": payoff_dur,
            "gap_s": gap,
            "gap_min": gap / 60.0,
            "plant_hhmmss": seconds_to_hhmmss(max(ps, 0)),
            "payoff_hhmmss": seconds_to_hhmmss(max(os_, 0)),
            "plant_desc": str(r.get("plant_desc", "")).strip(),
            "payoff_desc": str(r.get("payoff_desc", "")).strip(),
            "plant_run": row_runs.get(pid, (None, None))[0],
            "payoff_run": row_runs.get(pid, (None, None))[1],
            "crosses_run": (
                None if row_runs.get(pid, (None, None))[0] is None
                else int(row_runs[pid][0] != row_runs[pid][1])
            ),
            "characters_plant": ";".join(chars_p),
            "characters_payoff": ";".join(chars_o),
            "shared_characters": ";".join(sorted(set(chars_p) & set(chars_o))),
            "n_shared_characters": len(set(chars_p) & set(chars_o)),
        })

    summary = pd.DataFrame(rows)

    # --- crosses_run is COMPUTED, never trusted from the file ---------------
    if not summary.empty and summary["crosses_run"].notna().any():
        # df.get() on a missing column returns None, and pd.to_numeric(None)
        # yields a scalar rather than a Series -- guard before touching it.
        stated = (pd.to_numeric(df["crosses_run"], errors="coerce")
                  if "crosses_run" in df.columns else None)
        computed = summary["crosses_run"]
        n_cross = int(computed.sum())
        rep.note(f"crosses_run computed from the run boundary: **{n_cross}** of "
                 f"{len(summary)} pair(s) span the two runs "
                 f"({len(summary) - n_cross} within-run).")
        if stated is not None and stated.notna().any():
            # Only rows that actually stated a value can disagree; blanks are
            # the expected case, since the column is meant to be computed.
            s = stated.to_numpy()[:len(computed)]
            c = computed.to_numpy()
            filled = ~pd.isna(s)
            mism = int((s[filled] != c[filled]).sum())
            if mism:
                rep.warn(f"{mism} of {int(filled.sum())} row(s) that filled in "
                         "crosses_run disagreed with the run boundary. The computed "
                         "value is used; the column in the file is advisory only.")

    # --- moments inside the padded tail of the missing-data subject ----------
    if not summary.empty:
        for zone in _missing_zones(cfg):
            lo_s, hi_s, who = zone
            for _, r in summary.iterrows():
                for kind in ("plant", "payoff"):
                    if r[f"{kind}_start_s"] < hi_s and lo_s < r[f"{kind}_end_s"]:
                        rep.warn(
                            f"{r['pair_id']}.{kind} falls in {lo_s:.0f}-{hi_s:.0f}s, "
                            f"where {who} has no real data (the release pads it by "
                            "repeating the last volume). This pair will be dropped "
                            "for that subject only; every other subject keeps it."
                        )

    # --- cross-pair checks ---------------------------------------------------
    if not summary.empty:
        _check_cross_pair_overlap(summary, rep)

        n = len(summary)
        if example_rows:
            rep.warn(f"{len(example_rows)} template EXAMPLE row(s) present "
                     f"({', '.join(example_rows[:5])}). These are formatting "
                     "placeholders, not annotations -- delete them before locking.")
        if n < int(acfg["low_pair_count_warning"]):
            rep.warn(
                f"LOW PAIR COUNT -- {n} pair(s), below the target of "
                f"{acfg['low_pair_count_warning']}. Consider annotating Merlin, "
                "or reporting this as a small-N pilot."
            )
        shared = int((summary["n_shared_characters"] > 0).sum())
        if shared:
            rep.note(f"{shared}/{n} pair(s) share at least one character between "
                     "plant and payoff -- Stage 4 re-runs the analysis without them.")
        by_mod = summary["modality"].value_counts().to_dict()
        rep.note("modality counts: " + ", ".join(f"{k}={v}" for k, v in sorted(by_mod.items())))

    return summary


def _missing_zones(cfg: dict) -> list[tuple[float, float, str]]:
    """Time ranges where some subject has only padding, from the data-quality scan."""
    from src.data_quality import load as load_invalid
    tr = float(cfg["acquisition"]["tr_seconds"])
    zones: dict[tuple[int, int], set[str]] = {}
    for roi, subs in load_invalid(cfg).items():
        for subject, ranges in subs.items():
            for a, b in ranges:
                zones.setdefault((int(a), int(b)), set()).add(f"sub-{subject}")
    return [(a * tr, b * tr, ", ".join(sorted(who)))
            for (a, b), who in sorted(zones.items())]


def _run_of(timeline: dict):
    """Return a function mapping a time in seconds to its run number (1 or 2)."""
    boundary = float(timeline["part1_seconds"])
    duration = float(timeline["movie_duration_seconds"])

    def which(t: float) -> int | None:
        if t < 0 or t > duration:
            return None
        return 1 if t < boundary else 2

    return which


def _cartoon_span(timeline: dict, start: float, end: float):
    """The cartoon span this window overlaps, or None."""
    for span in timeline.get("cartoon_spans", []):
        if start < float(span["end_s"]) and float(span["start_s"]) < end:
            return float(span["start_s"]), float(span["end_s"])
    return None


def _split_chars(value) -> list[str]:
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    return [c.strip() for c in s.replace(",", ";").split(";") if c.strip()]


def _check_cross_pair_overlap(summary: pd.DataFrame, rep: Report) -> None:
    """Flag moments from different pairs that overlap in time.

    Not fatal -- a single film moment can legitimately pay off two plants --
    but it makes the pairs non-independent, which matters for the group stats.
    """
    moments = []
    for _, r in summary.iterrows():
        moments.append((r["pair_id"], "plant", r["plant_start_s"], r["plant_end_s"]))
        moments.append((r["pair_id"], "payoff", r["payoff_start_s"], r["payoff_end_s"]))
    for i in range(len(moments)):
        for j in range(i + 1, len(moments)):
            pid_a, ka, sa, ea = moments[i]
            pid_b, kb, sb, eb = moments[j]
            if pid_a == pid_b:
                continue
            if sa < eb and sb < ea:
                rep.warn(f"{pid_a}.{ka} overlaps {pid_b}.{kb} in time "
                         f"({sa:.1f}-{ea:.1f}s vs {sb:.1f}-{eb:.1f}s) -- "
                         "these pairs are not independent.")


def describe_gaps(summary: pd.DataFrame) -> list[str]:
    """Per-pair gap distribution, as required by the stage spec."""
    if summary.empty:
        return ["  (no valid pairs to summarise)"]
    g = summary["gap_s"].to_numpy(dtype=float)
    lines = [
        f"  n pairs         : {len(g)}",
        f"  gap min         : {g.min():8.1f}s  ({g.min() / 60:5.1f} min)",
        f"  gap 25th pct    : {np.percentile(g, 25):8.1f}s  ({np.percentile(g, 25) / 60:5.1f} min)",
        f"  gap median      : {np.median(g):8.1f}s  ({np.median(g) / 60:5.1f} min)",
        f"  gap 75th pct    : {np.percentile(g, 75):8.1f}s  ({np.percentile(g, 75) / 60:5.1f} min)",
        f"  gap max         : {g.max():8.1f}s  ({g.max() / 60:5.1f} min)",
        f"  gap mean +/- sd : {g.mean():8.1f}s +/- {g.std(ddof=1) if len(g) > 1 else 0:.1f}s",
        "",
        "  per pair:",
    ]
    for _, r in summary.sort_values("gap_s").iterrows():
        lines.append(
            f"    {r['pair_id']:<16s} {r['modality']:<9s} "
            f"plant {r['plant_hhmmss']} ({r['plant_dur_s']:.1f}s)  ->  "
            f"payoff {r['payoff_hhmmss']} ({r['payoff_dur_s']:.1f}s)   "
            f"gap {r['gap_s']:7.1f}s / {r['gap_min']:5.1f} min"
        )
    return lines


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Validate plant/payoff annotations.")
    p.add_argument("--file", default=None,
                   help="annotation CSV (default: annotations/pairs.csv, "
                        "falling back to the template)")
    p.add_argument("--config", default=None)
    p.add_argument("--no-time-frame-check", action="store_true",
                   help="skip the embedding-based wrong-time-frame check")
    p.add_argument("--strict", action="store_true",
                   help="treat warnings as failures")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    rep = Report()

    if args.file:
        path = Path(args.file)
        if not path.is_absolute():
            path = REPO_ROOT / path
    else:
        path = REPO_ROOT / cfg["annotations"]["path"]
        if not path.exists():
            template = REPO_ROOT / cfg["annotations"]["template_path"]
            print(f"NOTE: {path.relative_to(REPO_ROOT)} does not exist yet -- "
                  f"validating the template instead.\n"
                  f"      Annotation is human work: copy the template to "
                  f"{path.relative_to(REPO_ROOT)} and fill it in.\n")
            path = template

    print("=" * 72)
    print(f"Validating: {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
    print("=" * 72)

    if not path.exists():
        print(f"ERROR: no such file: {path}", file=sys.stderr)
        return 2

    df = parse_annotations(path, rep)
    if df is not None:
        df = apply_time_frame(df, cfg, rep)
    summary = pd.DataFrame() if df is None else validate(df, cfg, rep)

    if not summary.empty and not args.no_time_frame_check:
        from src.check_time_frame import check as _frame_check
        for msg in _frame_check(cfg, summary, verbose=False):
            rep.warn(msg)

    if not summary.empty:
        print("\nGap distribution")
        print("-" * 72)
        for line in describe_gaps(summary):
            print(line)

        out = resolve(cfg, "derived") / "annotation_summary.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        print(f"\n  per-pair summary written to {out.relative_to(REPO_ROOT)}")

    for label, items in (("NOTES", rep.notes), ("WARNINGS", rep.warnings), ("ERRORS", rep.errors)):
        if items:
            print(f"\n{label} ({len(items)})")
            print("-" * 72)
            for m in items:
                print(f"  - {m}")

    print("\n" + "=" * 72)
    if not rep.ok:
        print(f"INVALID -- {len(rep.errors)} error(s). Fix the annotation file and re-run.")
        print("=" * 72)
        return 1
    if args.strict and rep.warnings:
        print(f"VALID but --strict and {len(rep.warnings)} warning(s) present.")
        print("=" * 72)
        return 1
    print(f"VALID -- {len(summary)} pair(s), {len(rep.warnings)} warning(s).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
