"""Stage 0b -- parse the released Sherlock segment spreadsheet into a tidy CSV.

The released file (Sherlock_Segments_1000_NN_2017.xlsx) uses two clocks:

  * "Start/End Time (s)" is LOCAL to the movie part. It runs 0..1419 for
    Part 1, then RESETS to 0 for Part 2.
  * "Start/End Time (TRs, 1.5s)" is CUMULATIVE across both parts and
    1-BASED (segment 1 starts at 0 s and is labelled TR 1).

Everything downstream uses a single continuous clock: cumulative seconds from
the start of Part 1, and 0-based TR indices into the concatenated timeseries.
This module is the only place that conversion happens.

Both facts above were verified against the file rather than assumed; the
verification is re-run here on every parse and recorded in docs/data_notes.md.
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
    append_data_note,
    ensure_dir,
    load_config,
    resolve,
    save_movie_timeline,
    seconds_to_hhmmss,
)

SEGMENT_FILENAME = "Sherlock_Segments_1000_NN_2017.xlsx"
SHEET = "Segments"

# Column names as they appear in the released workbook (note trailing spaces).
COL_SEG = "Segment Number"
COL_START_S = "Start Time (s) "
COL_END_S = "End Time (s) "
COL_START_TR = "Start Time (TRs, 1.5s)"
COL_END_TR = "End Time (TRs, 1.5s)"
COL_SCENE = "Scene_Segment"
COL_SCENE_NAME = "Scene_Name"
COL_DESC = "Scene Details - A Level"
COL_NAMES_ALL = "Name - All"
COL_NAMES_FOCUS = "Name - Focus"
COL_NAMES_SPEAK = "Name - Speaking"
COL_LOCATION = "Location"
COL_INDOOR = "Space-In/Outdoor"
COL_MUSIC = "Music Presence "
COL_TEXT_ON_SCREEN = "Words on Screen "
COL_AROUSAL = [f"Arousal - Rater {k}" for k in range(1, 5)]
COL_VALENCE = [f"Valence - Rater {k}" for k in range(1, 5)]

# Main characters whose on-screen presence Family B matches on.
MAIN_CHARACTERS = {"sherlock": "sherlock_onscreen", "john": "john_onscreen"}


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _split_names(value) -> list[str]:
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", ""):
        return []
    return [x.strip() for x in s.replace(",", ";").split(";") if x.strip()]


def _affect_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Composite arousal and valence from the four raters.

    Arousal is a 1-5 rating, but the raters differ sharply in where they sit
    on that scale (rater 1 puts 69% of segments at 4-5, rater 3 puts 69% at
    1-2). Averaging raw scores would let one rater dominate, so each rater is
    z-scored over the whole movie before averaging.

    Valence is NOT numeric -- each rater codes only '+' or '-'. It is
    summarised as the count of '+' across available raters, which is the only
    faithful reduction; there is no SD to take a tolerance on.
    """
    notes: list[str] = []

    arous = pd.DataFrame({c: _num(raw[c]) for c in COL_AROUSAL if c in raw.columns})
    if arous.shape[1]:
        z = (arous - arous.mean()) / arous.std(ddof=0)
        arousal_z = z.mean(axis=1)
        arousal_raw = arous.mean(axis=1)
        spread = ", ".join(f"R{i+1} mean {m:.2f}" for i, m in enumerate(arous.mean()))
        notes.append(f"- Arousal: **{arous.shape[1]} raters**, scale 1-5, "
                     f"markedly different rater baselines ({spread}); composite is "
                     "the mean of per-rater z-scores.")
    else:
        arousal_z = arousal_raw = pd.Series(np.nan, index=raw.index)
        notes.append("- Arousal: **columns absent**.")

    val = pd.DataFrame({c: raw[c].astype(str).str.strip() for c in COL_VALENCE
                        if c in raw.columns})
    if val.shape[1]:
        pos = val.apply(lambda s: s.eq("+")).sum(axis=1)
        rated = val.apply(lambda s: s.isin(["+", "-"])).sum(axis=1)
        notes.append(f"- Valence: **{val.shape[1]} raters**, coded `+`/`-` only "
                     "(NOT numeric). Composite is the count of `+` (0-4); "
                     f"{int((rated < val.shape[1]).sum())} segment(s) have a "
                     "rater missing, so the proportion is also stored.")
    else:
        pos = rated = pd.Series(np.nan, index=raw.index)
        notes.append("- Valence: **columns absent**.")

    feats = pd.DataFrame({
        "arousal_z": arousal_z,
        "arousal_raw": arousal_raw,
        "valence_pos": pos,
        "valence_prop": pos / rated.replace(0, np.nan),
    })
    return feats, notes


def parse(xlsx: Path, cfg: dict) -> tuple[pd.DataFrame, dict, list[str]]:
    """Return (tidy scene table, timeline facts, human-readable findings)."""
    tr = float(cfg["acquisition"]["tr_seconds"])
    raw = pd.read_excel(xlsx, sheet_name=SHEET, dtype=object)
    raw = raw[raw[COL_SEG].notna()].reset_index(drop=True)

    findings: list[str] = [f"- Source file: `{xlsx.name}`", f"- Segments in file: **{len(raw)}**"]

    start_local = _num(raw[COL_START_S])
    end_local = _num(raw[COL_END_S])
    start_tr_1b = _num(raw[COL_START_TR])
    end_tr_1b = _num(raw[COL_END_TR])

    # --- locate the part boundary from the backward jump in local seconds ----
    backward = np.where(start_local.to_numpy()[1:] < start_local.to_numpy()[:-1])[0]
    if len(backward) != 1:
        raise RuntimeError(
            f"expected exactly one part boundary (backward jump in local seconds), "
            f"found {len(backward)} at rows {backward.tolist()}"
        )
    boundary_row = int(backward[0]) + 1  # first row of Part 2
    part = np.where(np.arange(len(raw)) < boundary_row, 1, 2)

    # --- derive part 1 length in TRs, and check it is self-consistent -------
    p2 = part == 2
    offsets = (start_tr_1b[p2] - (np.floor(start_local[p2] / tr) + 1)).dropna()
    part1_trs = int(offsets.mode().iloc[0])
    off_p1 = (start_tr_1b[~p2] - (np.floor(start_local[~p2] / tr) + 1)).dropna()
    if not off_p1.empty and int(off_p1.mode().iloc[0]) != 0:
        raise RuntimeError("Part 1 TRs are not aligned to local seconds; "
                           "the file's timing convention is not what this parser assumes.")

    total_trs = int(np.nanmax(end_tr_1b.to_numpy()))
    part2_trs = total_trs - part1_trs

    findings += [
        f"- Part boundary at segment **{int(raw[COL_SEG].iloc[boundary_row])}** "
        f"(row {boundary_row + 2} of the sheet).",
        f"- Local-seconds clock **resets to 0 at Part 2** -- it is NOT cumulative.",
        f"- TR column is **1-based and cumulative** across parts "
        f"(segment 1 spans 0-12 s and is labelled TR 1-8).",
        f"- Part 1 = **{part1_trs} TRs**, Part 2 = **{part2_trs} TRs**, "
        f"total = **{total_trs} TRs** = {total_trs * tr:.1f} s "
        f"({seconds_to_hhmmss(total_trs * tr)}) at TR={tr}s.",
    ]

    # --- build the single continuous clock ---------------------------------
    part1_seconds = part1_trs * tr
    cum_start = np.where(p2, start_local + part1_seconds, start_local)
    cum_end = np.where(p2, end_local + part1_seconds, end_local)

    # 0-based TR indices into the concatenated timeseries.
    start_tr0 = start_tr_1b - 1
    end_tr0 = end_tr_1b - 1

    unmapped = raw[start_tr_1b.isna()]
    if len(unmapped):
        segs = ", ".join(str(int(s)) for s in unmapped[COL_SEG])
        findings.append(
            f"- **{len(unmapped)} segment(s) have no TR** in the released file "
            f"(segment {segs}): they fall past the end of the Part 1 scan. "
            "They are kept in the scene table but excluded from TR-based sampling."
        )

    scenes = pd.DataFrame({
        "segment": _num(raw[COL_SEG]).astype("Int64"),
        "part": part,
        "start_local_s": start_local,
        "end_local_s": end_local,
        "start_s": cum_start,
        "end_s": cum_end,
        "start_tr": start_tr0.astype("Int64"),
        "end_tr": end_tr0.astype("Int64"),
        "scene_id": _num(raw[COL_SCENE]).astype("Int64"),
        "scene_name": raw[COL_SCENE_NAME].astype(str).str.strip(),
        "description": raw[COL_DESC].astype(str).str.strip(),
        "characters": raw[COL_NAMES_ALL].astype(str).str.strip(),
        "characters_focus": raw[COL_NAMES_FOCUS].astype(str).str.strip(),
        "characters_speaking": raw[COL_NAMES_SPEAK].astype(str).str.strip(),
        "location": raw[COL_LOCATION].astype(str).str.strip(),
        "indoor_outdoor": raw[COL_INDOOR].astype(str).str.strip(),
        "music": raw[COL_MUSIC].astype(str).str.strip(),
        "words_on_screen": raw[COL_TEXT_ON_SCREEN].astype(str).str.strip(),
    })
    scenes["duration_s"] = scenes["end_s"] - scenes["start_s"]

    # --- stimulus features used by Family-B control matching ---------------
    affect, affect_notes = _affect_features(raw)
    scenes = pd.concat([scenes, affect], axis=1)

    names = raw[COL_NAMES_ALL].map(_split_names)
    scenes["n_persons"] = names.map(len)
    for token, col in MAIN_CHARACTERS.items():
        scenes[col] = names.map(lambda ns, _t=token: any(_t in n.lower() for n in ns))
    scenes["speaking"] = raw[COL_NAMES_SPEAK].astype(str).str.strip().replace(
        {"nan": "", "None": ""})
    scenes["anyone_speaking"] = scenes["speaking"].str.len() > 0
    scenes["has_words_on_screen"] = (
        raw[COL_TEXT_ON_SCREEN].astype(str).str.strip()
        .replace({"nan": "", "None": ""}).str.len() > 0
    )
    scenes["is_cartoon"] = scenes["scene_name"].str.strip().str.casefold() == "cartoon"

    findings += affect_notes + [
        f"- Persons on screen (from `{COL_NAMES_ALL}`): "
        f"{scenes['n_persons'].min()}-{scenes['n_persons'].max()} per segment.",
        f"- Sherlock on screen in **{int(scenes['sherlock_onscreen'].sum())}** "
        f"segments, John in **{int(scenes['john_onscreen'].sum())}**; neither "
        "has a dedicated column, both are derived from the name list.",
        f"- Someone is speaking in **{int(scenes['anyone_speaking'].sum())}**/"
        f"{len(scenes)} segments.",
        f"- `{COL_TEXT_ON_SCREEN.strip()}` is non-empty in only "
        f"**{int(scenes['has_words_on_screen'].sum())}**/{len(scenes)} segments -- "
        "too sparse to match on, so it is carried as present/absent only.",
        f"- Run-opening cartoon: **{int(scenes['is_cartoon'].sum())}** segments "
        f"across {scenes.loc[scenes['is_cartoon'], 'part'].nunique()} runs.",
    ]

    n_scenes = int(scenes["scene_id"].nunique())
    findings += [
        f"- Distinct scenes (`Scene_Segment`): **{n_scenes}**; "
        f"every segment carries a text description "
        f"(mean {scenes['description'].str.len().mean():.0f} characters).",
        f"- Movie duration on the continuous clock: **{cum_end.max():.1f} s** "
        f"({seconds_to_hhmmss(float(cum_end.max()))}).",
    ]

    timeline = {
        "source": SEGMENT_FILENAME,
        "tr_seconds": tr,
        "part1_trs": part1_trs,
        "part2_trs": part2_trs,
        "total_trs": total_trs,
        "part1_seconds": part1_seconds,
        "part2_offset_seconds": part1_seconds,
        "movie_duration_seconds": float(total_trs * tr),
        "annotated_duration_seconds": float(cum_end.max()),
        "n_segments": int(len(scenes)),
        "n_scenes": n_scenes,
        "tr_index_base": 0,
        "tr_column_base_in_source": 1,
        "clock": "cumulative seconds from the start of Part 1",
        "n_segments_without_tr": int(len(unmapped)),
    }

    # Cartoon spans on the continuous clock, and the narrative range that
    # remains once they are removed. Stage 1 and Stage 2 both need these.
    cart = scenes.loc[scenes["is_cartoon"]]
    spans = []
    for part_no, g in cart.groupby("part"):
        spans.append({"part": int(part_no),
                      "start_s": float(g["start_s"].min()),
                      "end_s": float(g["end_s"].max()),
                      "scene_id": int(g["scene_id"].dropna().iloc[0])})
    timeline["cartoon_spans"] = spans
    timeline["cartoon_seconds"] = float(sum(s["end_s"] - s["start_s"] for s in spans))
    narrative = []
    for part_no, g in scenes[~scenes["is_cartoon"]].groupby("part"):
        gg = g[g["start_tr"].notna()]
        narrative.append({"part": int(part_no),
                          "start_s": float(gg["start_s"].min()),
                          "end_s": float(gg["end_s"].max())})
    timeline["narrative_spans"] = narrative

    # Episode time: the clock a viewer of the episode itself would read.
    # It differs from the continuous clock because each run opens with a
    # cartoon that is not part of the episode. Derived here, never assumed.
    cart_by_part = {int(s["part"]): s["end_s"] - s["start_s"] for s in spans}
    offsets, cum = {}, 0.0
    for part_no in sorted(cart_by_part):
        cum += float(cart_by_part[part_no])
        offsets[part_no] = cum
    timeline["episode_offset_by_part"] = {str(k): v for k, v in offsets.items()}
    if narrative:
        first_p2 = [s for s in narrative if s["part"] == 2]
        timeline["episode_split_s"] = (
            float(first_p2[0]["start_s"]) - offsets.get(2, 0.0) if first_p2 else None)
        timeline["episode_duration_s"] = float(
            max(s["end_s"] for s in narrative) - offsets[max(offsets)])
        findings += [
            "",
            f"- **Episode clock**: the stimulus is the episode's own "
            f"**00:00.0 - {timeline['episode_duration_s'] / 60:.0f}:"
            f"{timeline['episode_duration_s'] % 60:04.1f}** "
            f"({timeline['episode_duration_s']:.0f} s = "
            f"{timeline['episode_duration_s'] / 60:.1f} min), split between the two "
            f"runs at episode **{timeline['episode_split_s']:.1f} s**.",
            "- Convert with `episode = continuous - offset`, offset "
            + ", ".join(f"{v:.0f} s in part {k}" for k, v in offsets.items())
            + " (the run-opening cartoons are not episode content).",
        ]
    findings += [
        "",
        "Run-opening cartoon (identical clip shown at the start of BOTH runs) "
        "and the narrative range left once it is excluded:",
        "",
        "| | span (continuous clock) | scene_id |",
        "|---|---|---|",
        *[f"| cartoon, part {s['part']} | {s['start_s']:.1f}-{s['end_s']:.1f} s "
          f"({s['end_s'] - s['start_s']:.0f} s) | {s['scene_id']} |" for s in spans],
        *[f"| narrative, part {s['part']} | {s['start_s']:.1f}-{s['end_s']:.1f} s | |"
          for s in narrative],
    ]
    return scenes, timeline, findings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Parse the Sherlock segment spreadsheet.")
    p.add_argument("--config", default=None)
    p.add_argument("--file", default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    src_dir = resolve(cfg, "annotations_source")
    xlsx = Path(args.file) if args.file else src_dir / SEGMENT_FILENAME

    if not xlsx.exists():
        print(f"ERROR: segment file not found at {xlsx}\n"
              f"  Download it with:\n"
              f"    curl -L -o {xlsx} \\\n"
              f"      https://gin.g-node.org/ljchang/Sherlock/raw/master/onsets/{SEGMENT_FILENAME}",
              file=sys.stderr)
        return 2

    scenes, timeline, findings = parse(xlsx, cfg)

    out = ensure_dir(resolve(cfg, "derived")) / "scenes.csv"
    scenes.to_csv(out, index=False)
    save_movie_timeline(cfg, timeline)

    append_data_note(cfg, "Movie timeline and scene annotations", findings + [
        "",
        "Downstream convention (set here, used everywhere):",
        "",
        "- Times in `annotations/pairs.csv` are **cumulative seconds from the start of Part 1**.",
        f"- Part 2 local time converts as `cumulative = local + {timeline['part1_seconds']:.1f}`.",
        "- TR indices are **0-based** into the concatenated Part1+Part2 timeseries.",
        f"- Concatenated length: **{timeline['total_trs']} TRs**.",
        "",
        f"Parsed table: `data/derived/scenes.csv` ({len(scenes)} rows).",
    ])

    print(f"Parsed {len(scenes)} segments -> {out.relative_to(REPO_ROOT)}")
    for line in findings:
        print("  " + line.lstrip("- "))
    print(f"\nTimeline written to {resolve(cfg, 'derived').relative_to(REPO_ROOT)}/movie_timeline.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
