"""Detect plant/payoff timestamps entered in the wrong time frame.

The released scene spreadsheet restarts its seconds clock at Part 2, so a
timestamp like 300 s is genuinely ambiguous: it may mean 300 s into the
movie (Part 1), or 300 s into Part 2 -- which is 1719 s on the continuous
clock this pipeline uses.

A time cannot be checked arithmetically, but an annotation carries its own
description of what happens at that moment. So compare the annotator's
description against the scene text at both candidate times: if the
alternative reading matches the annotator's own words distinctly better,
the timestamp was probably entered in the other frame.

This warns; it never edits an annotation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_movie_timeline, seconds_to_hhmmss  # noqa: E402
from src.embeddings import embed_texts, scene_embeddings  # noqa: E402


def _scene_at(scenes: pd.DataFrame, t: float) -> int | None:
    """Row index of the segment containing time t on the continuous clock."""
    hit = scenes.index[(scenes["start_s"] <= t) & (scenes["end_s"] > t)]
    if len(hit):
        return int(hit[0])
    return None


def check(cfg: dict, summary: pd.DataFrame, verbose: bool = True) -> list[str]:
    """Return a list of warnings about probable wrong-frame timestamps."""
    timeline = load_movie_timeline(cfg)
    if timeline is None:
        return ["time-frame check skipped: data/derived/movie_timeline.json absent "
                "(run `make data`)."]
    if summary.empty:
        return []

    offset = float(timeline["part1_seconds"])
    duration = float(timeline["movie_duration_seconds"])
    margin = float(cfg["annotations"].get("time_frame_check_margin", 0.15))

    try:
        scenes, scene_emb = scene_embeddings(cfg, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        return [f"time-frame check skipped: {type(exc).__name__}: {exc}"]

    # Collect the moments worth checking: those whose alternative reading
    # still lands inside the movie.
    todo = []
    for _, r in summary.iterrows():
        for kind in ("plant", "payoff"):
            t = float(r[f"{kind}_start_s"])
            alt = t + offset
            if t < offset and alt < duration:
                todo.append((r["pair_id"], kind, t, alt))
    if not todo:
        return []

    descs = []
    for pid, kind, _t, _alt in todo:
        row = summary.loc[summary["pair_id"] == pid].iloc[0]
        descs.append(str(row.get(f"{kind}_desc", "")).strip())

    if not any(descs):
        return ["time-frame check skipped: no descriptions available to compare."]

    emb = embed_texts(cfg, descs)

    warnings: list[str] = []
    for (pid, kind, t, alt), e in zip(todo, emb):
        i_a, i_b = _scene_at(scenes, t), _scene_at(scenes, alt)
        if i_a is None or i_b is None:
            continue
        sim_a = float(np.dot(e, scene_emb[i_a]))
        sim_b = float(np.dot(e, scene_emb[i_b]))
        if sim_b - sim_a > margin:
            warnings.append(
                f"{pid}.{kind}: timestamp {t:.1f}s ({seconds_to_hhmmss(t)}) may be "
                f"Part-2-local. The description matches the scene at "
                f"{alt:.1f}s ({seconds_to_hhmmss(alt)}) better "
                f"(similarity {sim_b:.2f} vs {sim_a:.2f}).\n"
                f"      stated  scene: \"{str(scenes.at[i_a, 'description'])[:70]}\"\n"
                f"      alt     scene: \"{str(scenes.at[i_b, 'description'])[:70]}\"\n"
                f"      If Part-2-local was intended, the cumulative time is {alt:.1f}s."
            )
    return warnings
