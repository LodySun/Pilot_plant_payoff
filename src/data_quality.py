"""Detect timepoints that carry no real signal, per subject and ROI.

The Chen 17-subject ROI release contains one subject whose scan ends before
the movie does; the released file is padded to full length by repeating the
last real volume. Nothing in the file says so. Averaged into a pattern, those
timepoints look like ordinary data and would quietly contribute a constant to
every moment that touches them.

This module finds such stretches empirically -- consecutive timepoints
identical to their predecessor -- rather than hard-coding a subject index,
because which subject is affected depends on the release: the 16-subject
fmriprep derivatives have no such subject at all.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import append_data_note, ensure_dir, load_config, rel, resolve  # noqa: E402

CACHE_NAME = "invalid_trs.json"
MIN_RUN = 5  # a shorter coincidence is not evidence of padding


def detect_repeated_runs(data: np.ndarray, min_run: int = MIN_RUN) -> list[tuple[int, int]]:
    """Half-open TR ranges that merely repeat the preceding timepoint.

    Also catches all-constant columns, which is how some releases pad with
    zeros rather than by repetition.
    """
    if data.shape[1] < 2:
        return []
    identical = np.all(np.diff(data, axis=1) == 0, axis=0)  # length n_trs - 1
    flagged = np.zeros(data.shape[1], dtype=bool)
    flagged[1:] = identical

    # A column with no variance across voxels carries no pattern either.
    flagged |= data.std(axis=0) == 0

    out: list[tuple[int, int]] = []
    start = None
    for j, bad in enumerate(flagged):
        if bad and start is None:
            start = j
        elif not bad and start is not None:
            if j - start >= min_run:
                out.append((start, j))
            start = None
    if start is not None and data.shape[1] - start >= min_run:
        out.append((start, data.shape[1]))
    return out


def scan(cfg: dict, verbose: bool = True) -> dict[str, dict[str, list[list[int]]]]:
    """Scan every available ROI x subject. Returns {roi: {subject: [[lo, hi], ...]}}."""
    from src import data_io

    tr = float(cfg["acquisition"]["tr_seconds"])
    found: dict[str, dict[str, list[list[int]]]] = {}
    for roi, info in data_io.available_rois(cfg).items():
        for subject in info["subjects"]:
            ranges = detect_repeated_runs(data_io.load_roi(cfg, roi, subject))
            if ranges:
                found.setdefault(roi, {})[subject] = [[int(a), int(b)] for a, b in ranges]
                if verbose:
                    for a, b in ranges:
                        print(f"  {roi}/sub-{subject}: TRs [{a}, {b}) "
                              f"= {a * tr:.1f}-{b * tr:.1f}s "
                              f"({(b - a) * tr:.1f}s) repeat the previous timepoint")
    return found


def cache_path(cfg: dict) -> Path:
    return resolve(cfg, "derived") / CACHE_NAME


def load(cfg: dict) -> dict[str, dict[str, list[list[int]]]]:
    p = cache_path(cfg)
    if not p.exists():
        return {}
    with open(p) as fh:
        return json.load(fh)


def invalid_mask(cfg: dict, roi: str, subject: str, n_trs: int,
                 table: dict | None = None) -> np.ndarray | None:
    """Boolean mask of unusable TRs, or None when the subject is clean."""
    table = load(cfg) if table is None else table
    ranges = table.get(roi, {}).get(subject)
    if not ranges:
        return None
    mask = np.zeros(n_trs, dtype=bool)
    for a, b in ranges:
        mask[max(int(a), 0):min(int(b), n_trs)] = True
    return mask


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detect padded/unusable timepoints.")
    p.add_argument("--config", default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)

    print("=" * 72)
    print("Data quality scan -- repeated / constant timepoints")
    print("=" * 72)
    found = scan(cfg)

    out = ensure_dir(resolve(cfg, "derived")) / CACHE_NAME
    with open(out, "w") as fh:
        json.dump(found, fh, indent=2)

    tr = float(cfg["acquisition"]["tr_seconds"])
    if not found:
        print("  No repeated or constant stretches found in any ROI.")
        append_data_note(cfg, "Data quality", [
            "- Scanned every available ROI x subject for stretches of timepoints "
            f"that merely repeat the previous one (>= {MIN_RUN} TRs).",
            "- **None found.**",
        ])
        print(f"\nWrote {rel(out)}")
        return 0

    lines = [
        "- Scanned every available ROI x subject for stretches of timepoints that "
        f"merely repeat the previous one (>= {MIN_RUN} TRs) or carry no variance. "
        "This is padding, not data: it is not flagged anywhere in the released files.",
        "",
        "| ROI | subject | TRs | time | duration |",
        "|---|---|---|---|---|",
    ]
    for roi, subs in sorted(found.items()):
        for subject, ranges in sorted(subs.items(), key=lambda kv: int(kv[0])):
            for a, b in ranges:
                lines.append(
                    f"| `{roi}` | {subject} | [{a}, {b}) | "
                    f"{a * tr:.1f}-{b * tr:.1f} s | {(b - a) * tr:.1f} s |"
                )
    lines += [
        "",
        "Detected empirically, never by a hard-coded subject index: which subject "
        "is affected depends on the release. Any annotated moment overlapping these "
        "timepoints is dropped **for that subject only**, and the count is reported "
        "in `results/main/summary.md`.",
    ]
    append_data_note(cfg, "Data quality", lines)
    print(f"\nWrote {rel(out)} and recorded in docs/data_notes.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
