"""Shared helpers: config loading, path resolution, and time <-> TR conversion.

Every stage imports from here so that the TR, the HRF shift, and the window
clamping rules have exactly one definition.
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load config.yaml as a plain dict."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] against the repo root."""
    return REPO_ROOT / cfg["paths"][key]


def rel(p: str | os.PathLike) -> str:
    """Repo-relative path for display, falling back to the full path."""
    p = Path(p)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------
# Time conversion
# --------------------------------------------------------------------------

def hhmmss_to_seconds(ts: str) -> float:
    """'01:23:45.5' or '23:45' or '45' -> seconds."""
    ts = str(ts).strip()
    if not ts:
        raise ValueError("empty timestamp")
    parts = ts.split(":")
    if len(parts) > 3:
        raise ValueError(f"unparseable timestamp: {ts!r}")
    total = 0.0
    for part in parts:
        total = total * 60.0 + float(part)
    return total


def seconds_to_hhmmss(seconds: float) -> str:
    """Seconds -> 'HH:MM:SS.s' (fractional seconds kept to one decimal)."""
    if seconds < 0:
        raise ValueError("negative time")
    hours, rem = divmod(float(seconds), 3600.0)
    minutes, secs = divmod(rem, 60.0)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:04.1f}"


def seconds_to_tr(seconds: float, tr: float, hrf_shift: int = 0) -> int:
    """Stimulus time in seconds -> index of the TR that samples it.

    The HRF shift is applied in TR units: the BOLD response to an event at
    time t is read out at TR floor(t / tr) + hrf_shift.
    """
    if seconds < 0:
        raise ValueError("negative time")
    return int(math.floor(seconds / tr)) + int(hrf_shift)


def tr_to_seconds(tr_index: int, tr: float, hrf_shift: int = 0) -> float:
    """Inverse of seconds_to_tr: TR index -> stimulus time at its onset."""
    return (int(tr_index) - int(hrf_shift)) * tr


def window_trs(
    start_s: float,
    end_s: float,
    tr: float,
    hrf_shift: int,
    min_trs: int,
    max_trs: int,
) -> tuple[int, int]:
    """Moment [start_s, end_s) -> half-open TR window [lo, hi).

    Length is ceil(duration / tr) clamped to [min_trs, max_trs], anchored at
    the HRF-shifted onset TR. Returns indices into the concatenated timeseries;
    callers are responsible for bounds-checking against the run length.
    """
    if end_s <= start_s:
        raise ValueError(f"non-positive duration: {start_s} -> {end_s}")
    onset = seconds_to_tr(start_s, tr, hrf_shift)
    n = int(math.ceil((end_s - start_s) / tr))
    n = max(min_trs, min(max_trs, n))
    return onset, onset + n


def fixed_window_trs(start_s: float, tr: float, hrf_shift: int, n_trs: int) -> tuple[int, int]:
    """TR window of a forced fixed length, for the robustness sweep."""
    onset = seconds_to_tr(start_s, tr, hrf_shift)
    return onset, onset + int(n_trs)


# --------------------------------------------------------------------------
# Annotation locking (guardrail 2)
# --------------------------------------------------------------------------

def stable_seed(*parts: object) -> int:
    """A deterministic 32-bit seed from arbitrary parts.

    Python's builtin hash() is salted per process (PYTHONHASHSEED), so using
    it to seed an RNG makes "reproducible" synthetic data differ on every
    run. This uses blake2b instead, which is stable across processes and
    machines.
    """
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest()[:4], "big")


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class LockStatus:
    ok: bool
    message: str


def check_annotation_lock(cfg: dict) -> LockStatus:
    """Verify annotations/pairs.csv matches its recorded SHA256 lock.

    The pilot's order of operations -- annotate, lock, then compute anything
    involving real-pair neural similarity -- is enforced here.
    """
    ann = REPO_ROOT / cfg["annotations"]["path"]
    lock = REPO_ROOT / cfg["annotations"]["lock_path"]

    if not ann.exists():
        return LockStatus(False, f"annotation file missing: {rel(ann)}")
    if not lock.exists():
        return LockStatus(
            False,
            f"annotations are NOT locked (no {rel(lock)}).\n"
            "  Lock them once annotation is final:  make lock-annotations",
        )

    recorded = lock.read_text().split()[0].strip()
    actual = sha256_file(ann)
    if recorded != actual:
        return LockStatus(
            False,
            "ANNOTATIONS CHANGED AFTER LOCKING.\n"
            f"  locked hash: {recorded}\n"
            f"  actual hash: {actual}\n"
            "  Real-pair statistics computed before this change are no longer valid.\n"
            "  Re-lock deliberately (make lock-annotations) and re-run every stage.",
        )
    return LockStatus(True, f"annotations locked and unchanged ({actual[:12]}...)")


def require_lock_or_exit(cfg: dict, *, smoke: bool = False) -> None:
    """Hard gate. Smoke-test mode bypasses it (random data, no real claims).

    Distinguishes the two very different reasons this can fail:
    a missing annotation file is the ordinary state before annotation is
    done, and gets guidance (exit 4); a missing or stale lock on an
    existing file is a process violation, and gets refused (exit 3).
    """
    if smoke:
        print("[lock] --smoke-test: lock check bypassed (synthetic data).")
        sys.stdout.flush()
        return

    ann = REPO_ROOT / cfg["annotations"]["path"]
    if not ann.exists():
        sys.stdout.flush()
        print(
            f"\nNOT RUN -- there are no annotations yet ({rel(ann)}).\n"
            "\n"
            "  Annotation is human work; nothing here writes it for you.\n"
            "    1. cp annotations/pairs_template.csv annotations/pairs.csv\n"
            "    2. fill it in (delete the EXAMPLE_ rows)\n"
            "    3. make validate\n"
            "    4. make lock-annotations\n"
            "    5. make all\n"
            "\n"
            "  Everything that does not depend on the annotations has still\n"
            "  been built. To exercise the full pipeline meanwhile: make smoke\n",
            file=sys.stderr,
        )
        raise SystemExit(4)

    status = check_annotation_lock(cfg)
    if not status.ok:
        sys.stdout.flush()
        print("\n" + "=" * 72, file=sys.stderr)
        print("REFUSING TO RUN: " + status.message, file=sys.stderr)
        print("=" * 72 + "\n", file=sys.stderr)
        raise SystemExit(3)
    print(f"[lock] {status.message}")
    sys.stdout.flush()


# --------------------------------------------------------------------------
# Data notes
# --------------------------------------------------------------------------

def append_data_note(cfg: dict, section: str, lines: list[str]) -> None:
    """Append (or replace) a titled section in docs/data_notes.md.

    Stages record what they actually found in the data here, so that no
    dimension is left as an assumption.
    """
    notes = REPO_ROOT / cfg["paths"]["data_notes"]
    ensure_dir(notes.parent)
    header = f"## {section}"
    # Trailing blank line keeps the next '## ' heading separated.
    body = header + "\n\n" + "\n".join(lines).rstrip() + "\n\n"

    if not notes.exists():
        notes.write_text("# Data notes\n\nAuto-generated by the pipeline; "
                         "each section records what a stage read off the data.\n\n" + body)
        return

    text = notes.read_text()
    if header in text:
        # Replace the existing section, up to the next '## ' or EOF.
        start = text.index(header)
        rest = text[start + len(header):]
        nxt = rest.find("\n## ")
        end = len(text) if nxt == -1 else start + len(header) + nxt + 1
        text = text[:start] + body + text[end:]
    else:
        text = text.rstrip() + "\n\n" + body
    notes.write_text(text.rstrip() + "\n")


# --------------------------------------------------------------------------
# Movie timeline (written by Stage 0 from the actual data; read by 1-5)
# --------------------------------------------------------------------------

TIMELINE_FILENAME = "movie_timeline.json"


def timeline_path(cfg: dict) -> Path:
    return resolve(cfg, "derived") / TIMELINE_FILENAME


def run_bounds(timeline: dict) -> list[tuple[int, int]]:
    """Half-open [start, end) TR bounds of each run in the concatenated series.

    The two movie parts are separate scans concatenated along time. Nothing
    may be averaged, z-scored, or correlated across the seam between them.
    """
    p1 = int(timeline["part1_trs"])
    total = int(timeline["total_trs"])
    return [(0, p1), (p1, total)]


def load_movie_timeline(cfg: dict) -> dict | None:
    """Facts Stage 0 read off the data: run lengths, n subjects, duration.

    Returns None when Stage 0 has not run. Callers must degrade gracefully
    rather than substituting an assumed movie duration or TR count.
    """
    import json
    p = timeline_path(cfg)
    if not p.exists():
        return None
    with open(p) as fh:
        return json.load(fh)


def save_movie_timeline(cfg: dict, timeline: dict) -> Path:
    import json
    p = timeline_path(cfg)
    ensure_dir(p.parent)
    with open(p, "w") as fh:
        json.dump(timeline, fh, indent=2)
    return p


# --------------------------------------------------------------------------
# Episode clock
# --------------------------------------------------------------------------

def cumulative_to_episode(cum_s: float, timeline: dict) -> float | None:
    """Continuous-clock seconds -> the episode's own timecode.

    They differ because each run opens with a cartoon that is not part of the
    episode, so the episode clock runs behind the continuous clock by the
    cartoon time elapsed so far.
    """
    offsets = timeline.get("episode_offset_by_part")
    if not offsets:
        return None
    # A time inside a run-opening cartoon has NO episode time -- the cartoon
    # is not part of the episode. Returning a number there would silently
    # invent one.
    for span in timeline.get("cartoon_spans", []):
        if float(span["start_s"]) <= cum_s < float(span["end_s"]):
            return None
    part = 1 if cum_s < float(timeline["part1_seconds"]) else 2
    return float(cum_s) - float(offsets[str(part)])


def episode_to_cumulative(ep_s: float, timeline: dict) -> float | None:
    """The episode's own timecode -> continuous-clock seconds."""
    offsets = timeline.get("episode_offset_by_part")
    split = timeline.get("episode_split_s")
    if not offsets or split is None:
        return None
    part = 1 if float(ep_s) < float(split) else 2
    return float(ep_s) + float(offsets[str(part)])
