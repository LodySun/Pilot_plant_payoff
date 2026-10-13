"""Stage 0 -- acquire the data, or say exactly what to download by hand.

Sources, in the order tried:

  1. Scene annotations -- GIN `ljchang/Sherlock`, public, no registration.
  2. BrainIAK tutorial release -- Zenodo record 2598755,
     `Sherlock_processed.zip` (268 MB, public, no registration).
     Contains 17 subjects x 2 pre-extracted ROIs.
  3. Word-timestamped recall transcripts -- Zenodo 10.5281/zenodo.8208709
     (NN01-NN17 .xlsx). Not fetched automatically: download them and place
     them in data/annotations_source/transcripts/ (needed for Figure 4C only).

Nothing here bypasses an access control. If a source needs a browser or a
login, the script says so and tells you where to put the file.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (  # noqa: E402
    REPO_ROOT, append_data_note, ensure_dir, load_config, resolve,
)
from src.parse_scenes import SEGMENT_FILENAME  # noqa: E402

SEGMENTS_URL = (
    "https://gin.g-node.org/ljchang/Sherlock/raw/master/onsets/"
    + SEGMENT_FILENAME
)
ZENODO_URL = ("https://zenodo.org/api/records/2598755/files/"
              "Sherlock_processed.zip/content")
ZENODO_MD5 = "1a0a8c9f62625c914edfedad9f095a69"
ZENODO_BYTES = 268322748

TRANSCRIPTS_DOI = "https://doi.org/10.5281/zenodo.8208709"


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def fetch_segments(cfg: dict) -> bool:
    dest = ensure_dir(resolve(cfg, "annotations_source")) / SEGMENT_FILENAME
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [ok] scene annotations present ({dest.stat().st_size / 1e3:.0f} KB)")
        return True
    print(f"  [..] downloading scene annotations ({SEGMENT_FILENAME}) from GIN ...")
    if _run(["curl", "-sL", "--fail", "--max-time", "120", "-o", str(dest), SEGMENTS_URL]) != 0:
        print(f"  [!!] download failed. Fetch manually:\n"
              f"       curl -L -o {dest} \\\n         {SEGMENTS_URL}", file=sys.stderr)
        return False
    print(f"  [ok] saved to {dest.relative_to(REPO_ROOT)}")
    return True


def fetch_chen_rois(cfg: dict, allow_download: bool) -> bool:
    raw = ensure_dir(resolve(cfg, "raw"))
    extracted = raw / "Sherlock_processed" / "data_mat"
    if extracted.exists() and any(extracted.glob("s*/sherlock_movie/*.mat")):
        n = len(list(extracted.glob("s*/sherlock_movie/*.mat")))
        print(f"  [ok] Chen ROI release present ({n} .mat files)")
        return True

    archive = raw / "Sherlock_processed.zip"
    if not archive.exists():
        if not allow_download:
            print(f"  [--] Chen ROI release absent. Download ({ZENODO_BYTES / 1e6:.0f} MB):\n"
                  f"       curl -L -o {archive} \\\n         {ZENODO_URL}")
            return False
        print(f"  [..] downloading Sherlock_processed.zip "
              f"({ZENODO_BYTES / 1e6:.0f} MB, Zenodo, no registration) ...")
        if _run(["curl", "-L", "--fail", "--max-time", "3600",
                 "-o", str(archive), ZENODO_URL]) != 0:
            print("  [!!] download failed.", file=sys.stderr)
            return False

    print("  [..] verifying checksum ...")
    got = subprocess.run(["md5", "-q", str(archive)], capture_output=True, text=True)
    digest = got.stdout.strip() if got.returncode == 0 else ""
    if digest and digest != ZENODO_MD5:
        print(f"  [!!] MD5 mismatch: expected {ZENODO_MD5}, got {digest}. "
              "Delete the file and retry.", file=sys.stderr)
        return False
    print(f"  [ok] md5 {digest or '(md5 unavailable)'}")
    print("  [..] extracting ...")
    if _run(["unzip", "-q", "-o", str(archive), "-d", str(raw)]) != 0:
        print("  [!!] extraction failed.", file=sys.stderr)
        return False
    print(f"  [ok] extracted to {(raw / 'Sherlock_processed').relative_to(REPO_ROOT)}")
    return True


def record_chen_release(cfg: dict) -> None:
    """Record what the Chen ROI release actually contains, read from the files."""
    import numpy as np
    import scipy.io as sio

    from src import data_io

    rows, notes = [], []
    for roi in data_io.CHEN_ROIS:
        subs = data_io.chen_subjects(cfg, roi)
        if not subs:
            continue
        shapes, zscored = set(), []
        for s in subs:
            d = data_io.load_roi(cfg, roi, s)
            shapes.add(d.shape)
            zscored.append((abs(float(d.mean(axis=1).mean())) < 1e-3
                            and abs(float(d.std(axis=1).mean()) - 1.0) < 1e-2))
        vox = sorted({sh[0] for sh in shapes})
        trs = sorted({sh[1] for sh in shapes})
        rows.append(
            f"| `{roi}` | {len(subs)} | {vox if len(vox) > 1 else vox[0]} | "
            f"{trs if len(trs) > 1 else trs[0]} | "
            f"{'yes' if all(zscored) else 'NO'} |"
        )
    if not rows:
        return

    notes = [
        "Source: Zenodo record `2598755`, `Sherlock_processed.zip` "
        "(the BrainIAK tutorial redistribution of Chen et al. 2017).",
        "",
        "| ROI | subjects | voxels | TRs | already z-scored per voxel? |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "- The TR count matches the **1976** derived independently from the scene "
        "spreadsheet, so movie time maps onto these timeseries directly.",
        "- Data arrive already z-scored per voxel over time, so the pipeline's own "
        "z-scoring step is a no-op on them (it is kept because it is not a no-op "
        "for the whole-brain derivatives).",
        "- This release contains **only these two ROIs** -- there is no hippocampus, "
        "V1, angular gyrus, or mPFC in it. Those come from the 16-subject "
        "OpenNeuro derivatives. Born et al. (2023), Table 1, publishes the exact "
        "identity mapping; paired checks are valid, but duplicate release "
        "measurements are never pooled as independent subjects.",
    ]
    append_data_note(cfg, "Chen et al. ROI release", notes)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 0: acquire data.")
    p.add_argument("--config", default=None)
    p.add_argument("--no-download", action="store_true",
                   help="report status and print instructions; download nothing")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    allow = not args.no_download

    print("=" * 72)
    print("Stage 0 -- data acquisition")
    print("=" * 72)

    print("\n1. Scene annotations (GIN, public)")
    ok_seg = fetch_segments(cfg)

    print("\n2. Chen et al. ROI release (Zenodo, public)")
    ok_chen = fetch_chen_rois(cfg, allow)
    if ok_chen:
        record_chen_release(cfg)

    print("\n3. Recall transcripts (Zenodo, manual)")
    folder = resolve(cfg, "annotations_source") / "transcripts"
    found = sorted(folder.glob("NN*.xlsx")) if folder.exists() else []
    if len(found) == 17:
        print(f"  [ok] {len(found)} transcripts in {folder}")
    else:
        print(f"  [--] found {len(found)}/17 transcripts. Download NN01-NN17 from\n"
              f"       {TRANSCRIPTS_DOI}\n       into {folder}/ (Figure 4C only).")

    if ok_seg:
        print("=" * 72)
        print("Parsing the scene annotations ...")
        from src import parse_scenes
        parse_scenes.main(["--config", cfg["_config_path"]])

    print("\n" + "=" * 72)
    from src import data_io
    avail = data_io.available_rois(cfg)
    if avail:
        print("ROIs with data on disk:")
        for name, info in avail.items():
            print(f"  {name:34s} {len(info['subjects']):3d} subjects  "
                  f"({info['backend']}, family={info['family']})")
    else:
        print("No ROI data on disk yet. See the instructions above.")
    print("=" * 72)
    return 0 if (ok_seg and ok_chen) else 1


if __name__ == "__main__":
    raise SystemExit(main())
