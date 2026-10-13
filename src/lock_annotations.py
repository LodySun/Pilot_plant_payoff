"""Freeze the annotation file by recording its SHA256.

The pilot's order of operations is: annotate -> lock -> build controls ->
compute real-pair statistics. The lock is what makes that order auditable:
if the annotations change afterwards, every downstream stage fails loudly
rather than quietly mixing old results with new labels.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import REPO_ROOT, load_config, sha256_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Lock the annotation file.")
    p.add_argument("--config", default=None)
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing lock with a different hash")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    ann = REPO_ROOT / cfg["annotations"]["path"]
    lock = REPO_ROOT / cfg["annotations"]["lock_path"]

    if not ann.exists():
        print(f"ERROR: no annotation file at {ann}", file=sys.stderr)
        return 2

    digest = sha256_file(ann)

    if lock.exists():
        old = lock.read_text().split()[0].strip()
        if old == digest:
            print(f"Already locked, unchanged: {digest[:16]}...")
            return 0
        if not args.force:
            print(
                "ERROR: a lock already exists with a DIFFERENT hash.\n"
                f"  locked: {old}\n"
                f"  actual: {digest}\n"
                "\nThe annotations changed after locking. Any results in results/main/\n"
                "were computed against the old annotations and are now invalid.\n"
                "If the change is intentional:\n"
                "  1. rm -rf results/main/* results/robustness/* figures/*\n"
                "  2. make lock-annotations FORCE=1  (or: python -m src.lock_annotations --force)\n"
                "  3. re-run: make controls analysis robustness figures",
                file=sys.stderr,
            )
            return 1
        print(f"--force: replacing lock {old[:16]}... with {digest[:16]}...")

    n_rows = max(sum(1 for _ in open(ann)) - 1, 0)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lock.write_text(
        f"{digest}  {ann.name}\n"
        f"# locked      : {stamp}\n"
        f"# data rows   : {n_rows}\n"
        f"# Do not edit. Regenerate with `make lock-annotations`.\n"
        f"# Changing annotations/pairs.csv after this point invalidates every\n"
        f"# result in results/main/ and results/robustness/.\n"
    )
    print(f"Locked {ann.relative_to(REPO_ROOT)} ({n_rows} rows)")
    print(f"  SHA256: {digest}")
    print(f"  Lock:   {lock.relative_to(REPO_ROOT)}")
    print("\nControls and analysis may now run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
