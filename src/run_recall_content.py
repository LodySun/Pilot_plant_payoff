"""Does Chen recall-BOLD plant identity track plant content being spoken?

This is an interpretive follow-up to ``src.run_recall_bold``.  It uses only
the locked human-written relations and keywords.  Transcript NN numbers map
directly to Princeton DataSpace/Chen subjects according to the published word
timestamp data paper; the separate 16-subject mapping is not used here.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.build_controls import load_pairs  # noqa: E402
from src.common import (  # noqa: E402
    REPO_ROOT, ensure_dir, load_config, rel, require_lock_or_exit, resolve,
    sha256_file, stable_seed,
)
from src.prepare_transcripts import load_words  # noqa: E402
from src.run_recall_bold import load_event_release  # noqa: E402
from src.recall_text import binding_cells, keyword_cells, load_index, semantic_recall_tables  # noqa: E402
from src.stats_utils import fmt_p  # noqa: E402


def transcript_files(cfg: dict) -> dict[int, Path]:
    """Direct Princeton subject -> NN transcript files (1..17)."""
    folder = resolve(cfg, "annotations_source") / "transcripts"
    found: dict[int, Path] = {}
    for name in glob.glob(str(folder / "NN*.xlsx")):
        match = re.search(r"NN(\d+)", Path(name).name)
        if match:
            found[int(match.group(1))] = Path(name)
    expected = set(range(1, 18))
    if set(found) != expected:
        raise FileNotFoundError(
            f"expected NN01-NN17 in {rel(folder)}; found {sorted(found)}"
        )
    return found


def chen_scene_transcripts(
    cfg: dict, event_release: dict
) -> tuple[dict[tuple[int, int], str], pd.DataFrame]:
    """Slice each direct-mapped NN transcript with its released scene blocks."""
    files = transcript_files(cfg)
    texts: dict[tuple[int, int], str] = {}
    rows = []
    for subject in range(1, 18):
        words = load_words(files[subject])
        events = event_release[f"s{subject}"]["events"]
        bounds = np.atleast_2d(np.asarray(events["freerecall"], dtype=int))
        labels = np.asarray(events["freerecallnum"], dtype=int).ravel()
        if len(bounds) != len(labels):
            raise ValueError(f"s{subject}: recall bounds and labels differ")
        for scene in sorted(set(int(value) for value in labels)):
            blocks = bounds[labels == scene]
            inside = np.zeros(len(words), dtype=bool)
            for start, end in blocks:
                inside |= ((words["tr"].to_numpy() >= int(start))
                           & (words["tr"].to_numpy() <= int(end)))
            selected = words.loc[inside, "word"].astype(str).tolist()
            text = " ".join(selected).strip()
            if text:
                texts[(subject, scene)] = text
            rows.append({
                "subject": subject,
                "transcript_id": f"NN{subject:02d}",
                "scene": scene,
                "n_blocks": int(len(blocks)),
                "start_tr": int(blocks[:, 0].min()),
                "end_tr": int(blocks[:, 1].max()),
                "n_words": int(len(selected)),
                "text": text,
            })
    return texts, pd.DataFrame(rows)


def content_cells(
    cfg: dict,
    pairs: pd.DataFrame,
    scenes: pd.DataFrame,
    transcripts: dict[tuple[int, int], str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keyword, plant-context semantic, and recalled-scene binding measures."""
    keyword_all, _keywords, usable, _notes = keyword_cells(
        pairs, scenes, transcripts, "per-scene"
    )
    keyword_all["recalled_plant_scene"] = [
        int((int(row.subject), int(row.plant_scene)) in transcripts)
        for row in keyword_all.itertuples(index=False)
    ]
    if len(usable) != len(pairs):
        raise ValueError(
            "locked plant_keywords are required for every pair; "
            f"found {len(usable)} of {len(pairs)}"
        )
    keyword = keyword_all[keyword_all["recalled_payoff_scene"] == 1].copy()
    keyword = keyword.rename(columns={
        "plant_mentioned_in_payoff_recall": "keyword_hit",
        "payoff_recall_words": "n_words",
    })

    semantic, semantic_validity = semantic_recall_tables(
        cfg, pairs, scenes, transcripts,
        int(cfg["chen_recall_content"]["context_radius"]),
    )
    semantic = semantic.rename(columns={
        "plant_semantic_percentile": "semantic_percentile"
    })

    binding, _pair_scenes = binding_cells(
        cfg, pairs, scenes, transcripts,
        int(cfg["recall_binding"]["min_distractors"]),
    )
    compact = keyword[[
        "subject", "pair_id", "plant_scene", "payoff_scene", "n_words",
        "keyword_hit", "plant_mentioned_in_plant_recall",
    ]].merge(
        semantic[[
            "subject", "pair_id", "correct_plant_similarity",
            "wrong_plant_mean_similarity", "semantic_percentile",
        ]], on=["subject", "pair_id"], how="outer", validate="one_to_one"
    ).merge(
        binding[["subject", "pair_id", "binding_percentile"]],
        on=["subject", "pair_id"], how="left", validate="one_to_one",
    )
    return compact, keyword_all, semantic_validity, binding


def _residualise(design: np.ndarray, values: np.ndarray) -> np.ndarray:
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def clustered_association(
    table: pd.DataFrame,
    predictor: str,
    repetitions: int,
    bootstrap_repetitions: int,
    seed: int,
    minimum_cells: int,
) -> dict:
    """Two-way-FE partial r with subject-cluster sign flips and bootstrap."""
    needed = {
        "subject", "pair_id", "identity_percentile", "n_words", predictor
    }
    if missing := needed - set(table):
        raise ValueError(f"association table missing {sorted(missing)}")
    frame = table.dropna(subset=list(needed)).copy()
    if len(frame) < int(minimum_cells):
        raise ValueError(f"{predictor}: only {len(frame)} usable cells")

    subject_dummies = pd.get_dummies(
        frame["subject"].astype(str), drop_first=True, dtype=float
    ).to_numpy()
    pair_dummies = pd.get_dummies(
        frame["pair_id"].astype(str), drop_first=True, dtype=float
    ).to_numpy()
    design = np.column_stack([
        np.ones(len(frame)), subject_dummies, pair_dummies,
        np.log1p(frame["n_words"].to_numpy(dtype=float)),
    ])
    neural = frame["identity_percentile"].to_numpy(dtype=float)
    content = frame[predictor].to_numpy(dtype=float)
    neural_residual = _residualise(design, neural)
    content_residual = _residualise(design, content)
    denominator = np.sqrt(
        float(np.dot(neural_residual, neural_residual)
              * np.dot(content_residual, content_residual))
    )
    if denominator <= 0:
        raise ValueError(f"{predictor}: no residual variation")
    observed = float(np.dot(neural_residual, content_residual) / denominator)

    subjects = sorted(frame["subject"].unique(), key=int)
    subject_index = frame["subject"].map(
        {subject: index for index, subject in enumerate(subjects)}
    ).to_numpy(dtype=int)
    xy = np.bincount(
        subject_index, neural_residual * content_residual,
        minlength=len(subjects),
    )
    x2 = np.bincount(
        subject_index, content_residual ** 2, minlength=len(subjects)
    )
    y2 = np.bincount(
        subject_index, neural_residual ** 2, minlength=len(subjects)
    )
    rng = np.random.default_rng(int(seed))
    signs = rng.choice((-1.0, 1.0), size=(int(repetitions), len(subjects)))
    null = (signs @ xy) / denominator
    p_value = float(
        (np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)
    )

    draws = rng.integers(
        0, len(subjects), size=(int(bootstrap_repetitions), len(subjects))
    )
    boot_xy = xy[draws].sum(axis=1)
    boot_denominator = np.sqrt(x2[draws].sum(axis=1) * y2[draws].sum(axis=1))
    boot = np.divide(
        boot_xy, boot_denominator,
        out=np.full(int(bootstrap_repetitions), np.nan),
        where=boot_denominator > 0,
    )
    ci_low, ci_high = np.nanquantile(boot, [0.025, 0.975])

    augmented = np.column_stack([design, content])
    slope = float(np.linalg.lstsq(augmented, neural, rcond=None)[0][-1])
    return {
        "n_cells": int(len(frame)),
        "n_subjects": int(frame["subject"].nunique()),
        "n_pairs": int(frame["pair_id"].nunique()),
        "n_variable_pairs": int(
            frame.groupby("pair_id")[predictor].nunique().gt(1).sum()
        ),
        "content_mean": float(content.mean()),
        "partial_r": observed,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_cluster_signflip": p_value,
        "adjusted_slope": slope,
        "n_permutations": int(repetitions),
        "n_bootstrap": int(bootstrap_repetitions),
    }


def association_tests(cfg: dict, neural: pd.DataFrame,
                      content: pd.DataFrame) -> pd.DataFrame:
    section = cfg["chen_recall_content"]
    measures = ["keyword_hit", "semantic_percentile", "binding_percentile"]
    rows = []
    for roi in [section["primary_roi"], section["control_roi"]]:
        merged = neural[neural["roi"] == roi].merge(
            content, on=["subject", "pair_id"], validate="one_to_one"
        )
        for measure in measures:
            result = clustered_association(
                merged, measure, int(section["permutations"]),
                int(section["bootstrap_repetitions"]),
                stable_seed("chen-recall-content", roi, measure, int(cfg["seed"])),
                int(section["minimum_cells"]),
            )
            rows.append({
                "roi": roi,
                "content_measure": measure,
                "is_frozen_primary": bool(
                    roi == section["primary_roi"] and measure == "keyword_hit"
                ),
                **result,
            })
    out = pd.DataFrame(rows)
    out["q_fdr_secondary"] = np.nan
    out["significant_fdr_secondary"] = False
    secondary = ~out["is_frozen_primary"]
    reject, adjusted, _, _ = multipletests(
        out.loc[secondary, "p_cluster_signflip"], method="fdr_bh"
    )
    out.loc[secondary, "q_fdr_secondary"] = adjusted
    out.loc[secondary, "significant_fdr_secondary"] = reject
    return out


def write_summary(
    path: Path, content: pd.DataFrame, keyword_all: pd.DataFrame,
    semantic_validity: pd.DataFrame, tests: pd.DataFrame,
) -> None:
    primary = tests[tests["is_frozen_primary"]].iloc[0]
    target_roi = str(primary.roi)
    semantic = tests[
        (tests.roi == target_roi)
        & (tests.content_measure == "semantic_percentile")
    ].iloc[0]
    binding = tests[
        (tests.roi == target_roi)
        & (tests.content_measure == "binding_percentile")
    ].iloc[0]
    recalled_plants = keyword_all[keyword_all["recalled_plant_scene"] == 1]
    keyword_validity = recalled_plants[
        "plant_mentioned_in_plant_recall"
    ].mean()
    plant_recall_coverage = keyword_all["recalled_plant_scene"].mean()
    lines = [
        "# Does spoken plant content explain Chen recall-BOLD identity?", "",
        "> **Exploratory explanatory analysis.** It does not alter the locked",
        "> relations or keywords, and association cannot establish causal direction.",
        "", "## Frozen primary content test", "",
        "PMC plant-identity percentile was related to whether the subject actually",
        "said a locked plant keyword while recalling that payoff scene. Subject and",
        "pair fixed effects and log word count were removed; inference sign-flipped",
        "whole-subject residual cross-products.", "",
        "| cells | subjects | varying pairs | keyword hit rate | partial r (95% cluster CI) | p | adjusted hit−no-hit |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        f"| {int(primary.n_cells)} | {int(primary.n_subjects)} | {int(primary.n_variable_pairs)} | {primary.content_mean:.1%} | {primary.partial_r:+.3f} ({primary.ci_low:+.3f}, {primary.ci_high:+.3f}) | {fmt_p(primary.p_cluster_signflip)} | {primary.adjusted_slope:+.3f} |",
        "", "The last column is the adjusted difference in neural identity percentile",
        "for keyword-hit versus no-hit cells.", "",
        "## Continuous sensitivities in PMC", "",
        "| content measure | cells | partial r (95% cluster CI) | p | secondary q |",
        "|---|---:|---:|---:|---:|",
        f"| payoff speech → correct plant movie-context percentile | {int(semantic.n_cells)} | {semantic.partial_r:+.3f} ({semantic.ci_low:+.3f}, {semantic.ci_high:+.3f}) | {fmt_p(semantic.p_cluster_signflip)} | {fmt_p(semantic.q_fdr_secondary)} |",
        f"| recalled plant ↔ recalled payoff binding percentile | {int(binding.n_cells)} | {binding.partial_r:+.3f} ({binding.ci_low:+.3f}, {binding.ci_high:+.3f}) | {fmt_p(binding.p_cluster_signflip)} | {fmt_p(binding.q_fdr_secondary)} |",
        "", "## All pre-specified associations", "",
        "| ROI | content measure | cells | partial r | p | secondary q |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in tests.itertuples(index=False):
        q = "—" if row.is_frozen_primary else fmt_p(row.q_fdr_secondary)
        lines.append(
            f"| `{row.roi}` | `{row.content_measure}` | {row.n_cells} | "
            f"{row.partial_r:+.3f} | {fmt_p(row.p_cluster_signflip)} | {q} |"
        )
    lines += [
        "", "## Interpretation", "",
        "None of the three content measures showed a reliable positive PMC",
        "association. The explicit keyword and plant-context embedding estimates",
        "were slightly negative; recalled-scene binding was near zero. Therefore",
        "the observed PMC plant identity is not explained by a tendency to become",
        "stronger when plant content is overtly present in the payoff speech.", "",
        "This supports the claim that the neural result is not *merely* current",
        "spoken content, but it does not prove covert context reconstruction. In",
        f"particular, the semantic-content interval still permits r up to {semantic.ci_high:+.3f}",
        f"and the recall-binding interval permits r up to {binding.ci_high:+.3f}.",
        "", "## Measurement validity and limits", "",
        f"- A plant scene was recalled in {plant_recall_coverage:.1%} of subject × pair cells; conditional on that scene being recalled, its locked keyword occurred in {keyword_validity:.1%}.",
        f"- The correct plant-context embedding had mean own-plant-scene validity percentile {semantic_validity.plant_semantic_percentile.mean():.3f} (median {semantic_validity.plant_semantic_percentile.median():.3f}).",
        "- A null association does not prove covert reconstruction: the interval,",
        "  sparse keyword variation, coarse scene labels, and 17 clusters determine",
        "  how strongly that interpretation can be defended.",
        "- A positive association is compatible with currently spoken content",
        "  contributing to PMC identity, but cannot prove that speech caused it.",
        "- NN01–NN17 map directly to Princeton DataSpace subjects 01–17 according",
        "  to the published timestamp data paper (doi:10.1016/j.dib.2023.109490).",
        "  The unrelated 16-subject mapping is not used.", "",
        "Full outputs: `content_cells.csv`, `association_tests.csv`,",
        "`transcript_scene_cells.csv`, `semantic_validity_cells.csv`,",
        "`recall_binding_cells.csv`, and `meta.json`.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run_smoke(cfg: dict) -> int:
    rng = np.random.default_rng(612)
    rows = []
    for subject in range(1, 13):
        subject_offset = rng.normal(0, .04)
        for pair in range(1, 7):
            content = float(rng.normal())
            rows.append({
                "subject": str(subject), "pair_id": f"P{pair}",
                "identity_percentile": .5 + subject_offset + .12 * content
                                       + rng.normal(0, .05),
                "n_words": int(rng.integers(8, 80)), "content": content,
            })
    planted = clustered_association(
        pd.DataFrame(rows), "content", 2000, 2000, 1, 20
    )
    if not (planted["partial_r"] > .5 and planted["p_cluster_signflip"] < .05):
        raise AssertionError("smoke planted content association was not recovered")
    out = ensure_dir(resolve(cfg, "results_smoke") / "chen_recall_content")
    (out / "summary.md").write_text(
        "# SMOKE TEST — SYNTHETIC DATA, NOT A BRAIN RESULT\n\n"
        f"Planted association recovered: r={planted['partial_r']:.3f}, "
        f"p={planted['p_cluster_signflip']:.4f}.\n"
    )
    print(f"Wrote {rel(out)} (synthetic planted association)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chen recall-BOLD identity versus spoken plant content"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    require_lock_or_exit(cfg, smoke=args.smoke_test)
    if args.smoke_test:
        return run_smoke(cfg)

    pairs = load_pairs(cfg, smoke=False)
    scenes, _timeline = load_index(cfg)
    transcripts, transcript_table = chen_scene_transcripts(
        cfg, load_event_release(cfg)
    )
    content, keyword_all, semantic_validity, binding = content_cells(
        cfg, pairs, scenes, transcripts
    )
    section = cfg["chen_recall_content"]
    # Neural outcome: the source-aligned (timing-corrected) recall-BOLD cells.
    # Spoken-content measures do not depend on neural timing.
    neural = pd.read_csv(REPO_ROOT / "results/chen_timing_corrected/pair_cells.csv")
    neural["subject"] = neural.subject.astype(int)
    content["subject"] = content.subject.astype(int)
    tests = association_tests(cfg, neural, content)

    out = ensure_dir(REPO_ROOT / section["results_dir"])
    transcript_table.to_csv(out / "transcript_scene_cells.csv", index=False)
    content.to_csv(out / "content_cells.csv", index=False)
    keyword_all.to_csv(out / "keyword_cells_all.csv", index=False)
    semantic_validity.to_csv(out / "semantic_validity_cells.csv", index=False)
    binding.to_csv(out / "recall_binding_cells.csv", index=False)
    tests.to_csv(out / "association_tests.csv", index=False)
    write_summary(out / "summary.md", content, keyword_all, semantic_validity, tests)
    annotation_path = REPO_ROOT / cfg["annotations"]["path"]
    meta = {
        "exploratory": True,
        "analysis": "recall-BOLD plant identity versus concurrently spoken plant content",
        "frozen_plan": "docs/chen_recall_content_plan.md",
        "neural_outcome": "results/chen_timing_corrected/pair_cells.csv (source-aligned)",
        "annotation_hash": sha256_file(annotation_path),
        "direct_subject_mapping": "NN01-NN17 -> Princeton DataSpace s1-s17",
        "mapping_source_doi": "10.1016/j.dib.2023.109490",
        "n_scene_transcript_rows": int(len(transcript_table)),
        "n_content_cells": int(len(content)),
        "permutations": int(section["permutations"]),
        "bootstrap_repetitions": int(section["bootstrap_repetitions"]),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    primary = tests[tests["is_frozen_primary"]].iloc[0]
    print(
        f"PMC keyword association: r={primary.partial_r:+.3f}, "
        f"p={primary.p_cluster_signflip:.4f}; wrote {rel(out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
