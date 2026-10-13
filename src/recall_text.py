"""Recall-text helpers used by the spoken-content analysis (poster Figure 4C).

Copied verbatim from four modules of the original pipeline, keeping only the
functions this analysis needs (the standalone analyses those modules ran are
not part of this repository):

  run_recall        keyword stemming / keyword_cells / scene_of_time
  run_recall_binding binding_cells
  run_rsa_recall    semantic_recall_tables / percentile_against_others
  find_recurrence   load_index / local_context
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import load_movie_timeline, resolve  # noqa: E402
from src.embeddings import embed_texts  # noqa: E402


# ---- from run_recall.py --------------------------------------------------

def scene_of_time(scenes: pd.DataFrame, t: float) -> int | None:
    hit = scenes.index[(scenes["start_s"] <= t) & (scenes["end_s"] > t)]
    if not len(hit):
        return None
    val = scenes.at[hit[0], "scene_id"]
    return None if pd.isna(val) else int(val)
SUFFIXES = ("ing", "edly", "ies", "ied", "es", "ed", "ly", "s")

def _normalise(w: str) -> str:
    """Collapse a doubled final consonant, then a trailing silent 'e'."""
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "aeiou":
        w = w[:-1]
    # Only for longer words: "cane" -> "can" would fire on the modal "can",
    # which is everywhere in speech.
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w

def stem_variants(word: str) -> set[str]:
    """Plausible normalisations of one word.

    Picking a SINGLE stem cannot work with naive suffix rules: stripping
    "es" turns "canes" into "can" while "cane" stays "cane", and stripping
    only "s" turns "boxes" into "boxe" while "box" stays "box". Whichever
    rule is chosen, some ordinary plural fails to match its singular. So
    each word yields a small set of candidates instead, and two words match
    when their sets intersect. This keeps the spec's "keyword/stem matching
    only" -- no dependency, no real stemmer -- while being self-consistent.
    """
    # Keep digits: stripping them would turn "221B" into "b" and collapse
    # otherwise distinct keywords onto the same stem.
    w = re.sub(r"[^a-z0-9]", "", word.lower())
    if not w:
        return set()
    if len(w) <= 3:
        return {w}
    out = {w, _normalise(w)}
    for suf in SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            out.add(base)
            out.add(_normalise(base))
    return {v for v in out if v}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*|[0-9]+[A-Za-z]*")

def text_stems(text: str) -> set[str]:
    out: set[str] = set()
    for w in TOKEN_RE.findall(str(text)):
        out |= stem_variants(w)
    return out

def keyword_hit(keywords: list[str], text: str) -> bool:
    """True if any keyword (or a close variant) appears in the text.

    A multi-word keyword must have ALL its parts present; a single word needs
    only its own stem.
    """
    if not keywords:
        return False
    present = text_stems(text)
    for kw in keywords:
        parts = [stem_variants(w) for w in TOKEN_RE.findall(kw) if len(w) > 1]
        parts = [p for p in parts if p]
        if parts and all(p & present for p in parts):
            return True
    return False

def parse_keywords(value) -> list[str]:
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    return [k.strip() for k in re.split(r"[;,]", s) if k.strip()]

def keyword_cells(pairs: pd.DataFrame, scenes: pd.DataFrame,
                  transcripts: dict, mode: str) -> tuple[
                      pd.DataFrame, dict[str, list[str]], list[str], list[str]]:
    """Build human-keyword subject × pair cells without doing inference."""
    notes: list[str] = []
    kw = {str(r["pair_id"]): parse_keywords(r.get("plant_keywords", ""))
          for _, r in pairs.iterrows()}
    missing = [p for p, v in kw.items() if not v]
    if missing:
        notes.append(f"{len(missing)} of {len(kw)} pair(s) have no "
                     f"`plant_keywords` and are skipped: {', '.join(missing[:6])}"
                     + (" ..." if len(missing) > 6 else ""))
    usable = [p for p, v in kw.items() if v]
    if not usable:
        return pd.DataFrame(), kw, usable, notes + [
            "No pair has plant_keywords, so the keyword analysis cannot run. "
            "That column is annotation content and must be written by the "
            "annotator, not generated here."]

    plant_scene, payoff_scene = {}, {}
    for _, r in pairs.iterrows():
        pid = str(r["pair_id"])
        plant_scene[pid] = scene_of_time(scenes, float(r["plant_start_s"]))
        payoff_scene[pid] = scene_of_time(scenes, float(r["payoff_start_s"]))

    subjects = sorted({s for s, _ in transcripts})
    rows = []
    for pid in usable:
        for s in subjects:
            if mode == "whole-transcript":
                payoff_txt = transcripts.get((s, -1), "")
                plant_txt = payoff_txt
                recalled_payoff = bool(payoff_txt)
            else:
                payoff_txt = transcripts.get((s, payoff_scene[pid]), "")
                plant_txt = transcripts.get((s, plant_scene[pid]), "")
                recalled_payoff = bool(payoff_txt)
            rows.append({
                "pair_id": pid, "subject": s,
                "plant_scene": plant_scene[pid], "payoff_scene": payoff_scene[pid],
                "recalled_payoff_scene": int(recalled_payoff),
                "plant_mentioned_in_payoff_recall": int(keyword_hit(kw[pid], payoff_txt)),
                "plant_mentioned_in_plant_recall": int(keyword_hit(kw[pid], plant_txt)),
                "payoff_recall_words": len(TOKEN_RE.findall(payoff_txt)),
            })
    table = pd.DataFrame(rows)
    return table, kw, usable, notes

# ---- from run_recall_binding.py ------------------------------------------

def _pair_scenes(pairs: pd.DataFrame, scenes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in pairs.itertuples(index=False):
        plant_scene = scene_of_time(scenes, float(row.plant_start_s))
        payoff_scene = scene_of_time(scenes, float(row.payoff_start_s))
        if plant_scene is None or payoff_scene is None:
            raise ValueError(f"{row.pair_id}: moment is outside released scenes")
        if plant_scene == payoff_scene:
            raise ValueError(
                f"{row.pair_id}: plant and payoff share recall scene "
                f"{plant_scene}; semantic self-comparison would be invalid")
        rows.append({"pair_id": str(row.pair_id),
                     "plant_scene": int(plant_scene),
                     "payoff_scene": int(payoff_scene)})
    out = pd.DataFrame(rows)
    counts = out["plant_scene"].value_counts()
    out["shared_plant_scene"] = out["plant_scene"].map(counts).gt(1)
    return out

def binding_cells(cfg: dict, pairs: pd.DataFrame, scenes: pd.DataFrame,
                  transcripts: dict[tuple[int, int], str],
                  min_distractors: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build subject × pair semantic-distance and relative-binding cells."""
    pair_scenes = _pair_scenes(pairs, scenes)
    subjects = sorted({subject for subject, _scene in transcripts})
    unique_plant_scenes = sorted(pair_scenes["plant_scene"].unique())
    texts = list(dict.fromkeys(text for text in transcripts.values() if text))
    vectors = embed_texts(cfg, texts)
    lookup = dict(zip(texts, vectors))

    rows = []
    for pair in pair_scenes.itertuples(index=False):
        for subject in subjects:
            plant_text = transcripts.get((subject, pair.plant_scene), "")
            payoff_text = transcripts.get((subject, pair.payoff_scene), "")
            if not plant_text or not payoff_text:
                continue
            correct = float(lookup[payoff_text] @ lookup[plant_text])
            distractors = []
            for other_scene in unique_plant_scenes:
                if other_scene == pair.plant_scene:
                    continue
                other_text = transcripts.get((subject, other_scene), "")
                if other_text:
                    distractors.append(float(lookup[payoff_text] @ lookup[other_text]))
            if len(distractors) < min_distractors:
                continue
            percentile = percentile_against_others(
                correct, np.asarray(distractors, dtype=float))
            rows.append({
                "subject": int(subject), "pair_id": pair.pair_id,
                "plant_scene": pair.plant_scene,
                "payoff_scene": pair.payoff_scene,
                "shared_plant_scene": bool(pair.shared_plant_scene),
                "plant_recall_words": len(plant_text.split()),
                "payoff_recall_words": len(payoff_text.split()),
                "n_distractor_plant_scenes": len(distractors),
                "paired_cosine_similarity": correct,
                "semantic_distance": 1.0 - correct,
                "binding_percentile": percentile,
                "binding_score": 2.0 * percentile - 1.0,
            })
    return pd.DataFrame(rows), pair_scenes

# ---- from run_rsa_recall.py ----------------------------------------------

def percentile_against_others(correct: float, others: np.ndarray) -> float:
    """Midrank percentile of one similarity against mismatched plants."""
    values = np.asarray(others, dtype=float)
    if not len(values):
        return np.nan
    return float(((values < correct).sum() + 0.5 * (values == correct).sum())
                 / len(values))

def _pair_contexts(pairs: pd.DataFrame, scenes: pd.DataFrame,
                   radius: int) -> tuple[list[str], list[int], list[int]]:
    starts = scenes["start_s"].to_numpy(dtype=float)
    ends = scenes["end_s"].to_numpy(dtype=float)
    contexts: list[str] = []
    plant_scenes: list[int] = []
    payoff_scenes: list[int] = []
    for row in pairs.itertuples(index=False):
        hit = np.flatnonzero(
            (starts <= float(row.plant_start_s))
            & (ends > float(row.plant_start_s)))
        if len(hit) != 1:
            raise ValueError(f"{row.pair_id}: plant maps to {len(hit)} segments")
        contexts.append(local_context(scenes, int(hit[0]), radius))
        plant_scene = scene_of_time(scenes, float(row.plant_start_s))
        payoff_scene = scene_of_time(scenes, float(row.payoff_start_s))
        if plant_scene is None or payoff_scene is None:
            raise ValueError(f"{row.pair_id}: moment is outside released scenes")
        plant_scenes.append(plant_scene)
        payoff_scenes.append(payoff_scene)
    return contexts, plant_scenes, payoff_scenes

def semantic_recall_tables(
    cfg: dict, pairs: pd.DataFrame, scenes: pd.DataFrame,
    transcripts: dict[tuple[int, int], str], context_radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Payoff-recall outcome and own-plant-scene validity tables."""
    contexts, plant_scenes, payoff_scenes = _pair_contexts(
        pairs, scenes, context_radius)
    subjects = sorted({subject for subject, _scene in transcripts})
    transcript_texts = [text for text in transcripts.values() if text]
    unique_texts = list(dict.fromkeys(contexts + transcript_texts))
    embeddings = embed_texts(cfg, unique_texts)
    lookup = dict(zip(unique_texts, embeddings))
    plant_embeddings = np.vstack([lookup[text] for text in contexts])

    def score(scene_ids: list[int], source: str) -> pd.DataFrame:
        rows = []
        for pair_i, pair in enumerate(pairs.itertuples(index=False)):
            for subject in subjects:
                text = transcripts.get((subject, scene_ids[pair_i]), "")
                if not text:
                    continue
                similarities = plant_embeddings @ lookup[text]
                correct = float(similarities[pair_i])
                others = np.delete(similarities, pair_i)
                rows.append({
                    "subject": int(subject), "pair_id": str(pair.pair_id),
                    "recall_source": source, "recall_scene": scene_ids[pair_i],
                    "n_words": len(text.split()),
                    "correct_plant_similarity": correct,
                    "wrong_plant_mean_similarity": float(np.mean(others)),
                    "plant_semantic_percentile": percentile_against_others(
                        correct, others),
                })
        return pd.DataFrame(rows)

    return score(payoff_scenes, "payoff_scene"), score(
        plant_scenes, "plant_scene_validity")

# ---- from find_recurrence.py ---------------------------------------------

def episode_time(row, timeline) -> float | None:
    offs = timeline["episode_offset_by_part"]
    for span in timeline.get("cartoon_spans", []):
        if float(span["start_s"]) <= row.start_s < float(span["end_s"]):
            return None
    return float(row.start_s) - float(offs[str(int(row.part))])

def load_index(cfg: dict):
    scenes = pd.read_csv(resolve(cfg, "derived") / "scenes.csv")
    # Embeddings are cached in the original scenes.csv row order.  Preserve
    # that mapping through the no-TR/cartoon filtering below.
    scenes["embedding_index"] = np.arange(len(scenes), dtype=int)
    timeline = load_movie_timeline(cfg)
    if timeline is None:
        raise FileNotFoundError("run `make data` first")
    scenes = scenes[scenes["start_tr"].notna()].copy()
    scenes["episode_s"] = [episode_time(r, timeline) for _, r in scenes.iterrows()]
    scenes = scenes[scenes["episode_s"].notna()].reset_index(drop=True)
    return scenes, timeline

def local_context(scenes: pd.DataFrame, index: int, radius: int) -> str:
    """Released descriptions around a hit, restricted to the same coarse scene."""
    row = scenes.iloc[index]
    parts = []
    for j in range(max(0, index - radius), min(len(scenes), index + radius + 1)):
        other = scenes.iloc[j]
        if other["scene_id"] == row["scene_id"]:
            parts.append(str(other["description"]))
    return " ".join(parts)
