"""Sentence embeddings for scene descriptions, computed once and cached.

Used by two stages: Family-B control matching (Stage 2) and the
wrong-time-frame check in annotation validation (Stage 1).
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import ensure_dir, resolve  # noqa: E402

CACHE_NAME = "scene_embeddings.npy"
MANIFEST_NAME = "scene_embeddings.manifest.txt"


def _model_name(cfg: dict) -> str:
    return cfg["controls"]["embedding_model"]


def _fingerprint(texts: list[str], model_name: str) -> str:
    h = hashlib.sha256(model_name.encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def load_scenes(cfg: dict) -> pd.DataFrame:
    path = resolve(cfg, "derived") / "scenes.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `make data` (src/parse_scenes.py) first."
        )
    return pd.read_csv(path)


def _encode(texts: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=64, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def scene_embeddings(cfg: dict, scenes: pd.DataFrame | None = None,
                     verbose: bool = True) -> tuple[pd.DataFrame, np.ndarray]:
    """Embeddings for every segment description, cached to data/derived/.

    The cache is keyed on the exact text list plus the model name, so an
    edited scene table or a changed model invalidates it automatically.
    """
    scenes = load_scenes(cfg) if scenes is None else scenes
    texts = scenes["description"].fillna("").astype(str).tolist()
    model_name = _model_name(cfg)
    fp = _fingerprint(texts, model_name)

    derived = ensure_dir(resolve(cfg, "derived"))
    cache, manifest = derived / CACHE_NAME, derived / MANIFEST_NAME

    if cache.exists() and manifest.exists():
        recorded = manifest.read_text().split()[0].strip()
        if recorded == fp:
            emb = np.load(cache)
            if emb.shape[0] == len(texts):
                if verbose:
                    print(f"  [cache] scene embeddings {emb.shape} "
                          f"({model_name})")
                return scenes, emb

    if verbose:
        print(f"  [..] encoding {len(texts)} scene descriptions with {model_name} ...")
    emb = _encode(texts, model_name)
    np.save(cache, emb)
    manifest.write_text(f"{fp}  {model_name}  n={len(texts)}  dim={emb.shape[1]}\n")
    if verbose:
        print(f"  [ok] cached {emb.shape} -> {cache.name}")
    return scenes, emb


def embed_texts(cfg: dict, texts: list[str]) -> np.ndarray:
    """Encode arbitrary text with the same model (not cached)."""
    return _encode(texts, _model_name(cfg))


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity for L2-normalised vectors."""
    return np.sum(a * b, axis=-1)
