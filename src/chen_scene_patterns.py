"""Scene-level movie and recall patterns for the Chen ROIs.

`extract` builds, for every subject, one pattern per Chen scene for the movie
and for free recall, with an optional extra haemodynamic shift. Moved
unchanged from run_shared_recall.py (whose own analysis is not part of this
repository); used by run_chen_timing_corrected.
"""
from __future__ import annotations

import numpy as np
from scipy.io import loadmat

from src import data_io
from src.common import load_movie_timeline
from src.data_quality import invalid_mask, load as load_invalid
from src.run_recall_bold import (
    load_chen_phase,
    load_event_release,
    subject_scene_blocks,
    one_based_blocks_pattern,
)


def unit(x):
    x = np.asarray(x, float)
    x = x - x.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.full_like(x, np.nan), where=norm > 1e-10)


def extract(cfg, roi, subjects, scene_ids, shift, normalize=True):
    timeline = load_movie_timeline(cfg)
    events = load_event_release(cfg)
    invalid = load_invalid(cfg)
    section = cfg['chen_recall_bold']
    ms, rs, audit = [], [], []
    expected = None
    for s in subjects:
        arrays = []
        for phase in ('movie', 'recall'):
            stem = data_io.CHEN_ROIS[roi][0]
            path = data_io.chen_root(cfg) / f's{s}/sherlock_{phase}/{stem}_sherlock_{phase}_s{s}.mat'
            raw = loadmat(path)
            kept = raw['keptvox'].ravel().astype(bool)
            if not kept.all() or len(kept) != raw['rdata'].shape[0]:
                raise ValueError('non-complete ROI mask: need explicit row intersection')
            expected = len(kept) if expected is None else expected
            if len(kept) != expected:
                raise ValueError('ROI row counts disagree across participants/phases')
            runs = [(0, int(timeline['part1_trs'])),
                    (int(timeline['part1_trs']), int(timeline['total_trs']))] if phase == 'movie' else None
            arrays.append(data_io.zscore_time(load_chen_phase(cfg, roi, s, phase), runs))
        movie, recall = arrays
        mb, rb = subject_scene_blocks(events[f's{s}'], int(timeline['part1_trs']), recall.shape[1])
        bad = invalid_mask(cfg, roi, s, movie.shape[1], invalid)
        mats = []
        for data, blocks, badmask in [(movie, mb, bad), (recall, rb, None)]:
            rows = []
            for scene in scene_ids:
                p = one_based_blocks_pattern(data, blocks.get(int(scene), []), shift,
                                             int(section['minimum_scene_trs']), badmask)
                rows.append((unit(p) if normalize else p) if p is not None else np.full(expected, np.nan))
            mats.append(np.stack(rows))
        ms.append(mats[0]); rs.append(mats[1])
        audit.append(dict(subject=s, roi=roi, shift=shift, voxels=expected,
                          movie_scenes=int(np.isfinite(mats[0]).all(axis=1).sum()),
                          recall_scenes=int(np.isfinite(mats[1]).all(axis=1).sum())))
    return np.stack(ms), np.stack(rs), audit
