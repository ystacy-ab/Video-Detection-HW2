"""Ensembling of saved canonical predictions (CPU only).

Weighted Boxes Fusion: Solovyev et al., "Weighted boxes fusion: Ensembling boxes from different object
detection models", Image and Vision Computing 2021. Implementation: `ensemble-boxes`
(https://github.com/ZFTurbo/Weighted-Boxes-Fusion). WBF averages the coordinates of overlapping boxes of
the same class instead of discarding all but one (NMS), and lowers the score of boxes found by few models.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .utils import topk_dets

EMPTY = np.zeros((0, 6), np.float32)


def image_sizes(index: pd.DataFrame | None = None, test_csv=None) -> dict:
    """{key: (W, H)} from a frame index (val) and/or test_images.csv."""
    sizes = {}
    if index is not None:
        sizes.update({k: (int(w), int(h)) for k, w, h in zip(index["key"], index["width"], index["height"])})
    if test_csv is not None:
        df = pd.read_csv(test_csv)
        sizes.update({k: (int(w), int(h)) for k, w, h in zip(df["image_id"], df["width"], df["height"])})
    return sizes


def wbf(pred_list: list[dict], sizes: dict, weights=None, iou_thr: float = 0.6, skip_box_thr: float = 0.001,
        max_det: int = 500, per_model_top: int = 500, desc: str = "WBF") -> tuple[dict, float]:
    """Fuse several {key: (N,6)} dicts. Returns (fused preds, ms per image of the fusion itself)."""
    from ensemble_boxes import weighted_boxes_fusion

    keys = sorted(set().union(*[p.keys() for p in pred_list]))
    out, t0 = {}, time.perf_counter()
    for k in tqdm(keys, desc=desc, leave=False):
        W, H = sizes[k]
        scale = np.array([W, H, W, H], np.float32)
        boxes, scores, labels = [], [], []
        for p in pred_list:
            d = topk_dets(np.asarray(p.get(k, EMPTY), np.float32).reshape(-1, 6), k=per_model_top)
            boxes.append((d[:, :4] / scale).clip(0, 1).tolist())
            scores.append(d[:, 4].tolist())
            labels.append(d[:, 5].astype(int).tolist())
        if sum(len(s) for s in scores) == 0:
            out[k] = EMPTY.copy()
            continue
        b, s, l = weighted_boxes_fusion(boxes, scores, labels, weights=weights, iou_thr=iou_thr,
                                        skip_box_thr=skip_box_thr, conf_type="avg")
        d = np.concatenate([np.asarray(b, np.float32).reshape(-1, 4) * scale,
                            np.asarray(s, np.float32)[:, None], np.asarray(l, np.float32)[:, None]], 1)
        out[k] = topk_dets(d, k=max_det)
    ms = (time.perf_counter() - t0) / max(len(keys), 1) * 1000
    return out, ms
