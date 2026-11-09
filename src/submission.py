"""Build and validate the competition CSV.

Format (one row per row of test_images.csv):
    image_id,PredictionString
    clip_001_0000001,2 0.91 1366 580 26 44 0 0.55 809 625 29 66
PredictionString = groups of `class score x y width height`, (x, y) = top-left corner in pixels
of the ORIGINAL image, class 0-based, at most 500 groups, empty string if no detections.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import NUM_CLASSES, clip_dets, topk_dets, xyxy_to_xywh

SUB_COLUMNS = ["image_id", "PredictionString"]


def dets_to_string(dets: np.ndarray, width: int, height: int, max_det: int = 500,
                   conf_thres: float = 0.001, decimals: int = 1) -> str:
    """Canonical (N,6) [x1,y1,x2,y2,score,cls0] -> 'cls score x y w h ...'."""
    # min_size = one rounding unit, so no box is written with width/height "0.0" after rounding
    d = clip_dets(np.asarray(dets, dtype=np.float32).reshape(-1, 6), width, height,
                  min_size=10.0 ** -decimals)
    d = topk_dets(d, k=max_det, conf_thres=conf_thres)
    if len(d) == 0:
        return ""
    xywh = xyxy_to_xywh(d[:, :4])
    parts = []
    for (x, y, w, h), s, c in zip(xywh.tolist(), d[:, 4].tolist(), d[:, 5].tolist()):
        parts.append(f"{int(c)} {s:.4f} {x:.{decimals}f} {y:.{decimals}f} {w:.{decimals}f} {h:.{decimals}f}")
    return " ".join(parts)


def make_submission(preds: dict, test_csv: str | Path, out_path: str | Path, max_det: int = 500,
                    conf_thres: float = 0.001, decimals: int = 1) -> pd.DataFrame:
    """Write the submission CSV for ALL rows of test_images.csv (missing keys -> empty string)."""
    test_df = pd.read_csv(test_csv)
    empty = np.zeros((0, 6), dtype=np.float32)
    strings = [
        dets_to_string(preds.get(r.image_id, empty), int(r.width), int(r.height),
                       max_det=max_det, conf_thres=conf_thres, decimals=decimals)
        for r in test_df.itertuples(index=False)
    ]
    sub = pd.DataFrame({"image_id": test_df["image_id"], "PredictionString": strings})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    n_missing = sum(k not in preds for k in test_df["image_id"])
    print(f"[submission] wrote {out_path} | rows={len(sub)} | images without preds={n_missing}")
    return sub


def validate_submission(sub_path: str | Path, test_csv: str | Path, max_det: int = 500,
                        tol: float = 0.5) -> bool:
    """Check the CSV against test_images.csv. Raises AssertionError with a clear message on failure.

    Checks: column names, every image_id present exactly once, no NaN (other than empty strings),
    groups of 6 numbers, <= max_det boxes, class ids in 0..7, score in [0, 1],
    positive w/h, boxes inside the image (with `tol` pixels tolerance for rounding).
    """
    sub = pd.read_csv(sub_path, keep_default_na=False, dtype={"PredictionString": str})
    test_df = pd.read_csv(test_csv)

    assert list(sub.columns) == SUB_COLUMNS, f"columns must be {SUB_COLUMNS}, got {list(sub.columns)}"
    assert not sub["image_id"].duplicated().any(), "duplicate image_id rows"
    missing = set(test_df["image_id"]) - set(sub["image_id"])
    extra = set(sub["image_id"]) - set(test_df["image_id"])
    assert not missing, f"{len(missing)} image_ids missing, e.g. {sorted(missing)[:3]}"
    assert not extra, f"{len(extra)} unknown image_ids, e.g. {sorted(extra)[:3]}"
    assert sub["image_id"].astype(str).str.len().gt(0).all(), "empty image_id"

    sizes = test_df.set_index("image_id")[["width", "height"]].to_dict("index")
    n_boxes = 0
    for image_id, s in zip(sub["image_id"], sub["PredictionString"]):
        s = str(s).strip()
        if s == "":
            continue
        vals = s.split()
        assert len(vals) % 6 == 0, f"{image_id}: number of values {len(vals)} is not a multiple of 6"
        arr = np.asarray(vals, dtype=np.float64).reshape(-1, 6)
        assert np.isfinite(arr).all(), f"{image_id}: NaN/inf values"
        assert len(arr) <= max_det, f"{image_id}: {len(arr)} boxes > {max_det}"
        cls, score, x, y, w, h = arr.T
        assert np.all(cls == np.round(cls)) and cls.min() >= 0 and cls.max() < NUM_CLASSES, \
            f"{image_id}: class ids must be integers in 0..{NUM_CLASSES - 1}"
        assert score.min() >= 0 and score.max() <= 1, f"{image_id}: scores outside [0, 1]"
        assert (w > 0).all() and (h > 0).all(), f"{image_id}: non-positive width/height"
        W, H = sizes[image_id]["width"], sizes[image_id]["height"]
        assert x.min() >= -tol and y.min() >= -tol, f"{image_id}: negative x/y"
        assert (x + w).max() <= W + tol and (y + h).max() <= H + tol, f"{image_id}: box outside {W}x{H}"
        n_boxes += len(arr)

    print(f"[validate] OK | rows={len(sub)} | boxes={n_boxes} | "
          f"mean boxes/img={n_boxes / max(len(sub), 1):.1f}")
    return True
