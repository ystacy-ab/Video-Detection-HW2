"""Single evaluation function used by EVERY experiment (same metric, same split).

* COCO mAP@[.5:.95] with maxDets=500, via faster-coco-eval (falls back to pycocotools).
  - faster-coco-eval: https://github.com/MiXaiLL76/faster_coco_eval
  - COCO protocol:    Lin et al., "Microsoft COCO", ECCV 2014; https://github.com/cocodataset/cocoapi
* Ignored regions = GT annotations with iscrowd=1. COCOeval matches a detection that overlaps a
  crowd region (IoU computed as intersection / det-area) to it and then IGNORES that detection,
  so detections inside ignore regions are not counted as false positives.
  COCOeval does this per category, so every crowd region is replicated for all 8 categories
  (otherwise it would only ignore detections of the crowd box's own category_id).
* Id shift: canonical predictions use 0-based classes; COCO category_id = cls + 1.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np

from .utils import CLASS_NAMES, key_from_path, xyxy_to_xywh

try:  # fast C++ implementation, API-compatible with pycocotools
    from faster_coco_eval import COCO
    from faster_coco_eval import COCOeval_faster as COCOeval

    EVAL_BACKEND = "faster_coco_eval"
except ImportError:  # pragma: no cover
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    EVAL_BACKEND = "pycocotools"

AREA_NAMES = ["all", "small", "medium", "large"]  # COCO areaRng order


# ----------------------------------------------------------------------------- GT loading
def load_gt(gt_json: str | Path | dict) -> dict:
    """Load a COCO GT dict and replicate iscrowd=1 (ignore) regions for every category."""
    if isinstance(gt_json, dict):
        gt = copy.deepcopy(gt_json)
    else:
        with open(gt_json) as f:
            gt = json.load(f)

    cat_ids = [c["id"] for c in gt["categories"]]
    normal = [a for a in gt["annotations"] if not a.get("iscrowd", 0)]
    crowd = [a for a in gt["annotations"] if a.get("iscrowd", 0)]
    next_id = max((a["id"] for a in gt["annotations"]), default=0) + 1

    seen, expanded = set(), []
    for a in crowd:
        x, y, w, h = a["bbox"]
        for c in cat_ids:
            sig = (a["image_id"], c, round(x, 2), round(y, 2), round(w, 2), round(h, 2))
            if sig in seen:
                continue
            seen.add(sig)
            expanded.append({"id": next_id, "image_id": a["image_id"], "category_id": c,
                             "bbox": [x, y, w, h], "area": float(w * h), "iscrowd": 1})
            next_id += 1
    for a in normal:
        a.setdefault("area", float(a["bbox"][2] * a["bbox"][3]))
    gt["annotations"] = normal + expanded
    return gt


def build_key_to_image_id(gt: dict) -> dict:
    """Map '<clip>_<frame>' -> COCO image id (derived from images[].file_name)."""
    mapping = {key_from_path(img["file_name"]): img["id"] for img in gt["images"]}
    if len(mapping) != len(gt["images"]):
        raise ValueError("image keys derived from file_name are not unique - check file_name format")
    return mapping


def _make_coco(gt: dict):
    coco = COCO()
    coco.dataset = gt
    coco.createIndex()
    return coco


# ----------------------------------------------------------------------------- conversion
def preds_to_coco(preds: dict, key_to_id: dict) -> list[dict]:
    """{key: (N,6) [x1,y1,x2,y2,score,cls0]} -> COCO result list (category_id = cls + 1)."""
    out, missing = [], 0
    for key, dets in preds.items():
        if key not in key_to_id:
            missing += 1
            continue
        dets = np.asarray(dets, dtype=np.float32).reshape(-1, 6)
        if len(dets) == 0:
            continue
        xywh = xyxy_to_xywh(dets[:, :4])
        img_id = key_to_id[key]
        for (x, y, w, h), s, c in zip(xywh.tolist(), dets[:, 4].tolist(), dets[:, 5].tolist()):
            out.append({"image_id": img_id, "category_id": int(c) + 1,
                        "bbox": [x, y, w, h], "score": float(s)})
    if missing:
        print(f"[eval] warning: {missing} prediction keys not found in GT (ignored)")
    return out


# ----------------------------------------------------------------------------- evaluation
def _mean_valid(x: np.ndarray) -> float:
    x = x[x > -1]
    return float(x.mean()) if x.size else float("nan")


def evaluate(preds: dict, gt_json: str | Path | dict, max_dets: int = 500,
             img_keys: list[str] | None = None, verbose: bool = True) -> dict:
    """Evaluate canonical predictions against the COCO GT.

    Args:
        preds: {image_key: (N,6) [x1,y1,x2,y2,score,cls0]} in original pixels.
        gt_json: path to annotations_coco.json (or an already loaded dict).
        max_dets: detections per image that are scored (competition: 500).
        img_keys: evaluate only these images (e.g. a quick subset). Default: every GT image.

    Returns:
        dict with mAP, AP50, AP75, AP_small, AP_medium, AP_large, AR (@max_dets) and
        per-class AP under 'AP_<classname>'.
    """
    gt = load_gt(gt_json)
    key_to_id = build_key_to_image_id(gt)
    coco_gt = _make_coco(gt)

    results = preds_to_coco(preds, key_to_id)
    metrics = {k: 0.0 for k in ["mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR"]}
    if not results:
        print("[eval] no detections -> all metrics are 0")
        return metrics
    coco_dt = coco_gt.loadRes(results)

    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.maxDets = [1, 100, max_dets]  # last entry is the one used for every AP below
    if img_keys is not None:
        ev.params.imgIds = sorted(key_to_id[k] for k in img_keys if k in key_to_id)
    ev.evaluate()
    ev.accumulate()

    # precision: [T iou thr, R recall pts, K classes, A areas, M maxDets]; -1 = no GT
    prec = np.asarray(ev.eval["precision"])
    iou_thrs = np.asarray(ev.params.iouThrs)
    t50, t75 = int(np.argmin(np.abs(iou_thrs - 0.5))), int(np.argmin(np.abs(iou_thrs - 0.75)))
    m = len(ev.params.maxDets) - 1

    metrics["mAP"] = _mean_valid(prec[:, :, :, 0, m])
    metrics["AP50"] = _mean_valid(prec[t50, :, :, 0, m])
    metrics["AP75"] = _mean_valid(prec[t75, :, :, 0, m])
    for a, name in enumerate(AREA_NAMES[1:], start=1):
        metrics[f"AP_{name}"] = _mean_valid(prec[:, :, :, a, m])
    recall = np.asarray(ev.eval["recall"])  # [T, K, A, M]
    metrics["AR"] = _mean_valid(recall[:, :, 0, m])

    cat_ids = list(ev.params.catIds)
    for k, cid in enumerate(cat_ids):
        name = CLASS_NAMES[cid - 1] if 1 <= cid <= len(CLASS_NAMES) else str(cid)
        metrics[f"AP_{name}"] = _mean_valid(prec[:, :, k, 0, m])

    if verbose:
        print(f"[eval:{EVAL_BACKEND}] mAP={metrics['mAP']:.4f} AP50={metrics['AP50']:.4f} "
              f"AP75={metrics['AP75']:.4f} APs={metrics['AP_small']:.4f} "
              f"APm={metrics['AP_medium']:.4f} APl={metrics['AP_large']:.4f}")
    return metrics


def per_class_table(metrics: dict):
    """Per-class AP as a small pandas DataFrame (for the report tables)."""
    import pandas as pd

    return pd.DataFrame({"class": CLASS_NAMES,
                         "AP": [metrics.get(f"AP_{c}", float("nan")) for c in CLASS_NAMES]})


# ----------------------------------------------------------------------------- speed
def measure_speed(predict_fn, inputs: list, warmup: int = 10, iters: int = 100) -> dict:
    """Time ``predict_fn(x)`` per image with a fixed warm-up (GPU-synchronised).

    ``inputs`` is a list of whatever predict_fn takes (paths or arrays); it is cycled if short.
    Includes everything predict_fn does (pre-processing, model, NMS), i.e. end-to-end latency.
    """
    try:
        import torch

        sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    except ImportError:
        sync = lambda: None  # noqa: E731

    for i in range(warmup):
        predict_fn(inputs[i % len(inputs)])
    sync()
    t0 = time.perf_counter()
    for i in range(iters):
        predict_fn(inputs[i % len(inputs)])
    sync()
    ms = (time.perf_counter() - t0) / iters * 1000
    return {"ms_per_img": ms, "fps": 1000.0 / ms}


# ----------------------------------------------------------------------------- grouped evaluation
def evaluate_groups(preds: dict, gt_json: str | Path | dict, groups: dict, max_dets: int = 500):
    """mAP / AP50 / AP_small on subsets of images. groups = {name: [image_key, ...]} -> DataFrame."""
    import pandas as pd

    gt = gt_json if isinstance(gt_json, dict) else json.load(open(gt_json))
    rows = []
    for name, keys in groups.items():
        m = evaluate({k: preds[k] for k in keys if k in preds}, gt, max_dets=max_dets, img_keys=list(keys),
                     verbose=False)
        rows.append({"group": name, "n_images": len(keys), "mAP": m["mAP"], "AP50": m["AP50"],
                     "AP_small": m["AP_small"]})
    return pd.DataFrame(rows)


def clip_groups(gt_json: str | Path | dict) -> dict:
    """{clip: [image keys]} from the GT json (per-clip mAP shows how noisy a 7-clip val score is)."""
    gt = gt_json if isinstance(gt_json, dict) else json.load(open(gt_json))
    out = {}
    for img in gt["images"]:
        out.setdefault(Path(img["file_name"]).parent.name, []).append(key_from_path(img["file_name"]))
    return out


def density_groups(gt_json: str | Path | dict, bins=(0, 20, 50, 100, 10_000)) -> dict:
    """{'0-19 obj': keys, ...}: images grouped by number of (non-ignore) GT objects (scene density)."""
    gt = gt_json if isinstance(gt_json, dict) else json.load(open(gt_json))
    counts = {img["id"]: 0 for img in gt["images"]}
    for a in gt["annotations"]:
        if not a.get("iscrowd", 0):
            counts[a["image_id"]] += 1
    id2key = {img["id"]: key_from_path(img["file_name"]) for img in gt["images"]}
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        name = f"{lo}-{hi - 1} obj" if hi < 10_000 else f">={lo} obj"
        out[name] = [id2key[i] for i, n in counts.items() if lo <= n < hi]
    return {k: v for k, v in out.items() if v}
