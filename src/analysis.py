"""Failure analysis on saved predictions (CPU only, works for every model's canonical preds).

Matching is greedy by score at IoU >= iou_thr and CLASS-AGNOSTIC, so a car predicted as a van is a
"wrong class" match (visible in the confusion matrix) instead of one FP + one FN. Detections that lie
that match no GT and lie mostly inside an ignore region are dropped (COCOeval ignores them too).
This is a diagnostic in the spirit of Hoiem et al., "Diagnosing Error in Object Detectors", ECCV 2012
and the TIDE toolbox (Bolya et al., ECCV 2020) - simplified to TP / FN / FP / wrong-class.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import CLASS_NAMES, key_from_path

BG = len(CLASS_NAMES)  # index used for "background" (FP) / "missed" (FN) in the confusion matrix
LABELS_CM = CLASS_NAMES + ["bg/missed"]


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (len(a), len(b)) for xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def ioa_np(a: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """Max over regions of intersection / area(a) for each box in a."""
    if len(a) == 0 or len(regions) == 0:
        return np.zeros(len(a), np.float32)
    lt = np.maximum(a[:, None, :2], regions[None, :, :2])
    rb = np.minimum(a[:, None, 2:], regions[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-9)
    return (inter / area[:, None]).max(1)


def size_bucket_np(area: np.ndarray) -> np.ndarray:
    """COCO buckets by area: small < 32^2 <= medium < 96^2 <= large."""
    return np.where(area < 32 ** 2, "small", np.where(area < 96 ** 2, "medium", "large"))


def gt_from_coco(gt_json) -> dict:
    """{key: {"boxes" xyxy (N,4), "cls" 0-based, "occ" (N,), "ignore" xyxy (M,4), "file_name"}}."""
    with open(gt_json) as f:
        coco = json.load(f)
    out = {}
    id2key = {}
    for img in coco["images"]:
        k = key_from_path(img["file_name"])
        id2key[img["id"]] = k
        out[k] = {"boxes": [], "cls": [], "occ": [], "ignore": [], "file_name": img["file_name"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        g = out[id2key[a["image_id"]]]
        if a.get("iscrowd", 0):
            g["ignore"].append([x, y, x + w, y + h])
        else:
            g["boxes"].append([x, y, x + w, y + h])
            g["cls"].append(a["category_id"] - 1)
            g["occ"].append(a.get("occlusion", -1))
    for g in out.values():
        g["boxes"] = np.asarray(g["boxes"], np.float32).reshape(-1, 4)
        g["ignore"] = np.unique(np.asarray(g["ignore"], np.float32).reshape(-1, 4), axis=0)
        g["cls"] = np.asarray(g["cls"], np.int64)
        g["occ"] = np.asarray(g["occ"], np.int64)
    return out


def match_frame(dets: np.ndarray, gt_boxes: np.ndarray, iou_thr: float = 0.5):
    """Greedy class-agnostic matching. Returns (det_to_gt (N,), gt_to_det (M,)), -1 = unmatched."""
    det_to_gt = -np.ones(len(dets), np.int64)
    gt_to_det = -np.ones(len(gt_boxes), np.int64)
    if len(dets) == 0 or len(gt_boxes) == 0:
        return det_to_gt, gt_to_det
    iou = box_iou_np(dets[:, :4], gt_boxes)
    for i in np.argsort(-dets[:, 4], kind="stable"):
        cand = np.where(gt_to_det < 0, iou[i], -1.0)
        j = int(cand.argmax())
        if cand[j] >= iou_thr:
            det_to_gt[i], gt_to_det[j] = j, i
    return det_to_gt, gt_to_det


def _match_with_ignore(d: np.ndarray, g: dict, iou_thr: float, ignore_ioa: float):
    """Match to real GT first (like COCOeval), then drop UNMATCHED detections inside ignore regions."""
    d2g, g2d = match_frame(d, g["boxes"], iou_thr)
    if len(g["ignore"]) and len(d):
        keep = (d2g >= 0) | (ioa_np(d[:, :4], g["ignore"]) <= ignore_ioa)
        new_idx = np.cumsum(keep) - 1
        g2d = np.where(g2d >= 0, new_idx[np.maximum(g2d, 0)], -1)
        d, d2g = d[keep], d2g[keep]
    return d, d2g, g2d


def analyse_predictions(preds: dict, gt: dict, conf: float = 0.3, iou_thr: float = 0.5,
                        ignore_ioa: float = 0.5):
    """Returns three DataFrames:
    frames: key, clip, n_gt, tp, wrong_cls, fn, fp, errors, error_rate
    gts:    one row per GT box: key, gt_idx, cls, size, bucket, occ, matched, pred_cls, score
    fps:    one row per unmatched detection: key, pred_cls, score, size
    """
    frames, gts, fps = [], [], []
    empty = np.zeros((0, 6), np.float32)
    for key, g in gt.items():
        d = np.asarray(preds.get(key, empty), np.float32).reshape(-1, 6)
        d = d[d[:, 4] >= conf]
        d, d2g, g2d = _match_with_ignore(d, g, iou_thr, ignore_ioa)
        wh = g["boxes"][:, 2:] - g["boxes"][:, :2]
        area = wh[:, 0] * wh[:, 1]
        pred_cls = np.where(g2d >= 0, d[np.maximum(g2d, 0), 5] if len(d) else -1, -1).astype(int)
        score = np.where(g2d >= 0, d[np.maximum(g2d, 0), 4] if len(d) else 0, 0.0)
        gts.append(pd.DataFrame({"key": key, "gt_idx": np.arange(len(g["cls"])), "cls": g["cls"], "size": np.sqrt(area), "bucket": size_bucket_np(area),
                                 "occ": g["occ"], "matched": g2d >= 0, "pred_cls": pred_cls, "score": score}))
        un = d[d2g < 0]
        if len(un):
            dwh = un[:, 2:4] - un[:, :2]
            fps.append(pd.DataFrame({"key": key, "pred_cls": un[:, 5].astype(int), "score": un[:, 4],
                                     "size": np.sqrt(np.clip(dwh[:, 0] * dwh[:, 1], 0, None))}))
        tp_any = int((g2d >= 0).sum())
        wrong = int(((g2d >= 0) & (pred_cls != g["cls"])).sum())
        fn = int((g2d < 0).sum())
        fp = int((d2g < 0).sum())
        frames.append({"key": key, "clip": Path(g["file_name"]).parent.name, "n_gt": len(g["boxes"]),
                       "tp": tp_any - wrong, "wrong_cls": wrong, "fn": fn, "fp": fp})
    frames = pd.DataFrame(frames)
    frames["errors"] = frames["wrong_cls"] + frames["fn"] + frames["fp"]
    frames["error_rate"] = frames["errors"] / frames["n_gt"].clip(lower=1)
    gts = pd.concat(gts, ignore_index=True) if gts else pd.DataFrame()
    fps = pd.concat(fps, ignore_index=True) if fps else pd.DataFrame(columns=["key", "pred_cls", "score", "size"])
    return frames, gts, fps


def confusion_matrix(gts: pd.DataFrame, fps: pd.DataFrame) -> np.ndarray:
    """(9,9) counts: rows = GT class (+ 'bg' for false positives), cols = predicted class (+ 'missed')."""
    cm = np.zeros((BG + 1, BG + 1), np.int64)
    m = gts["matched"].to_numpy()
    np.add.at(cm, (gts["cls"].to_numpy()[m], gts["pred_cls"].to_numpy()[m]), 1)
    np.add.at(cm, (gts["cls"].to_numpy()[~m], np.full((~m).sum(), BG)), 1)
    np.add.at(cm, (np.full(len(fps), BG), fps["pred_cls"].to_numpy().astype(int)), 1)
    return cm


def plot_confusion(cm: np.ndarray, title: str = "", normalize: bool = True, ax=None):
    """Row-normalised confusion matrix (each GT row sums to 1; the bg row is normalised too)."""
    import matplotlib.pyplot as plt

    m = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalize else cm
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(LABELS_CM)), LABELS_CM, rotation=45, ha="right")
    ax.set_yticks(range(len(LABELS_CM)), LABELS_CM[:-1] + ["background"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if cm[i, j]:
                ax.text(j, i, f"{m[i, j]:.2f}" if normalize else str(cm[i, j]), ha="center", va="center",
                        fontsize=7, color="white" if m[i, j] > 0.5 * m.max() else "black")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046)
    return ax


def recall_table(gts: pd.DataFrame, by: str) -> pd.DataFrame:
    """Recall (class-agnostic localisation, IoU>=thr) per class x `by` (e.g. 'bucket' or 'occ')."""
    t = gts.assign(name=gts["cls"].map(lambda c: CLASS_NAMES[c]))
    return t.pivot_table(index="name", columns=by, values="matched", aggfunc="mean").reindex(CLASS_NAMES)


# ----------------------------------------------------------------------------- drawing
COL_TP, COL_WRONG, COL_FN, COL_FP = (0, 200, 0), (255, 140, 0), (255, 0, 0), (255, 0, 255)  # RGB


def draw_errors(img_rgb: np.ndarray, dets: np.ndarray, g: dict, conf: float = 0.3, iou_thr: float = 0.5,
                ignore_ioa: float = 0.5, thickness: int = 2) -> np.ndarray:
    """TP green, wrong class orange, missed GT red, false positive magenta, ignore regions gray."""
    import cv2

    out = img_rgb.copy()
    for x1, y1, x2, y2 in np.round(g["ignore"]).astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)
    d, d2g, g2d = _match_with_ignore(dets[dets[:, 4] >= conf], g, iou_thr, ignore_ioa)
    for j, b in enumerate(np.round(g["boxes"]).astype(int)):
        if g2d[j] < 0:
            color = COL_FN
        else:
            color = COL_TP if int(d[g2d[j], 5]) == g["cls"][j] else COL_WRONG
        cv2.rectangle(out, tuple(b[:2]), tuple(b[2:]), color, thickness)
    for b in np.round(d[d2g < 0, :4]).astype(int):
        cv2.rectangle(out, tuple(b[:2]), tuple(b[2:]), COL_FP, thickness)
    return out


def error_legend():
    from matplotlib.patches import Patch

    return [Patch(color=np.array(c) / 255, label=n) for n, c in
            [("TP", COL_TP), ("wrong class", COL_WRONG), ("missed (FN)", COL_FN), ("false pos.", COL_FP)]]


def crop(img_rgb: np.ndarray, box, pad: float = 1.0, min_side: int = 48) -> np.ndarray:
    """Crop around a box with `pad` x box size of context on each side (for tiny objects)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    s = max(w, h) * (1 + 2 * pad) / 2
    s = max(s, min_side / 2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    H, W = img_rgb.shape[:2]
    a, b = int(max(cx - s, 0)), int(max(cy - s, 0))
    c, d = int(min(cx + s, W)), int(min(cy + s, H))
    return img_rgb[b:d, a:c]
