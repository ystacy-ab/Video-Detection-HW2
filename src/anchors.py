"""K-means anchor clustering on GT (w, h) with the 1 - IoU distance.

Source: Redmon & Farhadi, "YOLO9000: Better, Faster, Stronger" (YOLOv2), CVPR 2017, Sec. 2
("Dimension Clusters"): d(box, centroid) = 1 - IoU(box, centroid), boxes aligned at a common center.
"""
from __future__ import annotations

import numpy as np


def wh_iou(wh: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """IoU between (N,2) boxes and (K,2) centroids, all centered at the origin -> (N,K)."""
    inter = np.minimum(wh[:, None, 0], centroids[None, :, 0]) * np.minimum(wh[:, None, 1], centroids[None, :, 1])
    union = wh[:, 0:1] * wh[:, 1:2] + (centroids[:, 0] * centroids[:, 1])[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def kmeans_anchors(wh: np.ndarray, k: int = 9, iters: int = 300, max_samples: int = 200_000,
                   seed: int = 42):
    """Cluster (w, h) pixel sizes into k anchors.

    Returns (centroids sorted by area (k,2), mean best IoU of the samples with their anchor).
    Uses the median for the update step (more robust to outliers than the mean).
    """
    rng = np.random.default_rng(seed)
    wh = np.asarray(wh, dtype=np.float64)
    wh = wh[(wh[:, 0] > 0) & (wh[:, 1] > 0)]
    if len(wh) > max_samples:
        wh = wh[rng.choice(len(wh), max_samples, replace=False)]

    centroids = wh[rng.choice(len(wh), k, replace=False)]
    assign = None
    for _ in range(iters):
        new_assign = np.argmax(wh_iou(wh, centroids), axis=1)
        if assign is not None and np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for j in range(k):
            members = wh[assign == j]
            if len(members):
                centroids[j] = np.median(members, axis=0)
    centroids = centroids[np.argsort(centroids.prod(axis=1))]
    mean_iou = float(wh_iou(wh, centroids).max(axis=1).mean())
    return centroids, mean_iou


def anchors_to_sizes_ratios(centroids: np.ndarray):
    """Translate (w, h) centroids into torchvision AnchorGenerator terms.

    size = sqrt(w*h) (one per FPN level), aspect ratio = h / w (torchvision convention).
    """
    sizes = np.sqrt(centroids.prod(axis=1))
    ratios = centroids[:, 1] / centroids[:, 0]
    return sizes, ratios
