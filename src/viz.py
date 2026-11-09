"""Drawing helpers for GT / predictions / ignore regions (OpenCV, RGB output for matplotlib)."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils import CLASS_NAMES

# one fixed BGR color per class (tab10-like) so figures are consistent across notebooks
CLASS_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
                (148, 103, 189), (140, 86, 75), (227, 119, 194), (23, 190, 207)]
CLASS_COLORS = [(b, g, r) for (r, g, b) in CLASS_COLORS]
IGNORE_COLOR = (0, 255, 255)  # yellow (BGR)


def draw_boxes(img_bgr: np.ndarray, boxes_xyxy: np.ndarray, classes: np.ndarray,
               scores: np.ndarray | None = None, ignore_xyxy: np.ndarray | None = None,
               thickness: int = 1, show_labels: bool = False) -> np.ndarray:
    """Return an RGB copy with class-colored boxes and dashed-looking yellow ignore regions."""
    out = img_bgr.copy()
    if ignore_xyxy is not None and len(ignore_xyxy):
        overlay = out.copy()
        for x1, y1, x2, y2 in np.round(ignore_xyxy).astype(int):
            cv2.rectangle(overlay, (x1, y1), (x2, y2), IGNORE_COLOR, -1)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
        for x1, y1, x2, y2 in np.round(ignore_xyxy).astype(int):
            cv2.rectangle(out, (x1, y1), (x2, y2), IGNORE_COLOR, 2)
    for i, ((x1, y1, x2, y2), c) in enumerate(zip(np.round(boxes_xyxy).astype(int), classes)):
        color = CLASS_COLORS[int(c) % len(CLASS_COLORS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if show_labels:
            txt = CLASS_NAMES[int(c)] + (f" {scores[i]:.2f}" if scores is not None else "")
            cv2.putText(out, txt, (x1, max(y1 - 2, 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def show_grid(images: list[np.ndarray], titles: list[str] | None = None, ncols: int = 2,
              figsize_per: tuple = (8, 4.5), save_path: str | Path | None = None):
    """Plot RGB images in a grid; optionally save the figure."""
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for i, (ax, im) in enumerate(zip(np.atleast_1d(axes).ravel(), images)):
        ax.imshow(im)
        if titles:
            ax.set_title(titles[i], fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def legend_handles():
    """Matplotlib legend handles for the class colors + ignore regions."""
    from matplotlib.patches import Patch

    handles = [Patch(color=np.array(c[::-1]) / 255, label=n) for n, c in zip(CLASS_NAMES, CLASS_COLORS)]
    handles.append(Patch(color=np.array(IGNORE_COLOR[::-1]) / 255, label="ignore"))
    return handles
