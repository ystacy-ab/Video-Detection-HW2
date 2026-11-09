"""Convert the competition layout into framework-specific training sets.

Ultralytics (YOLO / RT-DETR)
----------------------------
Ultralytics has no "ignore region" concept in its loss, so (following the common VisDrone practice,
e.g. the VisDrone toolkit / Ultralytics VisDrone.yaml which drops 'ignored' boxes) we:
  1. paint every ignore region with gray (114 = Ultralytics letterbox fill) on a COPY of the frame,
  2. drop GT boxes covered > `drop_ioa` by ignore regions (they are no longer visible),
  3. paste the pixels of the remaining GT boxes back, so a partially overlapping object is intact.
The network therefore never sees unlabeled objects inside ignore regions as "background".
Frames without ignore regions are symlinked (no disk cost).

torchvision (Faster R-CNN / RetinaNet)
--------------------------------------
No conversion needed: the torchvision Dataset (next stage) reads the YOLO txt + ignore txt
directly and masks ignore regions in the loss.

Output layout (flat file names, so keys stay unique):
    <out_root>/<name>/images/<clip>_<frame>.jpg
    <out_root>/<name>/labels/<clip>_<frame>.txt
"""
from __future__ import annotations

import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from .dataset import read_ignore, read_yolo_labels
from .utils import CLASS_NAMES, ensure_dir, xyxy_to_yolo


def ignore_mask(ignore_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Boolean HxW mask, True inside any ignore region."""
    mask = np.zeros((height, width), dtype=bool)
    for x1, y1, x2, y2 in np.round(ignore_xyxy).astype(int):
        mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = True
    return mask


def box_ioa(boxes_xyxy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fraction of each box's area that lies inside the mask (intersection over box area)."""
    out = np.zeros(len(boxes_xyxy), dtype=np.float32)
    h, w = mask.shape
    for i, (x1, y1, x2, y2) in enumerate(np.round(boxes_xyxy).astype(int)):
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        if x2 > x1 and y2 > y1:
            out[i] = mask[y1:y2, x1:x2].mean()
    return out


def mask_frame(img: np.ndarray, ignore_xyxy: np.ndarray, boxes_xyxy: np.ndarray, classes: np.ndarray,
               fill: int = 114, drop_ioa: float = 0.5):
    """Paint ignore regions gray, drop covered GT boxes, restore pixels of the kept boxes.

    Returns (masked image, kept boxes, kept classes).
    """
    h, w = img.shape[:2]
    mask = ignore_mask(ignore_xyxy, w, h)
    keep = box_ioa(boxes_xyxy, mask) <= drop_ioa
    boxes_k, cls_k = boxes_xyxy[keep], classes[keep]
    out = img.copy()
    out[mask] = fill
    for x1, y1, x2, y2 in np.round(boxes_k).astype(int):
        x1, x2 = np.clip([x1, x2], 0, w)
        y1, y2 = np.clip([y1, y2], 0, h)
        out[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    return out, boxes_k, cls_k


def write_yolo_label(path: Path, boxes_xyxy: np.ndarray, classes: np.ndarray, width: int, height: int):
    """Write YOLO txt (cls cx cy w h normalised, clipped to [0, 1])."""
    if len(boxes_xyxy):
        b = boxes_xyxy.copy()
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, width)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, height)
        valid = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
        yolo = xyxy_to_yolo(b[valid], width, height).clip(0, 1)
        lines = [f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                 for c, (cx, cy, bw, bh) in zip(classes[valid], yolo)]
    else:
        lines = []
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _convert_one(row, img_dir: Path, lbl_dir: Path, fill: int, drop_ioa: float, link_unmasked: bool):
    """Convert a single frame; returns (#boxes kept, #boxes dropped, masked?)."""
    W, H = int(row.width), int(row.height)
    boxes, classes = read_yolo_labels(row.label_path, W, H)
    ignore = read_ignore(row.ignore_path, W, H)
    dst_img = img_dir / f"{row.key}.jpg"
    n_before = len(boxes)

    if len(ignore):
        img = cv2.imread(row.img_path)
        if img is None:
            raise FileNotFoundError(row.img_path)
        img, boxes, classes = mask_frame(img, ignore, boxes, classes, fill=fill, drop_ioa=drop_ioa)
        cv2.imwrite(str(dst_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif not dst_img.exists():
        if link_unmasked:
            os.symlink(os.path.abspath(row.img_path), dst_img)
        else:
            shutil.copy2(row.img_path, dst_img)

    write_yolo_label(lbl_dir / f"{row.key}.txt", boxes, classes, W, H)
    return len(boxes), n_before - len(boxes), bool(len(ignore))


def build_yolo_split(index: pd.DataFrame, out_root: str | Path, name: str, fill: int = 114,
                     drop_ioa: float = 0.5, link_unmasked: bool = True, workers: int = 8,
                     overwrite: bool = False) -> Path:
    """Create <out_root>/<name>/{images,labels} for Ultralytics from a frame index (see dataset.py).

    Idempotent: if the split already has as many labels as the index has frames, it is reused
    (useful after a Colab/Kaggle disconnect). Use overwrite=True to rebuild.
    """
    split_dir = Path(out_root) / name
    img_dir, lbl_dir = split_dir / "images", split_dir / "labels"
    if split_dir.exists() and not overwrite and len(list(lbl_dir.glob("*.txt"))) == len(index):
        print(f"[convert] {split_dir} already complete ({len(index)} frames) -> reuse")
        return split_dir
    if split_dir.exists():
        shutil.rmtree(split_dir)
    ensure_dir(img_dir), ensure_dir(lbl_dir)

    rows = list(index.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=workers) as ex:  # cv2 releases the GIL -> threads are enough
        stats = list(tqdm(ex.map(lambda r: _convert_one(r, img_dir, lbl_dir, fill, drop_ioa, link_unmasked),
                                 rows), total=len(rows), desc=f"convert {name}"))
    kept, dropped, masked = (np.array(x) for x in zip(*stats))
    print(f"[convert] {name}: frames={len(rows)} masked_frames={masked.sum()} "
          f"boxes_kept={kept.sum()} boxes_dropped_in_ignore={dropped.sum()}")
    return split_dir


def write_data_yaml(out_path: str | Path, train_dirs: list[str | Path], val_dir: str | Path) -> Path:
    """Ultralytics data.yaml with absolute paths (train may be a list, e.g. train + val for the final fit)."""
    data = {
        "path": str(Path(val_dir).parent),
        "train": [str(Path(d) / "images") for d in train_dirs],
        "val": str(Path(val_dir) / "images"),
        "nc": len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    print(f"[convert] wrote {out_path}")
    return out_path
