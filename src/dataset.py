"""Dataset indexing and label reading for the VisDrone2019-VID competition layout.

    <split>/images/<clip>/<frame>.jpg
    <split>/labels/<clip>/<frame>.txt   YOLO: cls cx cy w h (normalised)
    <split>/ignore/<clip>/<frame>.txt   cx cy w h (normalised), only if the frame has ignore regions

Torch Dataset classes for torchvision detectors (Faster R-CNN, RetinaNet) are at the bottom.
torch is imported lazily so the CPU-only EDA environment still works.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:  # torch is optional here (EDA / submission code runs without it)
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover
    torch = None
    _TorchDataset = object

from .utils import CLASS_NAMES, make_image_key, yolo_to_xyxy

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ----------------------------------------------------------------------------- frame index
def build_frame_index(split_root: str | Path, frame_step: int = 1) -> pd.DataFrame:
    """One row per frame: key, clip, frame, img/label/ignore paths, width, height, frame_idx.

    ``frame_step`` keeps every n-th frame *within each clip* (neighbouring frames are near-duplicates).
    Image size is read once per clip (all frames of a VisDrone clip share the resolution).
    """
    split_root = Path(split_root)
    rows = []
    for clip_dir in sorted(p for p in (split_root / "images").iterdir() if p.is_dir()):
        frames = sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not frames:
            continue
        with Image.open(frames[0]) as im:  # lazy: only reads the header
            width, height = im.size
        for idx, img_path in enumerate(frames):
            if idx % frame_step:
                continue
            clip, stem = clip_dir.name, img_path.stem
            ign = split_root / "ignore" / clip / f"{stem}.txt"
            rows.append({
                "key": make_image_key(clip, stem), "clip": clip, "frame": stem, "frame_idx": idx,
                "img_path": str(img_path),
                "label_path": str(split_root / "labels" / clip / f"{stem}.txt"),
                "ignore_path": str(ign) if ign.exists() else "",
                "width": width, "height": height,
            })
    df = pd.DataFrame(rows)
    print(f"[index] {split_root.name}: {df['clip'].nunique()} clips, {len(df)} frames (step={frame_step})")
    return df


def resolve_test_path(data_root: str | Path, file_name: str) -> Path:
    """test_images.csv file_name may be relative to DATA_ROOT, test/ or test/images/ - try all."""
    data_root = Path(data_root)
    for cand in [data_root / file_name, data_root / "test" / file_name,
                 data_root / "test" / "images" / file_name]:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"cannot find test image {file_name} under {data_root}")


# ----------------------------------------------------------------------------- label files
def read_yolo_labels(label_path: str | Path, width: int, height: int):
    """Return (boxes_xyxy float32 (N,4) in pixels, classes int64 (N,)). Missing/empty file -> empty."""
    p = Path(label_path)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
    arr = np.loadtxt(p, dtype=np.float32, ndmin=2)
    return yolo_to_xyxy(arr[:, 1:5], width, height), arr[:, 0].astype(np.int64)


def read_ignore(ignore_path: str | Path, width: int, height: int) -> np.ndarray:
    """Return ignore regions as pixel xyxy (N,4). '' / missing file -> empty."""
    if not ignore_path:
        return np.zeros((0, 4), np.float32)
    p = Path(ignore_path)
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 4), np.float32)
    arr = np.loadtxt(p, dtype=np.float32, ndmin=2)
    return yolo_to_xyxy(arr[:, :4], width, height)


# ----------------------------------------------------------------------------- COCO -> tables (EDA)
def load_coco_tables(coco_json: str | Path):
    """Load annotations_coco.json into two DataFrames (images, annotations) for EDA.

    annotations gets: cls (0-based), class name, x, y, w, h, area, size=sqrt(w*h), clip, key,
    image width/height and the optional VisDrone fields (occlusion, truncation, track_id).
    """
    with open(coco_json) as f:
        coco = json.load(f)
    imgs = pd.DataFrame(coco["images"])
    imgs["clip"] = imgs["file_name"].map(lambda s: Path(s).parent.name)
    imgs["key"] = imgs["file_name"].map(lambda s: make_image_key(Path(s).parent.name, Path(s).stem))

    ann = pd.DataFrame(coco["annotations"])
    ann[["x", "y", "w", "h"]] = pd.DataFrame(ann["bbox"].tolist(), index=ann.index)
    ann = ann.drop(columns=["bbox", "segmentation"], errors="ignore")
    if "iscrowd" not in ann:
        ann["iscrowd"] = 0
    ann["area_px"] = ann["w"] * ann["h"]
    ann["size"] = np.sqrt(ann["area_px"].clip(lower=0))
    ann["cls"] = ann["category_id"] - 1  # COCO 1-based -> 0-based
    ann["class"] = ann["cls"].map(lambda c: CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else "ignore")
    ann = ann.merge(imgs[["id", "clip", "key", "width", "height"]],
                    left_on="image_id", right_on="id", how="left", suffixes=("", "_img"))
    ann = ann.drop(columns=["id_img"], errors="ignore")
    return imgs, ann


def size_bucket(area_px: pd.Series) -> pd.Series:
    """COCO size buckets: small < 32^2 <= medium < 96^2 <= large."""
    return pd.cut(area_px, [-1, 32 ** 2, 96 ** 2, np.inf], labels=["small", "medium", "large"])


# ----------------------------------------------------------------------------- torchvision datasets
# Ignore-region handling for torchvision models (choice for this repo)
# ---------------------------------------------------------------------
# ignore_mode = "mask_loss" (default): the ignore boxes are appended to target["boxes"] with
#   target["is_ignore"] = True. They are resized/flipped exactly like the GT boxes (torchvision's
#   GeneralizedRCNNTransform resizes target["boxes"]), and the ignore-aware RPN / RoI heads in
#   src/models_tv.py split them off and turn NEGATIVE anchors / proposals that lie mostly inside an
#   ignore region (IoA > ioa_thresh) into "don't care" (label -1), so they are never sampled as
#   background. Positives are untouched, and all GT boxes are kept (no pixels are hidden).
#   Same idea as the "ignore" flag of the VisDrone/KITTI-style protocols and the crowd handling in
#   Detectron (Girshick et al., https://github.com/facebookresearch/Detectron, "crowd" boxes).
# ignore_mode = "paint": documented fallback, identical to src/convert.py (gray fill, GT boxes
#   covered > drop_ioa removed, kept boxes pasted back). No special model code needed.
# ignore_mode = "none": ignore regions are not used at all (ablation).
IGNORE_MODES = ("mask_loss", "paint", "none")


def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image as an RGB uint8 HxWx3 array (raises if the file is unreadable)."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pre_resize(img: np.ndarray, max_short: int | None, max_long: int | None):
    """Shrink (never enlarge) so that short <= max_short and long <= max_long. Returns (img, scale).

    The model's own transform resizes again to its min_size/max_size; shrinking here first only
    saves CPU->GPU transfer and memory for the 2.7K / 4K frames. Small frames are left as they are
    (the model transform upsamples them), so the final network input is the same.
    """
    import cv2

    h, w = img.shape[:2]
    scale = 1.0
    if max_short:
        scale = min(scale, max_short / min(h, w))
    if max_long:
        scale = min(scale, max_long / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img, scale


def _valid_boxes(boxes: np.ndarray, width: int, height: int, min_side: float = 1.0) -> np.ndarray:
    """Clip to the image; return a boolean mask of boxes with both sides >= min_side pixels."""
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    return ((boxes[:, 2] - boxes[:, 0]) >= min_side) & ((boxes[:, 3] - boxes[:, 1]) >= min_side)


class TVDetectionDataset(_TorchDataset):
    """Frame index (build_frame_index) -> (image uint8 CxHxW tensor, target dict, meta dict).

    target: boxes float32 (N,4) xyxy in the (pre-resized) image, labels int64 1..8 (0 = background
    in torchvision; ignore boxes get 0 as a placeholder), is_ignore bool (N,).
    meta: key, scale (pre-resize factor, divide predictions by it), orig_size (H, W).

    Augmentation (train only): horizontal flip with prob ``hflip``. Scale jitter is done by the
    model transform (random choice of min_size per image, see models_tv.set_input_size).
    Images stay uint8 here; the train loop converts to float on the GPU (4x less transfer).
    """

    def __init__(self, index: pd.DataFrame, train: bool = True, ignore_mode: str = "mask_loss",
                 hflip: float = 0.5, max_short: int | None = None, max_long: int | None = None,
                 ignore_fill: int = 114, ignore_drop_ioa: float = 0.5):
        if torch is None:
            raise ImportError("TVDetectionDataset needs torch")
        assert ignore_mode in IGNORE_MODES, ignore_mode
        # plain namespaces, not itertuples(): those cannot be pickled for spawn-based DataLoader workers (macOS)
        self.rows = [SimpleNamespace(**r) for r in index.to_dict("records")]
        self.train, self.ignore_mode, self.hflip = train, ignore_mode, hflip
        self.max_short, self.max_long = max_short, max_long
        self.ignore_fill, self.ignore_drop_ioa = ignore_fill, ignore_drop_ioa

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        W, H = int(row.width), int(row.height)
        img = load_rgb(row.img_path)
        boxes, cls = read_yolo_labels(row.label_path, W, H)
        ign = read_ignore(row.ignore_path, W, H)

        if self.ignore_mode == "paint" and len(ign):
            from .convert import mask_frame  # local import: convert.py imports this module

            img, boxes, cls = mask_frame(img, ign, boxes, cls, fill=self.ignore_fill,
                                         drop_ioa=self.ignore_drop_ioa)
        if self.ignore_mode != "mask_loss":
            ign = np.zeros((0, 4), np.float32)

        img, scale = pre_resize(img, self.max_short, self.max_long)
        h, w = img.shape[:2]
        boxes, ign = boxes * scale, ign * scale

        if self.train and np.random.rand() < self.hflip:
            img = img[:, ::-1]
            boxes = boxes.copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            ign = ign.copy()
            ign[:, [0, 2]] = w - ign[:, [2, 0]]

        keep = _valid_boxes(boxes, w, h)
        boxes, cls = boxes[keep], cls[keep]
        ign = ign[_valid_boxes(ign, w, h)]

        all_boxes = np.concatenate([boxes, ign]).astype(np.float32).reshape(-1, 4)
        labels = np.concatenate([cls + 1, np.zeros(len(ign), np.int64)]).astype(np.int64)
        is_ignore = np.concatenate([np.zeros(len(boxes), bool), np.ones(len(ign), bool)])

        target = {"boxes": torch.from_numpy(all_boxes), "labels": torch.from_numpy(labels),
                  "is_ignore": torch.from_numpy(is_ignore)}
        meta = {"key": row.key, "scale": scale, "orig_size": (H, W)}
        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # uint8 CxHxW
        return img_t, target, meta


class TVImageDataset(_TorchDataset):
    """Inference only: list of (key, image path) -> (image uint8 tensor, meta)."""

    def __init__(self, keys: list[str], paths: list[str], max_short: int | None = None,
                 max_long: int | None = None):
        if torch is None:
            raise ImportError("TVImageDataset needs torch")
        assert len(keys) == len(paths)
        self.keys, self.paths = list(keys), [str(p) for p in paths]
        self.max_short, self.max_long = max_short, max_long

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i])
        H, W = img.shape[:2]
        img, scale = pre_resize(img, self.max_short, self.max_long)
        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        return img_t, {"key": self.keys[i], "scale": scale, "orig_size": (H, W)}


def collate_tv(batch):
    """torchvision detectors take lists of images/targets of different sizes."""
    return tuple(zip(*batch))
