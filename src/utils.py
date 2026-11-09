"""Shared helpers: config, seeding, environment, box formats, experiment logging, W&B.

Canonical detection format used everywhere in this repo
-------------------------------------------------------
For one image, detections are a float32 numpy array of shape (N, 6):

    [x1, y1, x2, y2, score, cls]

* pixel coordinates in the ORIGINAL image (no resize / letterbox / tile offsets)
* ``cls`` is the 0-based class id (0 = person ... 7 = motor)

Predictions for a whole split are a dict ``{image_key: dets}``, where
``image_key = f"{clip}_{frame_stem}"`` (same as ``image_id`` in test_images.csv).
Converters to COCO (1-based ids, xywh) and to the submission string live in
``src/eval.py`` and ``src/submission.py``.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
NUM_CLASSES = len(CLASS_NAMES)
REPO_ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- config
def load_config(path: str | Path = REPO_ROOT / "configs" / "base.yaml", **overrides) -> dict:
    """Load a YAML config. ``overrides`` use ``section__key=value`` for nested keys."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in overrides.items():
        node = cfg
        parts = key.split("__")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def deep_update(base: dict, new: dict) -> dict:
    """Recursively merge ``new`` into a copy of ``base`` (nested dicts are merged, the rest replaced)."""
    out = dict(base)
    for k, v in new.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_exp_config(path: str | Path, **overrides) -> dict:
    """Load an experiment config that may inherit from others via ``_base_: other.yaml``.

    The chain is resolved relative to the file's folder and merged (child wins), e.g.
    frcnn_tuned.yaml -> frcnn_small_anchors.yaml -> frcnn_baseline.yaml -> base.yaml.
    ``overrides`` work like in load_config (``section__key=value``).
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("_base_", None)
    if parent:
        cfg = deep_update(load_exp_config(path.parent / parent), cfg)
    for key, value in overrides.items():
        node = cfg
        parts = key.split("__")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def seed_everything(seed: int = 42) -> None:
    """Fix python / numpy / torch seeds (torch is optional so EDA works on CPU-only envs)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def detect_env() -> str:
    """Return 'kaggle', 'colab' or 'local'."""
    try:  # check Colab first: Colab also has /kaggle/... folders (Kaggle integration)
        import google.colab  # noqa: F401

        return "colab"
    except ImportError:
        pass
    return "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "local"


def print_versions() -> None:
    """Print versions of the libraries that matter for reproducibility (one line each)."""
    import importlib
    import platform

    print(f"python          {platform.python_version()}")
    for name in ["numpy", "torch", "torchvision", "ultralytics", "faster_coco_eval",
                 "pycocotools", "ensemble_boxes", "sahi", "cv2", "wandb"]:
        try:
            mod = importlib.import_module(name)
            print(f"{name:15s} {getattr(mod, '__version__', 'installed')}")
        except ImportError:
            print(f"{name:15s} NOT INSTALLED")
    try:
        import torch

        dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
        print(f"device          {dev}")
    except ImportError:
        pass


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def timer(name: str = "block"):
    """``with timer('x'):`` prints the elapsed wall time."""
    t0 = time.perf_counter()
    yield
    print(f"[{name}] {time.perf_counter() - t0:.1f}s")


# ----------------------------------------------------------------------------- keys
def make_image_key(clip: str, frame_stem: str) -> str:
    """Unique image key, identical to `image_id` in test_images.csv (e.g. clip_001_0000001)."""
    return f"{clip}_{frame_stem}"


def key_from_path(file_name: str | Path) -> str:
    """'.../images/<clip>/<frame>.jpg' -> '<clip>_<frame>'."""
    p = Path(file_name)
    return make_image_key(p.parent.name, p.stem)


# ----------------------------------------------------------------------------- boxes
def yolo_to_xyxy(cxcywh: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised (cx, cy, w, h) -> pixel (x1, y1, x2, y2)."""
    b = np.asarray(cxcywh, dtype=np.float32).reshape(-1, 4)
    x1 = (b[:, 0] - b[:, 2] / 2) * width
    y1 = (b[:, 1] - b[:, 3] / 2) * height
    x2 = (b[:, 0] + b[:, 2] / 2) * width
    y2 = (b[:, 1] + b[:, 3] / 2) * height
    return np.stack([x1, y1, x2, y2], axis=1)


def xyxy_to_yolo(xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel (x1, y1, x2, y2) -> normalised (cx, cy, w, h)."""
    b = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
    cx = (b[:, 0] + b[:, 2]) / 2 / width
    cy = (b[:, 1] + b[:, 3]) / 2 / height
    w = (b[:, 2] - b[:, 0]) / width
    h = (b[:, 3] - b[:, 1]) / height
    return np.stack([cx, cy, w, h], axis=1)


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """(x1, y1, x2, y2) -> (x, y, w, h) with (x, y) = top-left (COCO / submission)."""
    b = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4).copy()
    b[:, 2] -= b[:, 0]
    b[:, 3] -= b[:, 1]
    return b


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    b = np.asarray(xywh, dtype=np.float32).reshape(-1, 4).copy()
    b[:, 2] += b[:, 0]
    b[:, 3] += b[:, 1]
    return b


def clip_dets(dets: np.ndarray, width: int, height: int, min_size: float = 1e-3) -> np.ndarray:
    """Clip canonical detections to the image and drop degenerate boxes."""
    if len(dets) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    d = np.asarray(dets, dtype=np.float32).copy()
    d[:, [0, 2]] = d[:, [0, 2]].clip(0, width)
    d[:, [1, 3]] = d[:, [1, 3]].clip(0, height)
    keep = ((d[:, 2] - d[:, 0]) > min_size) & ((d[:, 3] - d[:, 1]) > min_size)
    return d[keep]


def topk_dets(dets: np.ndarray, k: int = 500, conf_thres: float = 0.0) -> np.ndarray:
    """Keep detections with score >= conf_thres, then the top-k by score."""
    if len(dets) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    d = dets[dets[:, 4] >= conf_thres]
    order = np.argsort(-d[:, 4], kind="stable")[:k]
    return d[order]


# ----------------------------------------------------------------------------- predictions I/O
def save_predictions(preds: dict, path: str | Path) -> None:
    """Persist {image_key: (N,6) array} so ensembles / tracking can reuse them without the GPU."""
    ensure_dir(Path(path).parent)
    with open(path, "wb") as f:
        pickle.dump({k: np.asarray(v, dtype=np.float32) for k, v in preds.items()}, f)


def load_predictions(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------- experiment log
EXPERIMENT_FIELDS = ["timestamp", "experiment", "model", "params", "val_mAP", "AP50", "AP75",
                     "AP_small", "AP_medium", "AP_large", "ms_per_img", "FPS",
                     "train_time_min", "public_lb", "notes"]


def log_experiment(experiment: str, model: str, params: dict, metrics: dict | None = None,
                   speed: dict | None = None, train_time_min: float | None = None,
                   notes: str = "", csv_path: str | Path = REPO_ROOT / "results" / "experiments.csv") -> dict:
    """Append one row to results/experiments.csv (public_lb is left blank for manual entry)."""
    metrics, speed = metrics or {}, speed or {}

    def fmt(v, nd=4):
        return "" if v is None else round(float(v), nd)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiment": experiment,
        "model": model,
        "params": json.dumps(params, default=str),
        "val_mAP": fmt(metrics.get("mAP")),
        "AP50": fmt(metrics.get("AP50")),
        "AP75": fmt(metrics.get("AP75")),
        "AP_small": fmt(metrics.get("AP_small")),
        "AP_medium": fmt(metrics.get("AP_medium")),
        "AP_large": fmt(metrics.get("AP_large")),
        "ms_per_img": fmt(speed.get("ms_per_img"), 2),
        "FPS": fmt(speed.get("fps"), 2),
        "train_time_min": fmt(train_time_min, 1),
        "public_lb": "",
        "notes": notes,
    }
    csv_path = Path(csv_path)
    ensure_dir(csv_path.parent)
    new_file = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    return row


# ----------------------------------------------------------------------------- W&B (optional)
def init_wandb(cfg: dict, run_name: str, job_type: str = "train", extra_config: dict | None = None):
    """Start a W&B run if ``cfg['wandb']['enabled']``; otherwise return None (all code must accept None)."""
    wb = cfg.get("wandb", {})
    if not wb.get("enabled", False):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed -> logging disabled")
        return None
    return wandb.init(project=wb.get("project"), entity=wb.get("entity"), name=run_name,
                      job_type=job_type, config={**cfg, **(extra_config or {})}, reinit=True)


def wandb_log(run, data: dict, step: int | None = None) -> None:
    """No-op when run is None."""
    if run is not None:
        run.log(data, step=step)
