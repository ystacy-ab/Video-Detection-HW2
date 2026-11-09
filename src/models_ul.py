"""Ultralytics models: YOLOv8 (one-stage anchor-free) and RT-DETR (query-based), same protocol as torchvision.

* YOLOv8: anchor-free head with Task-Aligned assignment (TOOD, Feng et al., ICCV 2021) and DFL box loss
  (Generalized Focal Loss, Li et al., NeurIPS 2020). https://github.com/ultralytics/ultralytics
  P2 variant = the official `yolov8-p2.yaml` (extra stride-4 head for small objects).
* RT-DETR: Zhao et al., "DETRs Beat YOLOs on Real-time Object Detection", CVPR 2024. Number of queries
  is a decoder argument; it does not change any weight shape, so COCO weights transfer completely.

Training data: WORK_DIR/yolo/... built by src/convert.py (ignore regions painted gray, same frames as the
torchvision models). Per-epoch curves come from the Ultralytics validator on every 4th val frame; the
reported score always comes from src.eval.evaluate on ALL val frames (original images, ignore = crowd).
Weights used for the score = last.pt (end of the schedule, no checkpoint selection on val).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from .convert import build_yolo_split, write_data_yaml
from .eval import clip_groups, evaluate, evaluate_groups, measure_speed, per_class_table
from .utils import (clip_dets, ensure_dir, load_predictions, log_experiment, save_predictions,
                    seed_everything, topk_dets)


def _ul():
    """Import Ultralytics lazily and switch off its own W&B / hub integrations (no login prompts)."""
    import ultralytics
    from ultralytics import settings

    settings.update({"wandb": False, "hub": False, "mlflow": False, "clearml": False, "comet": False,
                     "dvc": False, "neptune": False, "raytune": False, "tensorboard": False})
    return ultralytics


# ----------------------------------------------------------------------------- data
def prepare_ul_data(cfg: dict, data_root, work_dir, smoke: bool = False) -> Path:
    """Build (or reuse) the Ultralytics train / per-epoch-val splits for cfg's frame step -> data.yaml."""
    from .train_tv import prepare_indices  # same frame selection as the torchvision models

    d = cfg["data"]
    train_idx, val_epoch, _ = prepare_indices(cfg, data_root, smoke)
    root = Path(work_dir) / "yolo"
    sfx = "_smoke" if smoke else ""
    kw = dict(fill=d["ignore_fill"], drop_ioa=d["ignore_drop_ioa"], link_unmasked=d.get("link_unmasked", True))
    tr = build_yolo_split(train_idx, root, f"train_step{d['train_frame_step']}{sfx}", **kw)
    va = build_yolo_split(val_epoch, root, f"val_step{d['val_frame_step_epoch']}{sfx}", **kw)
    return write_data_yaml(root / f"data_step{d['train_frame_step']}{sfx}.yaml", [tr], va)


# ----------------------------------------------------------------------------- model
def _weights_path(work_dir, name: str) -> str:
    """Download official weights into WORK_DIR/weights (not into the notebook folder)."""
    return str(ensure_dir(Path(work_dir) / "weights") / name)


def rtdetr_yaml(work_dir, base: str = "rtdetr-l.yaml", queries: int = 300) -> Path:
    """Copy of the official RT-DETR yaml with another number of decoder queries."""
    ul = _ul()
    src = Path(ul.__file__).parent / "cfg" / "models" / "rt-detr" / base
    cfg = yaml.safe_load(src.read_text())
    head = cfg["head"][-1]  # [[from], repeats, "RTDETRDecoder", [nc, hd, nq, ...]]
    assert head[2] == "RTDETRDecoder", head
    head[3] = [head[3][0], 256, int(queries)]  # nc, hidden dim (default 256), number of queries
    out = ensure_dir(Path(work_dir) / "weights") / f"{Path(base).stem}-q{queries}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def build_ul_model(cfg: dict, work_dir):
    """New (pretrained) model from cfg['model']: arch yolo|rtdetr, weights, optional cfg yaml / queries."""
    ul = _ul()
    m = cfg["model"]
    cls = ul.RTDETR if m["arch"] == "rtdetr" else ul.YOLO
    weights = _weights_path(work_dir, m["weights"])
    if m["arch"] == "rtdetr" and m.get("queries", 300) != 300:
        model = cls(str(rtdetr_yaml(work_dir, m.get("cfg") or "rtdetr-l.yaml", m["queries"]))).load(weights)
    elif m.get("cfg"):  # e.g. yolov8s-p2.yaml: new architecture, matching layers copied from `weights`
        model = cls(m["cfg"]).load(weights)
    else:
        model = cls(weights)
    return model


def load_ul(path, arch: str):
    ul = _ul()
    return (ul.RTDETR if arch == "rtdetr" else ul.YOLO)(str(path))


def _epochs_done(run_dir: Path) -> int:
    f = run_dir / "results.csv"
    return len(pd.read_csv(f)) if f.exists() else 0


def train_ul(cfg: dict, data_yaml, work_dir, exp: str, smoke: bool = False) -> Path:
    """Train with Ultralytics (args = cfg['train'] verbatim). Resumes from last.pt; skips if finished."""
    project = Path(work_dir) / "ul"
    run_dir = project / exp
    args = dict(cfg["train"])
    if smoke:
        args.update(epochs=1, fraction=0.05)
    last = run_dir / "weights" / "last.pt"
    done = _epochs_done(run_dir)
    if done >= args["epochs"] and last.exists():
        print(f"[ul] {exp}: {done} epochs already trained -> skip training")
        return run_dir
    if last.exists() and done > 0:
        print(f"[ul] {exp}: resuming after epoch {done}")
        load_ul(last, cfg["model"]["arch"]).train(resume=True)
    else:
        model = build_ul_model(cfg, work_dir)
        model.train(data=str(data_yaml), project=str(project), name=exp, exist_ok=True,
                    seed=cfg["seed"], **args)
    return run_dir


def train_minutes(run_dir: Path) -> float:
    """Training time from results.csv (its 'time' column restarts at 0 after a resume)."""
    f = Path(run_dir) / "results.csv"
    if not f.exists():
        return float("nan")
    total, prev = 0.0, 0.0
    for v in pd.read_csv(f)["time"].tolist():
        total += v - prev if v >= prev else v
        prev = v
    return total / 60


# ----------------------------------------------------------------------------- inference
def _canonical(res, max_det: int) -> np.ndarray:
    b = res.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 6), np.float32)
    d = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1)
    h, w = res.orig_shape
    return topk_dets(clip_dets(d.astype(np.float32), w, h), k=max_det)


def predict_ul(model, keys: list[str], paths: list[str], pcfg: dict, imgsz: int, augment: bool = False,
               chunk: int = 256, desc: str = "predict") -> dict:
    """Predict image paths -> {key: (N,6) canonical} in original pixels (Ultralytics rescales boxes)."""
    import torch

    half = torch.cuda.is_available()
    preds = {}
    for s in tqdm(range(0, len(paths), chunk), desc=desc, leave=False):
        res = model.predict(source=list(map(str, paths[s:s + chunk])), imgsz=imgsz, conf=pcfg["conf"],
                            iou=pcfg["iou"], max_det=pcfg["max_det"], batch=pcfg.get("batch", 16), half=half,
                            augment=augment, verbose=False, stream=True)
        for k, r in zip(keys[s:s + chunk], res):
            preds[k] = _canonical(r, pcfg["max_det"])
    return preds


def tile_origins(size: int, tile: int, n: int) -> list[int]:
    return [0] if n == 1 else [int(round(x)) for x in np.linspace(0, size - tile, n)]


def predict_tiled(model, keys: list[str], paths: list[str], pcfg: dict, imgsz: int, grid: int = 2,
                  overlap: float = 0.25, merge_iou: float = 0.6, desc: str = "tiles") -> dict:
    """Full image + grid x grid overlapping tiles (each resized to imgsz, i.e. zoomed in), merged with
    class-wise NMS. Simplified slicing-aided inference (SAHI: Akyon et al., ICIP 2022)."""
    import torch
    from torchvision.ops import batched_nms

    half = torch.cuda.is_available()
    preds = {}
    for k, p in tqdm(list(zip(keys, paths)), desc=desc, leave=False):
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        tw, th = int(W * (1 + overlap) / grid), int(H * (1 + overlap) / grid)
        crops, offs = [img], [(0, 0)]
        for y0 in tile_origins(H, th, grid):
            for x0 in tile_origins(W, tw, grid):
                crops.append(img[y0:y0 + th, x0:x0 + tw])
                offs.append((x0, y0))
        res = model.predict(source=crops, imgsz=imgsz, conf=pcfg["conf"], iou=pcfg["iou"],
                            max_det=pcfg["max_det"], half=half, verbose=False)
        parts = []
        for r, (x0, y0) in zip(res, offs):
            d = _canonical(r, pcfg["max_det"])
            d[:, [0, 2]] += x0
            d[:, [1, 3]] += y0
            parts.append(d)
        d = np.concatenate(parts)
        if len(d):
            t = torch.from_numpy(d)
            keep = batched_nms(t[:, :4], t[:, 4], t[:, 5].long(), merge_iou).numpy()
            d = d[keep]
        preds[k] = topk_dets(clip_dets(d, W, H), k=pcfg["max_det"])
    return preds


def make_predict_fn_ul(model, pcfg: dict, imgsz: int):
    """Single image, end to end (decode + letterbox + model + NMS) for measure_speed."""
    import torch

    half = torch.cuda.is_available()

    def fn(path):
        return model.predict(str(path), imgsz=imgsz, conf=pcfg["conf"], iou=pcfg["iou"],
                             max_det=pcfg["max_det"], half=half, verbose=False)

    return fn


# ----------------------------------------------------------------------------- experiment pipeline
def summarize_params_ul(cfg: dict) -> dict:
    m, t = cfg["model"], cfg["train"]
    out = {k: v for k, v in m.items() if v is not None}
    out.update({k: t.get(k) for k in ["imgsz", "epochs", "batch", "optimizer", "cos_lr", "close_mosaic"]})
    out.update(frame_step=cfg["data"]["train_frame_step"], ignore="paint")
    return out


def test_lists(data_root):
    data_root = Path(data_root)
    from .dataset import resolve_test_path

    df = pd.read_csv(data_root / "test_images.csv")
    return df["image_id"].tolist(), [str(resolve_test_path(data_root, f)) for f in df["file_name"]]


def write_submission(preds: dict, cfg: dict, data_root, work_dir, name: str) -> Path:
    from .submission import make_submission, validate_submission

    test_csv = Path(data_root) / "test_images.csv"
    s = cfg["submission"]
    out = Path(work_dir) / "submissions" / f"{name}.csv"
    make_submission(preds, test_csv, out, max_det=s["max_det"], conf_thres=s["conf_thres"], decimals=s["decimals"])
    validate_submission(out, test_csv)
    return out


def run_ul_experiment(cfg: dict, data_yaml, data_root, work_dir, smoke: bool = False, test: bool = False,
                      notes: str = "") -> dict:
    """Train (resumable) -> predict ALL val frames -> COCO metrics, per clip, speed -> experiments.csv
    [-> test predictions + validated submission]. Re-running a finished experiment only reloads it."""
    from .train_tv import prepare_indices

    exp = cfg["experiment"] + ("_smoke" if smoke else "")
    seed_everything(cfg["seed"])
    work_dir = Path(work_dir)
    gt_json = Path(data_root) / "val" / "annotations_coco.json"
    run_dir = work_dir / "ul" / exp
    done_file = run_dir / "final.json"
    pcfg, imgsz, arch = cfg["predict"], cfg["train"]["imgsz"], cfg["model"]["arch"]

    if done_file.exists() and not smoke:
        info = json.loads(done_file.read_text())
        preds = load_predictions(work_dir / "preds" / f"{exp}_val.pkl")
        res = {"metrics": info["metrics"], "speed": info["speed"], "preds": preds, "exp": exp,
               "per_class": per_class_table(info["metrics"]), "run_dir": run_dir,
               "per_clip": evaluate_groups(preds, gt_json, clip_groups(gt_json), cfg["eval"]["max_dets"])}
        print(f"[run] {exp} already finished: mAP={info['metrics']['mAP']:.4f} (from {done_file})")
    else:
        train_ul(cfg, data_yaml, work_dir, exp, smoke)
        model = load_ul(run_dir / "weights" / "last.pt", arch)
        _, _, val_full = prepare_indices(cfg, data_root, smoke)
        preds = predict_ul(model, val_full["key"].tolist(), val_full["img_path"].tolist(), pcfg, imgsz,
                           desc="val full")
        save_predictions(preds, work_dir / "preds" / f"{exp}_val.pkl")
        metrics = evaluate(preds, gt_json, max_dets=cfg["eval"]["max_dets"])
        paths = val_full["img_path"].tolist()[:: max(1, len(val_full) // 50)]
        speed = measure_speed(make_predict_fn_ul(model, pcfg, imgsz), paths,
                              warmup=cfg["eval"]["speed_warmup"], iters=cfg["eval"]["speed_iters"])
        print(f"[final] {exp}: mAP={metrics['mAP']:.4f} | {speed['ms_per_img']:.1f} ms/img ({speed['fps']:.1f} FPS)")
        res = {"metrics": metrics, "speed": speed, "preds": preds, "exp": exp, "run_dir": run_dir,
               "per_class": per_class_table(metrics),
               "per_clip": evaluate_groups(preds, gt_json, clip_groups(gt_json), cfg["eval"]["max_dets"])}
        if not smoke:
            log_experiment(exp, f"ultralytics-{arch}", summarize_params_ul(cfg), metrics, speed,
                           train_time_min=train_minutes(run_dir), notes=notes)
            done_file.write_text(json.dumps({"metrics": metrics, "speed": speed}, default=float))

    sub = work_dir / "submissions" / f"{exp}.csv"
    if test and not smoke and not sub.exists():
        model = load_ul(run_dir / "weights" / "last.pt", arch)
        keys, paths = test_lists(data_root)
        tpreds = predict_ul(model, keys, paths, pcfg, imgsz, desc="test")
        save_predictions(tpreds, work_dir / "preds" / f"{exp}_test.pkl")
        res["submission"] = write_submission(tpreds, cfg, data_root, work_dir, exp)
    return res


def read_curves(work_dir, exps: list[str]) -> pd.DataFrame:
    """Per-epoch Ultralytics validation curves (mAP50-95 on every 4th val frame) for several runs."""
    rows = []
    for e in exps:
        f = Path(work_dir) / "ul" / e / "results.csv"
        if f.exists():
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            df["experiment"] = e
            df["train_min"] = np.cumsum(np.maximum(np.diff(np.r_[0.0, df["time"].values]), 0)) / 60
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
