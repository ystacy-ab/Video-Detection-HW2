"""Training / inference loop for torchvision detectors (Faster R-CNN now, RetinaNet later).

Any model that follows the torchvision detection API works:
    train mode: model(images, targets) -> dict of losses
    eval mode : model(images) -> [{"boxes", "labels" (1-based), "scores"}]

Features: AMP (fp16, T4 has no bf16), gradient accumulation, SGD/AdamW, warmup + cosine / OneCycle /
step schedule (per optimizer step), gradient clipping, per-epoch checkpoint (resumable), per-epoch
val mAP on a frame subset, optional W&B, and a prediction function returning the canonical format.
"""
from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .dataset import (TVDetectionDataset, TVImageDataset, build_frame_index, collate_tv, load_rgb,
                      pre_resize, resolve_test_path)
from .eval import clip_groups, evaluate, evaluate_groups, measure_speed, per_class_table
from .models_tv import build_faster_rcnn, build_retinanet, set_bn_eval, set_input_size, strip_ignore
from .submission import make_submission, validate_submission
from .utils import (clip_dets, ensure_dir, init_wandb, load_predictions, log_experiment, save_predictions,
                    seed_everything, topk_dets, wandb_log)


# ----------------------------------------------------------------------------- setup helpers
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: dict):
    """Dispatch on cfg['model']['arch']."""
    arch = cfg["model"]["arch"]
    if arch == "fasterrcnn_resnet50_fpn_v2":
        return build_faster_rcnn(cfg["model"], ignore_ioa=cfg["ignore"]["ioa_thresh"])
    if arch == "retinanet_resnet50_fpn_v2":
        return build_retinanet(cfg["model"], ignore_ioa=cfg["ignore"]["ioa_thresh"])
    raise ValueError(f"unknown arch {arch}")


def _autocast(device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_optimizer(model, tcfg: dict):
    """No weight decay on biases and norm weights (1-D params), common detection practice."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [{"params": decay, "weight_decay": tcfg["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    if tcfg["optimizer"] == "sgd":
        return torch.optim.SGD(groups, lr=tcfg["lr"], momentum=tcfg.get("momentum", 0.9), nesterov=False)
    if tcfg["optimizer"] == "adamw":
        return torch.optim.AdamW(groups, lr=tcfg["lr"])
    raise ValueError(tcfg["optimizer"])


def build_scheduler(opt, tcfg: dict, total_steps: int):
    """Per-optimizer-step LR schedule: linear warmup then cosine (default) / step, or OneCycle."""
    kind = tcfg.get("schedule", "cosine")
    if kind == "onecycle":  # Smith & Topin, "Super-Convergence", 2017
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[tcfg["lr"]] * len(opt.param_groups),
                                                   total_steps=total_steps, pct_start=0.1)
    warmup, min_ratio = tcfg.get("warmup_iters", 500), tcfg.get("min_lr_ratio", 0.05)

    def factor(step):
        if step < warmup:
            return 0.001 + (1 - 0.001) * step / warmup
        p = (step - warmup) / max(1, total_steps - warmup)
        if kind == "cosine":  # Loshchilov & Hutter, "SGDR", ICLR 2017
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        if kind == "step":  # x0.1 at 2/3 and 8/9 of training (Detectron "1x"-like)
            return 1.0 if p < 2 / 3 else (0.1 if p < 8 / 9 else 0.01)
        raise ValueError(kind)

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def make_train_loader(index: pd.DataFrame, cfg: dict, seed: int):
    m, t = cfg["model"], cfg["train"]
    ds = TVDetectionDataset(index, train=True, ignore_mode=cfg["ignore"]["mode"],
                            hflip=cfg["data"].get("hflip", 0.5),
                            max_short=max(m["min_size_train"]), max_long=m["max_size"],
                            ignore_fill=cfg["data"]["ignore_fill"],
                            ignore_drop_ioa=cfg["data"]["ignore_drop_ioa"])
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=t["batch_size"], shuffle=True, num_workers=t["num_workers"],
                      collate_fn=collate_tv, pin_memory=True, drop_last=True, generator=g,
                      persistent_workers=t["num_workers"] > 0)


# ----------------------------------------------------------------------------- inference
def _to_canonical(out: dict, meta: dict, max_det: int) -> np.ndarray:
    """torchvision output (pre-resized image coords, 1-based labels) -> (N,6) canonical array."""
    boxes = out["boxes"].float().cpu().numpy() / meta["scale"]
    scores = out["scores"].float().cpu().numpy()[:, None]
    cls = out["labels"].cpu().numpy()[:, None].astype(np.float32) - 1
    dets = np.concatenate([boxes, scores, cls], axis=1).astype(np.float32)
    dets = dets[dets[:, 5] >= 0]  # RetinaNet has an (untrained) column for label 0 -> drop it
    H, W = meta["orig_size"]
    return topk_dets(clip_dets(dets, W, H), k=max_det)


@torch.no_grad()
def predict(model, keys: list[str], paths: list[str], cfg: dict, device, batch_size: int = 4,
            num_workers: int = 4, desc: str = "predict") -> dict:
    """Predict a list of images -> {key: (N,6) [x1,y1,x2,y2,score,cls0]} in original pixels."""
    m = cfg["model"]
    model.eval()
    set_input_size(model, [m["min_size_test"]])
    ds = TVImageDataset(keys, paths, max_short=m["min_size_test"], max_long=m["max_size"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        collate_fn=collate_tv, pin_memory=True)
    preds = {}
    for imgs, metas in tqdm(loader, desc=desc, leave=False):
        x = [im.to(device, non_blocking=True).float().div_(255) for im in imgs]
        with _autocast(device, cfg["train"].get("amp", True)):
            outs = model(x)
        for o, meta in zip(outs, metas):
            preds[meta["key"]] = _to_canonical(o, meta, m["box_detections_per_img"])
    return preds


def predict_index(model, index: pd.DataFrame, cfg: dict, device, **kw) -> dict:
    return predict(model, index["key"].tolist(), index["img_path"].tolist(), cfg, device, **kw)


def make_predict_fn(model, cfg: dict, device):
    """Single-image end-to-end function (decode + resize + model + postprocess) for measure_speed."""
    m = cfg["model"]
    model.eval()
    set_input_size(model, [m["min_size_test"]])

    @torch.no_grad()
    def fn(path):
        img0 = load_rgb(path)
        img, s = pre_resize(img0, m["min_size_test"], m["max_size"])
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).to(device).float().div_(255)
        with _autocast(device, cfg["train"].get("amp", True)):
            out = model([x])[0]
        return _to_canonical(out, {"scale": s, "orig_size": img0.shape[:2]}, m["box_detections_per_img"])

    return fn


# ----------------------------------------------------------------------------- training
def train_one_epoch(model, loader, opt, sched, scaler, cfg, device, epoch: int, state: dict, run=None):
    """One pass over the loader with AMP + grad accumulation. Returns mean of each loss term."""
    t = cfg["train"]
    accum, amp, clip = t.get("accum_steps", 1), t.get("amp", True), t.get("grad_clip", 0)
    model.train()
    set_input_size(model, cfg["model"]["min_size_train"])
    if t.get("freeze_bn", True):
        set_bn_eval(model)
    needs_strip = not getattr(model, "supports_ignore", False)

    sums, n_ok, n_bad = {}, 0, 0
    opt.zero_grad(set_to_none=True)
    max_iters = t.get("max_iters_per_epoch") or len(loader)
    pbar = tqdm(enumerate(loader), total=min(len(loader), max_iters), desc=f"epoch {epoch + 1}")
    for it, (imgs, targets, _) in pbar:
        if it >= max_iters:
            break
        x = [im.to(device, non_blocking=True).float().div_(255) for im in imgs]
        tg = [{k: v.to(device, non_blocking=True) for k, v in d.items()} for d in targets]
        if needs_strip:
            tg = strip_ignore(tg)
        with _autocast(device, amp):
            loss_dict = model(x, tg)
        loss = sum(loss_dict.values())
        if not torch.isfinite(loss):  # skip the batch instead of corrupting the weights
            n_bad += 1
            opt.zero_grad(set_to_none=True)
            if n_bad > 20:
                raise RuntimeError("more than 20 non-finite losses in one epoch - lower the LR")
            continue
        scaler.scale(loss / accum).backward()
        if (it + 1) % accum == 0:
            if clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            state["global_step"] += 1
        n_ok += 1
        for k, v in loss_dict.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        sums["loss"] = sums.get("loss", 0.0) + float(loss)
        if it % t.get("log_every", 50) == 0:
            lr = opt.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{sums['loss'] / n_ok:.3f}", lr=f"{lr:.2e}")
            wandb_log(run, {"train/loss": float(loss), "train/lr": lr,
                            **{f"train/{k}": float(v) for k, v in loss_dict.items()}},
                      step=state["global_step"])
    if n_bad:
        print(f"[train] skipped {n_bad} batches with non-finite loss")
    return {k: v / max(n_ok, 1) for k, v in sums.items()}


def _save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)  # atomic: a killed session never leaves a half-written checkpoint


def fit(model, cfg: dict, train_index: pd.DataFrame, val_index: pd.DataFrame, gt_json,
        out_dir, device, run=None, resume: bool = True) -> dict:
    """Train for cfg.train.epochs with a checkpoint after every epoch. Re-running resumes.

    Writes out_dir/last.pt (full state), out_dir/best.pt (weights with best val-subset mAP),
    out_dir/history.csv. Returns the state dict (history, best_map, train_time_min, ...).
    """
    t = cfg["train"]
    out_dir = ensure_dir(out_dir)
    model.to(device)
    loader = make_train_loader(train_index, cfg, seed=cfg["seed"])
    steps_per_epoch = min(len(loader), t.get("max_iters_per_epoch") or len(loader)) // t.get("accum_steps", 1)
    total_steps = max(1, steps_per_epoch * t["epochs"])
    opt = build_optimizer(model, t)
    sched = build_scheduler(opt, t, total_steps)
    scaler = _grad_scaler(t.get("amp", True) and device.type == "cuda")
    state = {"epoch": 0, "global_step": 0, "best_map": -1.0, "history": [], "train_time_min": 0.0}

    last = out_dir / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        print(f"[fit] resumed from {last} after epoch {state['epoch']}")
    if state["epoch"] >= t["epochs"]:
        print(f"[fit] already trained for {t['epochs']} epochs -> nothing to do")
        return state

    print(f"[fit] {len(train_index)} train frames, {len(loader)} it/epoch (batch {t['batch_size']} x "
          f"accum {t.get('accum_steps', 1)}), {total_steps} optimizer steps, val subset {len(val_index)} frames")
    for epoch in range(state["epoch"], t["epochs"]):
        t0 = time.time()
        losses = train_one_epoch(model, loader, opt, sched, scaler, cfg, device, epoch, state, run)
        train_min = (time.time() - t0) / 60
        row = {"epoch": epoch + 1, "lr": opt.param_groups[0]["lr"], "train_min": round(train_min, 2), **losses}

        if (epoch + 1) % t.get("val_every", 1) == 0 or epoch + 1 == t["epochs"]:
            preds = predict_index(model, val_index, cfg, device, num_workers=t["num_workers"],
                                  desc="val subset")
            met = evaluate(preds, gt_json, max_dets=cfg["eval"]["max_dets"], img_keys=list(preds),
                           verbose=False)
            row.update({"val_mAP": met["mAP"], "val_AP50": met["AP50"], "val_AP_small": met["AP_small"]})
            if met["mAP"] > state["best_map"]:
                state["best_map"] = met["mAP"]
                _save(model.state_dict(), out_dir / "best.pt")

        state["epoch"] = epoch + 1
        state["train_time_min"] += train_min
        state["history"].append(row)
        _save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
               "scaler": scaler.state_dict(), "state": state, "cfg": cfg}, last)
        pd.DataFrame(state["history"]).to_csv(out_dir / "history.csv", index=False)
        wandb_log(run, {f"epoch/{k}": v for k, v in row.items()}, step=state["global_step"])
        print("[fit] " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    return state


# ----------------------------------------------------------------------------- experiment pipeline
def prepare_indices(cfg: dict, data_root, smoke: bool = False):
    """Train frames (step from config), val subset for per-epoch mAP, full val for the final score."""
    data_root = Path(data_root)
    d = cfg["data"]
    train_idx = build_frame_index(data_root / "train", d["train_frame_step"])
    val_full = build_frame_index(data_root / "val", d.get("val_frame_step", 1))
    val_epoch = val_full[val_full["frame_idx"] % d.get("val_frame_step_epoch", 4) == 0].reset_index(drop=True)
    if smoke:  # a few frames from every clip: checks the whole pipeline in a couple of minutes
        train_idx = train_idx.groupby("clip").head(2).reset_index(drop=True)
        val_epoch = val_epoch.groupby("clip").head(4).reset_index(drop=True)
        val_full = val_epoch
    print(f"[data] train {len(train_idx)} | val per-epoch {len(val_epoch)} | val final {len(val_full)}")
    return train_idx, val_epoch, val_full


def summarize_params(cfg: dict) -> dict:
    """The settings that differ between experiments (stored in experiments.csv)."""
    m, t = cfg["model"], cfg["train"]
    skip = {"arch", "weights", "box_score_thresh", "box_detections_per_img", "trainable_backbone_layers"}
    if m["arch"].startswith("retinanet"):  # keys inherited from the Faster R-CNN config but unused here
        skip |= {k for k in m if k.startswith(("rpn_", "box_batch", "box_positive", "box_loss", "giou"))}
    out = {k: v for k, v in m.items() if k not in skip}
    out.update({"ignore": cfg["ignore"]["mode"], "frame_step": cfg["data"]["train_frame_step"],
                "epochs": t["epochs"], "opt": t["optimizer"], "lr": t["lr"], "sched": t.get("schedule"),
                "batch": t["batch_size"] * t.get("accum_steps", 1)})
    return out


def final_eval(model, cfg: dict, val_index: pd.DataFrame, gt_json, work_dir, exp: str, device) -> dict:
    """Full-val predictions (saved as pickle for WBF) + COCO metrics + speed."""
    preds = predict_index(model, val_index, cfg, device, num_workers=cfg["train"]["num_workers"],
                          desc="val full")
    pred_path = Path(work_dir) / "preds" / f"{exp}_val.pkl"
    save_predictions(preds, pred_path)
    metrics = evaluate(preds, gt_json, max_dets=cfg["eval"]["max_dets"])
    paths = val_index["img_path"].tolist()[:: max(1, len(val_index) // 50)]
    speed = measure_speed(make_predict_fn(model, cfg, device), paths,
                          warmup=cfg["eval"]["speed_warmup"], iters=cfg["eval"]["speed_iters"])
    print(f"[final] {exp}: mAP={metrics['mAP']:.4f} | {speed['ms_per_img']:.1f} ms/img "
          f"({speed['fps']:.1f} FPS, batch 1, incl. JPEG decode) | preds -> {pred_path}")
    return {"metrics": metrics, "speed": speed, "preds": preds, "per_class": per_class_table(metrics),
            "per_clip": evaluate_groups(preds, gt_json, clip_groups(gt_json), cfg["eval"]["max_dets"])}


def predict_test(model, cfg: dict, data_root, work_dir, exp: str, device) -> Path:
    """Predict every row of test_images.csv, save the pickle and a validated submission CSV."""
    data_root = Path(data_root)
    test_csv = data_root / "test_images.csv"
    df = pd.read_csv(test_csv)
    paths = [str(resolve_test_path(data_root, f)) for f in df["file_name"]]
    preds = predict(model, df["image_id"].tolist(), paths, cfg, device,
                    num_workers=cfg["train"]["num_workers"], desc="test")
    save_predictions(preds, Path(work_dir) / "preds" / f"{exp}_test.pkl")
    s = cfg["submission"]
    out = Path(work_dir) / "submissions" / f"{exp}.csv"
    make_submission(preds, test_csv, out, max_det=s["max_det"], conf_thres=s["conf_thres"], decimals=s["decimals"])
    validate_submission(out, test_csv)
    return out


def run_experiment(cfg: dict, data_root, work_dir, smoke: bool = False, test: bool = False,
                   notes: str = "") -> dict:
    """Whole pipeline for one config: train (resumable) -> full val eval -> speed -> experiments.csv."""
    exp = cfg["experiment"] + ("_smoke" if smoke else "")
    seed_everything(cfg["seed"])
    device = get_device()
    gt_json = Path(data_root) / "val" / "annotations_coco.json"
    work_dir = Path(work_dir)
    done_file = work_dir / "tv" / exp / "final.json"
    if done_file.exists() and not smoke:  # finished earlier (e.g. before a disconnect): reuse, do not re-log
        info = json.loads(done_file.read_text())
        preds = load_predictions(work_dir / "preds" / f"{exp}_val.pkl")
        res = {"metrics": info["metrics"], "speed": info["speed"], "preds": preds, "model": None, "exp": exp,
               "per_class": per_class_table(info["metrics"]), "state": info["state"],
               "per_clip": evaluate_groups(preds, gt_json, clip_groups(gt_json), cfg["eval"]["max_dets"])}
        print(f"[run] {exp} already finished: mAP={info['metrics']['mAP']:.4f} (from {done_file})")
        if test and not (work_dir / "submissions" / f"{exp}.csv").exists():
            res["submission"] = predict_test(load_trained(cfg, work_dir, device), cfg, data_root, work_dir, exp, device)
        return res
    if smoke:
        cfg = json.loads(json.dumps(cfg))  # deep copy
        cfg["train"].update(epochs=1, max_iters_per_epoch=40, warmup_iters=10)
    train_idx, val_epoch, val_full = prepare_indices(cfg, data_root, smoke)

    model = build_model(cfg)
    run = init_wandb(cfg, exp, extra_config={"params": summarize_params(cfg)})
    state = fit(model, cfg, train_idx, val_epoch, gt_json, Path(work_dir) / "tv" / exp, device, run)
    res = final_eval(model, cfg, val_full, gt_json, work_dir, exp, device)
    res.update(model=model, state=state, exp=exp)
    if not smoke:
        log_experiment(exp, cfg["model"]["arch"], summarize_params(cfg), res["metrics"], res["speed"],
                       train_time_min=state["train_time_min"], notes=notes)
        done_file.write_text(json.dumps({"metrics": res["metrics"], "speed": res["speed"], "state": state},
                                        default=float))
        if test:
            res["submission"] = predict_test(model, cfg, data_root, work_dir, exp, device)
    if run is not None:
        wandb_log(run, {f"final/{k}": v for k, v in res["metrics"].items()})
        run.finish()
    return res


def load_trained(cfg: dict, work_dir, device, which: str = "last"):
    """Rebuild a model from its config and load WORK_DIR/tv/<experiment>/{last,best}.pt."""
    path = Path(work_dir) / "tv" / cfg["experiment"] / f"{which}.pt"
    model = build_model(cfg)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    print(f"[load] {path}")
    return model.to(device).eval()
