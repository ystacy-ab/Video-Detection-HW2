"""torchvision detectors: Faster R-CNN (this stage), shared helpers reused by RetinaNet later.

Faster R-CNN: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
Networks", NeurIPS 2015. FPN: Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017.
"v2" weights/architecture: Li et al., "Benchmarking Detection Transfer Learning with Vision
Transformers" (2021) as implemented in torchvision (`fasterrcnn_resnet50_fpn_v2`).
GIoU loss: Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019.

Model config keys (section `model:` in configs/frcnn_*.yaml) are documented in
configs/frcnn_baseline.yaml.
"""
from __future__ import annotations

import math
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import roi_heads as _roi_mod
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import box_iou, generalized_box_iou_loss

from .utils import NUM_CLASSES


# ----------------------------------------------------------------------------- ignore regions
def split_ignore(targets):
    """Separate ignore boxes (target['is_ignore']) from real GT. Returns (real_targets, ignore_boxes)."""
    if targets is None:
        return None, None
    real, ignore = [], []
    for t in targets:
        m = t.get("is_ignore")
        if m is None:
            real.append(t)
            ignore.append(t["boxes"].new_zeros((0, 4)))
            continue
        rest = {k: v for k, v in t.items() if k not in ("boxes", "labels", "is_ignore")}
        real.append({**rest, "boxes": t["boxes"][~m], "labels": t["labels"][~m]})
        ignore.append(t["boxes"][m])
    return real, ignore


def strip_ignore(targets):
    """For models without ignore support: drop the ignore boxes completely."""
    return split_ignore(targets)[0]


def max_ioa(boxes: torch.Tensor, ignore: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    """For each box: max over ignore regions of intersection / box area. Chunked (500k+ anchors)."""
    if ignore.numel() == 0 or boxes.numel() == 0:
        return boxes.new_zeros(len(boxes))
    boxes, ignore = boxes.float(), ignore.float()
    area = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)).clamp(min=1e-6)
    out = []
    for s in range(0, len(boxes), chunk):
        b = boxes[s:s + chunk]
        lt = torch.max(b[:, None, :2], ignore[None, :, :2])
        rb = torch.min(b[:, None, 2:], ignore[None, :, 2:])
        out.append((rb - lt).clamp(min=0).prod(-1).max(dim=1).values)
    return torch.cat(out) / area


class IgnoreAwareRPN(RegionProposalNetwork):
    """RPN whose NEGATIVE anchors inside ignore regions become 'don't care' (label -1)."""

    ignore_ioa_thresh = 0.5

    def forward(self, images, features, targets=None):
        targets, self._ignore = split_ignore(targets)
        return super().forward(images, features, targets)

    def assign_targets_to_anchors(self, anchors, targets):
        labels, matched_gt = super().assign_targets_to_anchors(anchors, targets)
        for lab, anc, ign in zip(labels, anchors, self._ignore or [None] * len(labels)):
            if ign is not None and len(ign):
                lab[(lab == 0) & (max_ioa(anc, ign) > self.ignore_ioa_thresh)] = -1.0
        return labels, matched_gt


class IgnoreAwareRoIHeads(RoIHeads):
    """RoI heads: background proposals inside ignore regions are not sampled; optional GIoU loss."""

    ignore_ioa_thresh = 0.5
    box_loss = "smooth_l1"
    giou_weight = 10.0

    def forward(self, features, proposals, image_shapes, targets=None):
        targets, self._ignore = split_ignore(targets)
        # RoIHeads.forward calls the module-level `fastrcnn_loss`; choose it for THIS model each call
        _roi_mod.fastrcnn_loss = (partial(giou_fastrcnn_loss, weight=self.giou_weight)
                                  if self.box_loss == "giou" else _ORIG_FASTRCNN_LOSS)
        return super().forward(features, proposals, image_shapes, targets)

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched_idxs, labels = super().assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        for lab, props, ign in zip(labels, proposals, self._ignore or [None] * len(labels)):
            if ign is not None and len(ign):
                lab[(lab == 0) & (max_ioa(props, ign) > self.ignore_ioa_thresh)] = -1
        return matched_idxs, labels


# ----------------------------------------------------------------------------- GIoU box loss
_ORIG_FASTRCNN_LOSS = _roi_mod.fastrcnn_loss
_BBOX_CLIP = math.log(1000.0 / 16)


def _decode_unit(deltas: torch.Tensor, weights=(10.0, 10.0, 5.0, 5.0)) -> torch.Tensor:
    """Decode (dx,dy,dw,dh) w.r.t. a unit reference box at the origin.

    IoU / GIoU are invariant to translating and scaling both boxes by the same reference box, so
    decoding prediction and target against the SAME unit box gives the exact GIoU of the real boxes
    (no need for the proposals, which fastrcnn_loss does not receive). Weights = torchvision default.
    """
    wx, wy, ww, wh = weights
    dx, dy = deltas[:, 0] / wx, deltas[:, 1] / wy
    dw = (deltas[:, 2] / ww).clamp(max=_BBOX_CLIP)
    dh = (deltas[:, 3] / wh).clamp(max=_BBOX_CLIP)
    w, h = torch.exp(dw), torch.exp(dh)
    return torch.stack([dx - w / 2, dy - h / 2, dx + w / 2, dy + h / 2], dim=1)


def giou_fastrcnn_loss(class_logits, box_regression, labels, regression_targets, weight: float = 10.0):
    """Drop-in for torchvision's fastrcnn_loss: same CE term, GIoU instead of smooth-L1.

    Normalised like the original (sum over positives / number of sampled RoIs). weight=10 follows
    mmdetection configs/faster_rcnn/faster-rcnn_r50_fpn_giou_1x_coco.py (GIoULoss, loss_weight=10).
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)
    pos = torch.where(labels > 0)[0]
    n = class_logits.shape[0]
    box_regression = box_regression.reshape(n, box_regression.size(-1) // 4, 4)
    pred = _decode_unit(box_regression[pos, labels[pos]].float())
    tgt = _decode_unit(regression_targets[pos].float())
    box_loss = weight * generalized_box_iou_loss(pred, tgt, reduction="sum") / labels.numel()
    return classification_loss, box_loss


# ----------------------------------------------------------------------------- builder
def make_anchor_generator(sizes, ratios, n_levels: int = 5) -> AnchorGenerator:
    """sizes: one number per FPN level (P2..P6) or a list per level; ratios = h/w, shared by levels."""
    sizes = [tuple(s) if isinstance(s, (list, tuple)) else (s,) for s in sizes]
    assert len(sizes) == n_levels, f"need {n_levels} anchor size groups (P2..P6), got {len(sizes)}"
    return AnchorGenerator(sizes=tuple(sizes), aspect_ratios=(tuple(ratios),) * n_levels)


def use_certifi_ca() -> None:
    """Point urllib (used by torch.hub to download weights) at certifi's CA bundle.

    python.org builds of Python on macOS ship without root certificates, so the weight download
    fails with CERTIFICATE_VERIFY_FAILED. Respects an SSL_CERT_FILE that is already set.
    """
    import os

    try:
        import certifi
    except ImportError:
        return
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())


def build_faster_rcnn(mcfg: dict, ignore_ioa: float = 0.5):
    """fasterrcnn_resnet50_fpn_v2 with COCO weights and a new (8 + background)-class box predictor."""
    use_certifi_ca()
    weights = mcfg.get("weights", "DEFAULT")  # "DEFAULT" = COCO_V1; None = ImageNet backbone only
    kwargs = dict(
        min_size=tuple(mcfg["min_size_train"]), max_size=mcfg["max_size"],
        rpn_pre_nms_top_n_train=mcfg["rpn_pre_nms_top_n_train"],
        rpn_pre_nms_top_n_test=mcfg["rpn_pre_nms_top_n_test"],
        rpn_post_nms_top_n_train=mcfg["rpn_post_nms_top_n_train"],
        rpn_post_nms_top_n_test=mcfg["rpn_post_nms_top_n_test"],
        rpn_batch_size_per_image=mcfg["rpn_batch_size_per_image"],
        rpn_positive_fraction=mcfg["rpn_positive_fraction"],
        box_batch_size_per_image=mcfg["box_batch_size_per_image"],
        box_positive_fraction=mcfg["box_positive_fraction"],
        box_nms_thresh=mcfg["box_nms_thresh"],
        box_score_thresh=mcfg["box_score_thresh"],
        box_detections_per_img=mcfg["box_detections_per_img"],
    )
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone=None if weights else "DEFAULT",
        trainable_backbone_layers=mcfg.get("trainable_backbone_layers", 3), **kwargs)

    # anchors: keep the pretrained RPN head if the number of anchors per location is unchanged (3)
    ag = make_anchor_generator(mcfg["anchor_sizes"], mcfg["aspect_ratios"])
    model.rpn.anchor_generator = ag
    n_anchors = ag.num_anchors_per_location()[0]
    if n_anchors != model.rpn.head.cls_logits.out_channels:
        print(f"[model] {n_anchors} anchors/location -> new (randomly initialised) RPN head")
        model.rpn.head = RPNHead(model.backbone.out_channels, n_anchors, conv_depth=2)

    # new classifier: 8 classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)

    # ignore-aware RPN / RoI heads (same modules and weights, only the target assignment differs)
    model.rpn.__class__ = IgnoreAwareRPN
    model.roi_heads.__class__ = IgnoreAwareRoIHeads
    model.rpn.ignore_ioa_thresh = model.roi_heads.ignore_ioa_thresh = ignore_ioa
    model.roi_heads.box_loss = mcfg.get("box_loss", "smooth_l1")
    model.roi_heads.giou_weight = mcfg.get("giou_weight", 10.0)
    model.supports_ignore = True  # read by train_tv (models without it get the ignore boxes stripped)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[model] fasterrcnn_resnet50_fpn_v2 | params {n_params:.1f}M (trainable {n_train:.1f}M) | "
          f"anchors {mcfg['anchor_sizes']} x {mcfg['aspect_ratios']} | box loss {model.roi_heads.box_loss}")
    return model


def set_input_size(model, sizes) -> None:
    """Short-side sizes of the model transform: tuple = random choice per image in train mode
    (scale jitter), the LAST value is used in eval mode (torchvision GeneralizedRCNNTransform)."""
    model.transform.min_size = tuple(int(s) for s in (sizes if isinstance(sizes, (list, tuple)) else [sizes]))


def set_bn_eval(model) -> None:
    """Freeze BatchNorm running stats (batch of 2 images is too small to estimate them)."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()


# ----------------------------------------------------------------------------- RPN recall
@torch.no_grad()
def rpn_recall(model, index, gt: dict, test_size: int, device, n_images: int = 100,
               top_ns=(100, 300, 1000, 2000), iou_thrs=(0.5, 0.7), seed: int = 0, max_long=None):
    """Recall of RPN proposals vs GT, per COCO size bucket. gt = src.analysis.gt_from_coco(...).

    Returns a DataFrame [top_n, iou, bucket, recall, n_gt]. Shows whether missed objects are lost
    already at the proposal stage (anchors / RPN) or later in the box head.
    """
    import pandas as pd

    from .analysis import size_bucket_np
    from .dataset import load_rgb, pre_resize

    model.eval()
    set_input_size(model, [test_size])
    rows = index.sample(min(n_images, len(index)), random_state=seed).itertuples(index=False)
    best = {(n, t): [] for n in top_ns for t in iou_thrs}
    sizes = []
    for row in rows:
        g = gt.get(row.key)
        if g is None or len(g["boxes"]) == 0:
            continue
        img, s = pre_resize(load_rgb(row.img_path), test_size, max_long)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).to(device).float().div_(255)
        images, _ = model.transform([x])
        feats = model.backbone(images.tensors)
        props, _ = model.rpn(images, feats)
        p = props[0]  # sorted by objectness (batched_nms keeps descending score order)
        (h_in, w_in), (h0, w0) = images.image_sizes[0], img.shape[:2]
        p = p * torch.tensor([w0 / w_in, h0 / h_in, w0 / w_in, h0 / h_in], device=p.device) / s
        gtb = torch.as_tensor(g["boxes"], dtype=torch.float32, device=p.device)
        iou = box_iou(gtb, p) if len(p) else torch.zeros((len(gtb), 1), device=p.device)
        for n in top_ns:
            m = iou[:, :n].max(dim=1).values.cpu().numpy() if iou.shape[1] else np.zeros(len(gtb))
            for t in iou_thrs:
                best[(n, t)].append(m >= t)
        wh = g["boxes"][:, 2:] - g["boxes"][:, :2]
        sizes.append(size_bucket_np(wh[:, 0] * wh[:, 1]))
    sizes = np.concatenate(sizes) if sizes else np.array([])
    out = []
    for (n, t), hits in best.items():
        hits = np.concatenate(hits) if hits else np.array([], bool)
        for b in ["all", "small", "medium", "large"]:
            m = np.ones_like(hits) if b == "all" else sizes == b
            out.append({"top_n": n, "iou": t, "bucket": b, "recall": float(hits[m].mean()) if m.sum() else np.nan,
                        "n_gt": int(m.sum())})
    return pd.DataFrame(out)
