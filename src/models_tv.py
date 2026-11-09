"""torchvision detectors: Faster R-CNN (two-stage) and RetinaNet (one-stage), both anchor-based.

Faster R-CNN: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal
Networks", NeurIPS 2015. FPN: Lin et al., "Feature Pyramid Networks for Object Detection", CVPR 2017.
"v2" weights/architecture: Li et al., "Benchmarking Detection Transfer Learning with Vision
Transformers" (2021) as implemented in torchvision (`fasterrcnn_resnet50_fpn_v2`).
GIoU loss: Rezatofighi et al., "Generalized Intersection over Union", CVPR 2019.
RetinaNet / focal loss: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
(torchvision `retinanet_resnet50_fpn_v2`: GroupNorm heads + GIoU box loss).

Model config keys (section `model:` in configs/frcnn_*.yaml) are documented in
configs/frcnn_baseline.yaml.
"""
from __future__ import annotations

import math
from functools import partial as _partial
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import roi_heads as _roi_mod
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import (RetinaNet, RetinaNetClassificationHead,
                                                    RetinaNetRegressionHead)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead
from torchvision.ops import box_iou, generalized_box_iou_loss, sigmoid_focal_loss

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


class IgnoreAwareRetinaNet(RetinaNet):
    """RetinaNet whose background anchors inside ignore regions are excluded from the focal loss.

    Same assignment as torchvision (IoU >= fg -> positive, < bg -> negative, in between -> ignored),
    plus: negative anchors with IoA > ignore_ioa_thresh inside an ignore box -> ignored (-2).
    """

    ignore_ioa_thresh = 0.5

    def compute_loss(self, targets, head_outputs, anchors):
        targets, ignore = split_ignore(targets)
        below, between = self.proposal_matcher.BELOW_LOW_THRESHOLD, self.proposal_matcher.BETWEEN_THRESHOLDS
        matched_idxs = []
        for anc, t, ign in zip(anchors, targets, ignore):
            if t["boxes"].numel() == 0:
                m = torch.full((anc.size(0),), below, dtype=torch.int64, device=anc.device)
            else:
                m = self.proposal_matcher(box_iou(t["boxes"], anc))
            if len(ign):
                m[(m == below) & (max_ioa(anc, ign) > self.ignore_ioa_thresh)] = between
            matched_idxs.append(m)
        return self.head.compute_loss(targets, head_outputs, anchors, matched_idxs)


class FocalClassificationHead(RetinaNetClassificationHead):
    """torchvision classification head with configurable focal-loss alpha / gamma (defaults 0.25 / 2)."""

    focal_alpha = 0.25
    focal_gamma = 2.0

    def compute_loss(self, targets, head_outputs, matched_idxs):
        losses = []
        for t, logits, m in zip(targets, head_outputs["cls_logits"], matched_idxs):
            fg = m >= 0
            target = torch.zeros_like(logits)
            target[fg, t["labels"][m[fg]]] = 1.0
            valid = m != self.BETWEEN_THRESHOLDS
            loss = sigmoid_focal_loss(logits[valid], target[valid], alpha=self.focal_alpha,
                                      gamma=self.focal_gamma, reduction="sum")
            losses.append(loss / max(1, int(fg.sum())))
        return sum(losses) / len(targets)


def build_retinanet(mcfg: dict, ignore_ioa: float = 0.5):
    """retinanet_resnet50_fpn_v2 with COCO weights, new classification head (8 classes + unused 0).

    Anchors: per FPN level P3..P7 one base size x anchor_scales (octaves) x aspect_ratios.
    Keeping 3 scales x 3 ratios = 9 anchors / location keeps the pretrained box-regression head.
    """
    use_certifi_ca()
    weights = mcfg.get("weights", "DEFAULT")
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        weights=weights, weights_backbone=None if weights else "DEFAULT",
        trainable_backbone_layers=mcfg.get("trainable_backbone_layers", 3),
        min_size=tuple(mcfg["min_size_train"]), max_size=mcfg["max_size"],
        score_thresh=mcfg["box_score_thresh"], nms_thresh=mcfg["box_nms_thresh"],
        detections_per_img=mcfg["box_detections_per_img"], topk_candidates=mcfg.get("topk_candidates", 1000),
        fg_iou_thresh=mcfg["fg_iou_thresh"], bg_iou_thresh=mcfg["bg_iou_thresh"])

    sizes = [tuple(int(round(b * k)) for k in mcfg["anchor_scales"]) for b in mcfg["anchor_sizes"]]
    ag = make_anchor_generator(sizes, mcfg["aspect_ratios"])
    model.anchor_generator = ag
    n_anchors = ag.num_anchors_per_location()[0]
    gn = _partial(torch.nn.GroupNorm, 32)
    if n_anchors != model.head.regression_head.bbox_reg.out_channels // 4:  # pretrained head: 9 anchors
        print(f"[model] {n_anchors} anchors/location -> new (randomly initialised) box-regression head")
        reg = RetinaNetRegressionHead(model.backbone.out_channels, n_anchors, norm_layer=gn)
        reg._loss_type = "giou"
        model.head.regression_head = reg
    cls_head = FocalClassificationHead(model.backbone.out_channels, n_anchors, NUM_CLASSES + 1, norm_layer=gn)
    cls_head.focal_alpha, cls_head.focal_gamma = mcfg.get("focal_alpha", 0.25), mcfg.get("focal_gamma", 2.0)
    model.head.classification_head = cls_head

    model.__class__ = IgnoreAwareRetinaNet
    model.ignore_ioa_thresh = ignore_ioa
    model.supports_ignore = True
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] retinanet_resnet50_fpn_v2 | params {n_params:.1f}M | anchors {sizes} x {mcfg['aspect_ratios']} | "
          f"IoU fg/bg {mcfg['fg_iou_thresh']}/{mcfg['bg_iou_thresh']} | focal a={cls_head.focal_alpha} g={cls_head.focal_gamma}")
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


# ----------------------------------------------------------------------------- anchor assignment statistics
@torch.no_grad()
def anchor_assignment_stats(mcfg: dict, index, n_images: int = 30, seed: int = 0,
                            strides=(8, 16, 32, 64, 128), ignore_ioa: float = 0.5):
    """RetinaNet-style assignment on real frames (CPU, no network): anchors per image, positives,
    negatives, ignored (between thresholds) and negatives removed by the ignore regions.
    Shows the foreground/background imbalance a one-stage detector faces without a proposal stage.
    """
    import pandas as pd
    from torchvision.models.detection._utils import Matcher
    from torchvision.models.detection.image_list import ImageList

    from .anchors import input_scale
    from .dataset import read_ignore, read_yolo_labels

    sizes = [tuple(int(round(b * k)) for k in mcfg.get("anchor_scales", [1.0])) for b in mcfg["anchor_sizes"]]
    ag = make_anchor_generator(sizes, mcfg["aspect_ratios"])
    matcher = Matcher(mcfg["fg_iou_thresh"], mcfg["bg_iou_thresh"], allow_low_quality_matches=True)
    rows = []
    for r in index.sample(min(n_images, len(index)), random_state=seed).itertuples():
        s = input_scale(r.width, r.height, mcfg["min_size_test"], mcfg["max_size"])
        H, W = int(round(r.height * s)), int(round(r.width * s))
        feats = [torch.zeros(1, 1, math.ceil(H / st), math.ceil(W / st)) for st in strides]
        anchors = ag(ImageList(torch.zeros(1, 3, H, W), [(H, W)]), feats)[0]
        gt = torch.from_numpy(read_yolo_labels(r.label_path, r.width, r.height)[0] * s)
        ign = torch.from_numpy(read_ignore(r.ignore_path, r.width, r.height) * s)
        if len(gt) == 0:
            continue
        m = matcher(box_iou(gt, anchors))
        neg = m == Matcher.BELOW_LOW_THRESHOLD
        neg_in_ignore = int((neg & (max_ioa(anchors, ign) > ignore_ioa)).sum()) if len(ign) else 0
        pos = m >= 0
        n_per_gt = torch.bincount(m[pos], minlength=len(gt))
        rows.append({"anchors": len(anchors), "gt": len(gt), "positives": int(pos.sum()),
                     "negatives": int(neg.sum()), "between": int((m == Matcher.BETWEEN_THRESHOLDS).sum()),
                     "neg_in_ignore": neg_in_ignore, "neg_per_pos": float(neg.sum()) / max(int(pos.sum()), 1),
                     "gt_with_1_anchor_%": float((n_per_gt <= 1).float().mean() * 100)})
    return pd.DataFrame(rows).mean().round(1)
