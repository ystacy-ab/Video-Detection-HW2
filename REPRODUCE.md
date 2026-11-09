# How to reproduce

## Layout
```
configs/base.yaml              shared settings: seed 42, frame step 6 (all models), eval / submission params, W&B flag
configs/frcnn_*.yaml           baseline -> giou, sampler (ablations) -> tuned        (each inherits via `_base_`)
configs/retinanet_*.yaml       baseline -> anchors -> assign -> tuned
configs/yolo_*.yaml            baseline (640) -> hires (1024) -> p2 (1024 + P2 head)
configs/rtdetr_*.yaml          baseline (300 queries, 640) -> q500 -> tuned (500 queries, 960)

src/utils.py                   config (+ `_base_` inheritance), seeds, box formats, prediction I/O, experiments.csv, W&B
src/eval.py                    the single COCO evaluation (maxDets 500, ignore = crowd), per-clip / per-density mAP, speed
src/submission.py              CSV writer + validator
src/dataset.py                 frame index, label readers, torchvision Dataset (ignore regions masked in the loss)
src/convert.py                 Ultralytics dataset (ignore regions painted gray)
src/anchors.py                 k-means anchors, anchor coverage of GT sizes
src/models_tv.py               Faster R-CNN / RetinaNet builders, ignore-aware assignment, GIoU, focal params, RPN recall
src/train_tv.py                torchvision train loop (AMP, accumulation, resumable), predict, run_experiment
src/models_ul.py               Ultralytics YOLOv8 / RT-DETR: data, resumable training, predict, TTA, tiled inference
src/ensemble.py                Weighted Boxes Fusion of saved predictions
src/analysis.py                failure analysis (matching, confusion, recall by size / occlusion)
src/viz.py                     drawing helpers

notebooks/00_sanity_checks.ipynb   eval + submission self-tests                         CPU
notebooks/01_eda.ipynb             EDA, k-means anchors, conversion check                CPU
notebooks/02_faster_rcnn.ipynb     two-stage anchor-based + failure analysis + RPN recall   GPU
notebooks/03_retinanet.ipynb       one-stage anchor-based                                GPU
notebooks/04_yolo.ipynb            one-stage anchor-free                                 GPU
notebooks/05_rtdetr.ipynb          query-based                                           GPU
notebooks/06_final_pipeline.ipynb  summary of all models, TTA / tiles, WBF, final submissions

results/experiments.csv        one row per experiment (val metrics, speed, train time; public_lb filled by hand)
results/*.csv, results/figures tables and figures written by the notebooks
```

## Protocol (same for every model)
- Split by clip: official train (56 clips) / val (7 clips). Train frames: every 6th frame of each clip (~4.0k).
- Score: `src.eval.evaluate` on all 2846 val frames, COCO mAP@[.5:.95] with maxDets 500, ignore regions as crowd.
- Weights scored = end of the schedule (`last.pt`), no checkpoint selection on val.
- Speed: batch 1, end to end (JPEG decode + resize + model + NMS), 10 warm-up + 100 timed images.
- Seed 42 (`configs/base.yaml`), seeds set in every notebook.

## Setup
**Kaggle**: new notebook -> File -> Import notebook -> add the competition data as input ->
Accelerator **GPU T4**, **Internet on**, Persistence **Files only**.
**Colab**: Runtime -> T4 GPU; put the competition data in `/content/data` (folder with `test_images.csv`).
Outputs go to `MyDrive/hw2_outputs`.

The first code cells detect the environment, clone this repository and install `requirements.txt`
(`torch` / `torchvision` already come with Kaggle / Colab). Locally: `pip install -r requirements.txt`,
data in `data/`, outputs in `outputs/`.

## Order and expected runtime (T4)
| Notebook | Runs | Time | Main outputs |
|---|---|---|---|
| 00, 01 | - | CPU, ~30 min | `results/eda_stats.json`, EDA figures |
| 02 | frcnn_baseline, frcnn_giou, frcnn_sampler, frcnn_tuned | ~9 h | `submissions/frcnn_tuned.csv` |
| 03 | retinanet_baseline, _anchors, _assign, _tuned | ~9 h | `submissions/retinanet_tuned.csv` |
| 04 | yolo_baseline, yolo_hires, yolo_p2 | ~6 h | `submissions/yolo_p2.csv` |
| 05 | rtdetr_baseline, rtdetr_q500, rtdetr_tuned | ~8 h | `submissions/rtdetr_tuned.csv` |
| 06 | TTA, tiles, WBF candidates | ~1 h | `submissions/final_1.csv`, `final_2.csv` |

Times are estimates; each notebook starts with a ~5 min smoke test whose it/s gives the real speed.
06 needs the tuned run of each family from 02-05 (their val and test predictions).

Paths below are relative to `WORK_DIR` (Kaggle: `/kaggle/working/outputs`):
- `tv/<exp>/` (torchvision) and `ul/<exp>/` (Ultralytics): checkpoints, `history.csv` / `results.csv`, `final.json`
- `preds/<exp>_val.pkl`, `preds/<exp>_test.pkl`: canonical predictions, reused by 06 without a GPU
- `submissions/<exp>.csv`: validated submission files

## Resuming and splitting the work
- Every run checkpoints after each epoch. If a session stops, run the same notebook again with the same
  `WORK_DIR`: unfinished runs resume, finished runs (`final.json`) are only reloaded and never logged twice.
- A Kaggle session is limited to 12 h. To spread one notebook over several sessions, edit `RUN = [...]` in the
  config cell (e.g. only `["frcnn_baseline", "frcnn_giou"]`). Runs not in `RUN` are skipped.
- Without a GPU, training cells are skipped and the CPU sections (data checks, analysis of saved predictions) still run.

## Submission
Download `WORK_DIR/submissions/final_1.csv` (best candidate by val mAP; `final_2.csv` = second best) and upload
it on the competition page. Enter the public LB score by hand in `results/experiments.csv` (`public_lb`).
