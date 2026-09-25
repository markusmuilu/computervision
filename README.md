# Automated basketball game statistics with computer vision

Player, ball and event detection on NBA broadcast footage, with tracking, drawing and a
measured evaluation. **Unfinished** — started December 2025, paused, picked back up
September 2026.

![One frame of pipeline output: players, referees, ball, jersey numbers and rim, each with a track id](docs/demo.jpg)

## What it does

A detect → track → draw → save pipeline over broadcast video.

| Stage | Implementation |
|---|---|
| Detection | **RF-DETR Medium**, fine-tuned. 30 epochs, batch 5, resolution 576, seed 42, trained on a Colab GPU |
| Tracking | **`supervision.ByteTrack`**, track ids carried across frames |
| Inference | Batched at 20 frames, BGR→RGB, results cached to a pickle stub so detection does not re-run |
| Output | Per-class colours, transparent overlays, `class #id` labels, written to video |

Ten classes, which are events rather than only objects: `ball`, `ball-in-basket`, `number`,
`player`, `player-in-possession`, `player-jump-shot`, `player-layup-dunk`, `player-shot-block`,
`referee`, `rim`.

## Results

Evaluated with `supervision.metrics.MeanAveragePrecision` over a **94-image held-out test split**.
`threshold=0` at prediction time, because mAP needs the full ranked detection list rather than a
thresholded one. Every number below comes from `scripts/eval_per_class.py` with the published
weights (September 2026, RTX 5070).

| Metric | Value |
|---|---|
| mAP @[IoU=0.50:0.95] | **0.561** |
| mAP @0.50 | **0.802** |
| mAP @0.75 | 0.630 |
| AP, small objects | **0.419** |
| AP, medium objects | 0.434 |
| AP, large objects | 0.683 |

The original December 2025 evaluation in `notebooks/train_and_evaluate.ipynb` reported 0.557 overall on the Colab
checkpoint. The headline agrees within 0.004 (a CPU run of the script gives 0.557 exactly), but
the size buckets differ by up to 0.03 (large 0.654 then, 0.683 now), most likely because the
published weights are a different checkpoint file from the same training run. The table above is
what the weights in this project actually produce.

### Per class

| Class | AP @[.50:.95] | AP @.50 | Test boxes |
|---|---|---|---|
| referee | 0.793 | 0.984 | 280 |
| player | 0.793 | 0.983 | 896 |
| player-shot-block | 0.689 | 0.857 | 14 — noisy |
| player-jump-shot | 0.582 | 0.769 | 16 — noisy |
| rim | 0.567 | 0.973 | 93 |
| player-in-possession | 0.520 | 0.619 | 18 — noisy |
| ball | 0.430 | 0.816 | 88 |
| number | 0.394 | 0.823 | 567 |
| ball-in-basket | 0.285 | 0.393 | 8 — very noisy |
| player-layup-dunk | not scored | — | 0 |

**The small-object weakness is both the ball and the jersey numbers.** Both sit near 0.4 AP under
the strict metric but above 0.8 at IoU 0.50: the model mostly finds them, and its boxes are not
tight enough for the higher IoU thresholds. With 567 test boxes the number result is solid; the
ball rests on 88.

Classes with fewer than about 20 test boxes swing noticeably from a single detection: the
action classes and `ball-in-basket` moved by up to 0.03 between a CPU and a GPU run of the same
weights. `player-layup-dunk` has no boxes in the test split, so the headline mAP is an average
over nine classes, not ten.

## What works and what does not

**Player and referee detection works.** Both are at 0.79 AP @[.50:.95] and 0.98 at IoU 0.50,
on hundreds of test boxes.

**The ball and jersey numbers are where it fails**, at 0.43 and 0.39. Those are exactly what a
box score depends on, since a shot cannot be attributed without knowing who took it or whether it
went in. That is why the automated-statistics goal in the title is not reached.

**Limits on every number above:**

- **The test split is in-distribution.** It comes from the same games and clips as training, so
  it measures fit, not generalisation to an unseen broadcast.
- **The tracking half is unmeasured.** No ID-switch count.
- **Small classes are noisy**, see the per-class table.

## Running it

Install PyTorch for your machine first ([pytorch.org](https://pytorch.org/get-started/locally/)),
then:

```
pip install -r requirements.txt
python main.py --input input_video/nba.mp4 --output output/result.avi
```

| Flag | Default | |
|---|---|---|
| `--input` | `input_video/nba.mp4` | input clip |
| `--output` | `output/result.avi` | annotated video, written at the input's frame rate |
| `--weights` | `BASKETBALL_CV_WEIGHTS`, else `models/object_detection.pth` | model weights |
| `--stub` | `stubs/object_track_stubs.pkl` | cache of tracked detections |
| `--fresh` | off | ignore the cache and re-run detection |

**The whole clip is held in memory**, so this suits clips of a few hundred frames: a minute of
1080p is already around 11 GB. Streaming frames through is future work.

To reproduce the evaluation, with the dataset extracted under `notebooks/`:

```
python scripts/eval_per_class.py --weights models/object_detection.pth
```

**The weights, dataset and video are not in the repository**, along with rendered output and the
virtualenv — see `.gitignore`. Reasons: the dataset is Roboflow's to distribute, weights and
video cannot be reviewed in a diff, and rendered output is regenerable.

- **Dataset:** [`basketball-player-detection-3`](https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo)
  version 10 from Roboflow Universe, exported in COCO format.
- **Weights:** RF-DETR Medium fine-tuned on that dataset. Not published; to get them, retrain
  with the training run in `notebooks/train_and_evaluate.ipynb`.
- **Input:** NBA broadcast clips.

## Future work — not started

None of this exists yet. The idea is that each step has published ground truth, since the NBA
publishes official box scores and shot charts:

1. **Cut and replay detection**, so only live play is counted. Broadcasts replay baskets, which
   would otherwise be counted twice.
2. **Final score**, from `ball-in-basket` events, checked against the official score.
3. **Shot chart**, via court homography (frame → court coordinates), checked against the NBA's
   published shot chart.
4. **Points per player**, which needs jersey number recognition and team assignment, checked
   against the official box score.

Also still to do for the existing half: an out-of-distribution test on footage from a different
game, ID-switch counting for the tracker, and streaming video instead of loading whole clips.

## Layout

```
main.py                                   pipeline entry point, CLI flags above
trackers/object_tracker.py                detection + ByteTrack, with stub caching
drawers/                                  overlays, boxes and labels, one colour per class
utils/                                    video read/write, stub load/save
scripts/eval_per_class.py                 mAP and per-class AP on the test split
notebooks/train_and_evaluate.ipynb        the Colab training run and original evaluation, outputs kept
notebooks/early_experiment_seg_preview.ipynb  an earlier 5-epoch try with RF-DETR Seg Preview, outputs cleared
docs/demo.jpg                             the frame shown above
```

## License

Code under the [MIT License](LICENSE). The dataset belongs to its Roboflow Universe publisher and
the footage to its broadcaster; neither is covered by this license or included in the repository.
