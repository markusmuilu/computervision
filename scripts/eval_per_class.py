"""Evaluate the fine-tuned RF-DETR on the test split: overall mAP and per-class AP.

Same metric call as the evaluation in notebooks/train_and_evaluate.ipynb, run from the repo root:

    python scripts/eval_per_class.py
"""
import argparse
from collections import Counter

import supervision as sv
from PIL import Image
from rfdetr import RFDETRMedium
from supervision.metrics import MeanAveragePrecision
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--weights", default="models/object_detection.pth")
parser.add_argument("--dataset", default="notebooks/basketball-player-detection-3-10")
args = parser.parse_args()

ds_test = sv.DetectionDataset.from_coco(
    images_directory_path=f"{args.dataset}/test",
    annotations_path=f"{args.dataset}/test/_annotations.coco.json",
)
model = RFDETRMedium(pretrain_weights=args.weights)

targets, predictions = [], []
for path, _, annotations in tqdm(ds_test):
    # threshold=0: mAP ranks every detection by confidence, so none are dropped up front
    predictions.append(model.predict(Image.open(path), threshold=0))
    targets.append(annotations)

result = MeanAveragePrecision().update(predictions, targets).compute()
print(result)

# ap_per_class: one row per class present in the ground truth, one column per IoU
# threshold (0.50, 0.55, ..., 0.95). Row mean is AP@[.50:.95], column 0 is AP@.50.
gt_counts = Counter(int(class_id) for t in targets for class_id in t.class_id)
print(f"\n{'class':<22}{'AP@[.50:.95]':>13}{'AP@.50':>8}{'GT boxes':>10}")
for class_id, ap in sorted(zip(result.matched_classes, result.ap_per_class), key=lambda r: -r[1].mean()):
    print(f"{ds_test.classes[class_id]:<22}{ap.mean():>13.3f}{ap[0]:>8.3f}{gt_counts[class_id]:>10}")

unscored = [name for i, name in enumerate(ds_test.classes) if i > 0 and gt_counts[i] == 0]
print(f"\nNo ground truth in the test split, so not scored: {', '.join(unscored) or 'none'}")
