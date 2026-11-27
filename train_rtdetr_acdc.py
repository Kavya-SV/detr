import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn


def install_package(package: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])


try:
    from ultralytics import RTDETR
except Exception:
    print("Installing ultralytics...")
    install_package("ultralytics")
    from ultralytics import RTDETR

try:
    import kagglehub
except Exception:
    print("Installing kagglehub...")
    install_package("kagglehub")
    import kagglehub

WEATHER_TO_ID = {
    "sunny": 0,
    "rain": 1,
    "fog": 2,
    "snow": 3,
    "night": 4,
}

CATEGORY_MAPPING = {
    24: 0,
    25: 1,
    26: 2,
    27: 3,
    28: 4,
    31: 5,
    32: 6,
    33: 7,
}

NAMES = {
    0: "person",
    1: "rider",
    2: "car",
    3: "truck",
    4: "bus",
    5: "train",
    6: "motorcycle",
    7: "bicycle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR on ACDC dataset on Windows GPU")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--project", default="RTDETR_Baseline")
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent / "acdc_output"))
    parser.add_argument("--dataset-root", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--weights", default="rtdetr-l.pt")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--weather", choices=["rain", "fog", "snow", "night", "both"],default="rain"
)
    return parser.parse_args()


def ensure_dirs(path: Path) -> None:
    for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
        (path / folder).mkdir(parents=True, exist_ok=True)


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w = w / img_w
    h = h / img_h
    return x_center, y_center, w, h


def convert_coco_to_yolo(dataset_dir: Path, weather: str, output_dir: Path, category_mapping: dict) -> None:
    label_dir = dataset_dir / "acdc" / "labels" / weather
    train_json = label_dir / f"instancesonly_{weather}_train_gt_detection.json"
    val_json = label_dir / f"instancesonly_{weather}_val_gt_detection.json"

    if not train_json.exists() or not val_json.exists():
        raise FileNotFoundError(f"Expected COCO files not found in {label_dir}")

    def process_split(json_path: Path, split: str) -> None:
        with open(json_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        image_dict = {img["id"]: img for img in coco["images"]}
        annotations_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            annotations_by_image[ann["image_id"]].append(ann)

        image_root = dataset_dir / "acdc" / "images"
        target_images_dir = output_dir / "images" / split
        target_labels_dir = output_dir / "labels" / split
        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_labels_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_id, anns in annotations_by_image.items():
            img_info = image_dict[img_id]
            img_file = img_info["file_name"]
            src_img = image_root / img_file
            dst_img = target_images_dir / os.path.basename(img_file)
            shutil.copy2(src_img, dst_img)

            txt_name = os.path.basename(img_file).replace(".png", ".txt")
            txt_path = target_labels_dir / txt_name
            with open(txt_path, "w", encoding="utf-8") as f:
                for ann in anns:
                    if ann["category_id"] not in category_mapping:
                        continue
                    cls = category_mapping[ann["category_id"]]
                    x_center, y_center, w, h = coco_to_yolo_bbox(ann["bbox"], img_info["width"], img_info["height"])
                    f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
            count += 1

        print(f"[{weather}/{split}] Converted {count} images")

    process_split(train_json, "train")
    process_split(val_json, "val")


def write_dataset_yaml(output_dir: Path) -> Path:
    yaml_path = output_dir / "dataset.yaml"
    yaml_content = f"""path: {output_dir}
train: images/train
val: images/val
names:
  0: person
  1: rider
  2: car
  3: truck
  4: bus
  5: train
  6: motorcycle
  7: bicycle
"""
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return yaml_path


def download_dataset() -> Path:
    attempts = 3
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading dataset (attempt {attempt}/{attempts})...")
            dataset_path = kagglehub.dataset_download("khalilusmanuk/acdc-clean-dataset-for-training", force_download=False)
            return Path(dataset_path).resolve()
        except Exception as exc:
            last_error = exc
            print(f"Dataset download attempt {attempt}/{attempts} failed: {exc}")
            if attempt < attempts:
                print("Retrying in 30 seconds...")
                time.sleep(30)
    raise RuntimeError(f"Dataset download failed after {attempts} attempts") from last_error



# Weather label mapping (must match SCEWeatherDetector num_weather_classes=4)
WEATHER_LABEL = {"rain": 0, "fog": 1, "snow": 2, "night": 3}


class WeatherLossCallback:
    """
    Ultralytics trainer callback that adds a weather classification loss
    on top of the standard RT-DETR detection loss.

    The decoder stores weather_logits on the head module after every forward
    pass, so we read those and compute CrossEntropyLoss against the known
    weather label for this training run.

    Args:
        weather (str): Which weather is being trained ('rain', 'fog', 'snow', 'night').
        weight (float): Scale factor for the weather loss added to the total loss.
    """

    WEATHER_LOSS_WEIGHT = 0.1

    def __init__(self, weather: str, weight: float = 0.1):
        if weather not in WEATHER_LABEL:
            raise ValueError(f"Unknown weather '{weather}'. Expected one of {list(WEATHER_LABEL)}.")
        self.weather_id = WEATHER_LABEL[weather]
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss()
        self._last_weather_loss: float = 0.0

    def on_train_batch_end(self, trainer) -> None:
        """Called by ultralytics after each training batch."""
        try:
            # Reach into the model to get the last head module (RTDETRDecoder)
            head = trainer.model.model[-1]
            if not hasattr(head, "weather_logits") or head.weather_logits is None:
                return

            logits = head.weather_logits           # (B, num_weather_classes)
            B = logits.shape[0]
            labels = torch.full((B,), self.weather_id, dtype=torch.long, device=logits.device)

            weather_loss = self.ce_loss(logits, labels) * self.weight
            self._last_weather_loss = weather_loss.item()

            # Add to the trainer's recorded loss (does NOT backprop — gradients are
            # To affect gradients, we hook on_before_zero_grad instead.
            if hasattr(trainer, "loss") and trainer.loss is not None:
                trainer.loss = trainer.loss + weather_loss.detach()
                
            # --- Diagnostic Printing ---
            self._batch_idx = getattr(self, '_batch_idx', 0) + 1
            if self._batch_idx in [1, 10, 50] or self._batch_idx == getattr(trainer, 'batches', 9999):
                if hasattr(head, "stat_total") and head.stat_total > 0:
                    t = head.stat_total
                    print(f"\n--- Epoch {trainer.epoch + 1}, Batch {self._batch_idx} ---")
                    print(f"High %: {head.stat_high / t * 100:.2f}%")
                    print(f"Medium %: {head.stat_med / t * 100:.2f}%")
                    print(f"Low %: {head.stat_low / t * 100:.2f}%")
                    print(f"Recovered %: {head.stat_recovered / t * 100:.2f}%")
                    print(f"Final kept %: {(head.stat_high + head.stat_recovered) / t * 100:.2f}%")
                    print(f"Final pruned %: {t - (head.stat_high + head.stat_recovered)} / {t} ({(t - (head.stat_high + head.stat_recovered)) / t * 100:.2f}%)")
                    print(f"tau_low: {getattr(head, 'last_tau_low', -1):.4f}, tau_high: {getattr(head, 'last_tau_high', -1):.4f}")
                    print(f"Mean score: {head.stat_score_sum / t:.4f}, Min: {head.stat_score_min:.4f}, Max: {head.stat_score_max:.4f}\n")

        except Exception as exc:
            # Never crash training due to a weather-loss error
            print(f"[WeatherLossCallback] Warning: {exc}")

    def on_train_epoch_end(self, trainer) -> None:
        """Log average weather loss at epoch end and print Chokam stats."""
        print(f"[WeatherLoss] Last batch weather loss: {self._last_weather_loss:.4f}")
        
        try:
            head = trainer.model.model[-1]
            if hasattr(head, "stat_total") and head.stat_total > 0:
                total = head.stat_total
                high = head.stat_high
                med = head.stat_med
                low = head.stat_low
                recovered = head.stat_recovered
                
                print("\n--- Epoch Chokam Routing Statistics ---")
                print(f"Total Tokens: {total}")
                print(f"High: {high / total * 100:.2f}%")
                print(f"Medium: {med / total * 100:.2f}%")
                print(f"Low: {low / total * 100:.2f}%")
                print(f"Recovered: {recovered / total * 100:.2f}%")
                print(f"Finally kept: {(high + recovered) / total * 100:.2f}%")
                print(f"Finally pruned: {total - (high + recovered)} / {total} ({(total - (high + recovered)) / total * 100:.2f}%)\n")
                
                # Reset for next epoch
                head.stat_total = 0
                head.stat_high = 0
                head.stat_med = 0
                head.stat_low = 0
                head.stat_recovered = 0
                head.stat_score_sum = 0
                head.stat_score_min = 1.0
                head.stat_score_max = 0.0
        except Exception as exc:
            pass
        self._batch_idx = 0


def train_weather(model_name: str, weather: str, dataset_path: Path, output_root: Path, args: argparse.Namespace) -> None:
    weather_output_dir = output_root / weather
    weather_output_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs(weather_output_dir)
    convert_coco_to_yolo(dataset_path, weather, weather_output_dir, CATEGORY_MAPPING)
    yaml_path = write_dataset_yaml(weather_output_dir)

    if args.device:
        device = args.device
    else:
        device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Training {weather} with device={device}")
    model = RTDETR(model_name)

    # Register weather loss callback before training starts
    weather_cb = WeatherLossCallback(weather=weather, weight=0.1)

    model.add_callback("on_train_batch_end", weather_cb.on_train_batch_end)
    model.add_callback("on_train_epoch_end", weather_cb.on_train_epoch_end)

    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=device,
        workers=0,
        project=str(args.project),
        name=weather,
        exist_ok=True,
        resume=args.resume,
    )


def main() -> None:
    args = parse_args()
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    project_root = Path(__file__).resolve().parent
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if args.dataset_root:
        dataset_path = Path(args.dataset_root).resolve()
    else:
        try:
            dataset_path = download_dataset()
        except Exception as e:
            print("Dataset download failed.")
            print("Make sure kagglehub is installed and your Kaggle credentials are configured.")
            print("If you already have the dataset locally, rerun with --dataset-root pointing to its folder.")
            print(e)
            return

    print("Dataset path:", dataset_path)

    if not (dataset_path / "acdc").exists():
        raise FileNotFoundError(f"The expected 'acdc' folder was not found in {dataset_path}")

    if args.weather in ["rain", "both"]:
        train_weather(args.weights, "rain", dataset_path, output_root, args)
    if args.weather in ["fog", "both"]:
        train_weather(args.weights, "fog", dataset_path, output_root, args)
    if args.weather in ["snow", "both"]:
        train_weather(args.weights, "snow", dataset_path, output_root, args)
    if args.weather in ["night", "both"]:
        train_weather(
        args.weights,
        "night",
        dataset_path,
        output_root,
        args
    )
if __name__ == "__main__":
    main()
