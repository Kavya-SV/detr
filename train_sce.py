"""
Dedicated Multi-Weather SCE Training Script
c:/Users/kavya_05/OneDrive/Desktop/detr/train_sce.py

Trains RT-DETR with the Multi-Scale SCEModule on the merged ACDC dataset
(chokam_dataset) containing all 4 weather conditions (rain, fog, snow, night).

Ensures TRUE gradient backpropagation into SCEModule by extending
RTDETRDetectionModel to compute CrossEntropyLoss directly within the loss graph.
"""

import argparse
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import yaml

from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, colorstr

from weather_dataset import WeatherRTDETRDataset


class WeatherAwareRTDETRModel(RTDETRDetectionModel):
    """
    RT-DETR model that includes Weather Cross-Entropy Loss in the main backprop graph.
    """

    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if nc is not None:
            self.nc = nc
        elif hasattr(self, "yaml") and isinstance(self.yaml, dict) and "nc" in self.yaml:
            self.nc = self.yaml["nc"]
        else:
            self.nc = 80
        self.weather_loss_weight = 1.0

    def loss(self, batch, preds=None):
        total_loss, loss_items = super().loss(batch, preds)

        # Retrieve RTDETRDecoder head
        head = self.model[-1]
        if hasattr(head, "weather_logits") and head.weather_logits is not None and "weather_id" in batch:
            logits = head.weather_logits  # (B, 4)
            labels = batch["weather_id"].to(logits.device)  # (B,)

            # Compute CrossEntropy loss for weather condition prediction
            weather_ce = F.cross_entropy(logits, labels)
            weighted_weather_loss = weather_ce * self.weather_loss_weight

            # Add directly to main total_loss to enable backpropagation!
            total_loss = total_loss + weighted_weather_loss

        return total_loss, loss_items


class WeatherAwareRTDETRTrainer(RTDETRTrainer):
    """
    Custom Trainer that uses WeatherAwareRTDETRModel and WeatherRTDETRDataset.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = WeatherAwareRTDETRModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None):
        return WeatherRTDETRDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == "train",
            hyp=self.args,
            rect=False,
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            prefix=colorstr(f"{mode}: "),
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == "train" else 1.0,
        )


def ensure_merged_dataset(project_root: Path) -> Path:
    """Ensure chokam_dataset exists and contains dataset.yaml."""
    dataset_dir = project_root / "acdc_output" / "chokam_dataset"
    yaml_path = dataset_dir / "dataset.yaml"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    yaml_content = f"""path: {dataset_dir.resolve()}
train: images/train
val: images/val
test: images/test
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

    print(f"Dataset YAML configured at: {yaml_path}")
    return yaml_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train RT-DETR with Multi-Scale SCE Weather Detector")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weights", default="rtdetr-l.pt")
    parser.add_argument("--project", default="RTDETR_SCE_Weather")
    parser.add_argument("--name", default="train_merged_sce")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("Starting Weather-Aware RT-DETR (SCE Stage-1) Training")
    print("PyTorch version:", torch.__version__)
    print("CUDA available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Device     :", torch.cuda.get_device_name(0))
    print("=" * 60)

    project_root = Path(__file__).resolve().parent
    yaml_path = ensure_merged_dataset(project_root)

    if args.device != "" and args.device != "cpu" and not torch.cuda.is_available():
        print(f"[Warning] CUDA is not available on this PyTorch build ({torch.__version__}). Falling back from device='{args.device}' to device='cpu'.")
        device = "cpu"
    else:
        device = args.device if args.device != "" else (0 if torch.cuda.is_available() else "cpu")

    trainer_overrides = {
        "model": args.weights,
        "data": str(yaml_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch_size,
        "device": device,
        "workers": 4,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        # Force heavy data augmentation to prevent weather memorization
        "augment": True,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "fliplr": 0.5,
        "scale": 0.5,
        "erasing": 0.4,
    }

    print(f"Launching WeatherAwareRTDETRTrainer on device {device}...")
    trainer = WeatherAwareRTDETRTrainer(overrides=trainer_overrides)
    trainer.train()


if __name__ == "__main__":
    main()
