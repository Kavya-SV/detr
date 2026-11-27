from ultralytics import RTDETR
import torch
import yaml
from pathlib import Path
import ultralytics.nn.modules.sce as sce
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from ultralytics.cfg import get_cfg
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.nn.tasks import RTDETRDetectionModel
import torch.nn.functional as F

class WeatherAwareRTDETRModel(RTDETRDetectionModel):
    def __init__(self, cfg="rtdetr-l.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        if nc is not None:
            self.nc = nc
        elif hasattr(self, "yaml") and isinstance(self.yaml, dict) and "nc" in self.yaml:
            self.nc = self.yaml["nc"]
        else:
            self.nc = 80
        self.weather_loss_weight = 0.2

    def loss(self, batch, preds=None):
        total_loss, loss_items = super().loss(batch, preds)
        head = self.model[-1]
        if hasattr(head, "weather_logits") and head.weather_logits is not None and "weather_id" in batch:
            logits = head.weather_logits
            labels = batch["weather_id"].to(logits.device)
            weather_ce = F.cross_entropy(logits, labels)
            weighted_weather_loss = weather_ce * self.weather_loss_weight
            total_loss = total_loss + weighted_weather_loss
        return total_loss, loss_items

print("Loading model...")

print("Imported SCE from:", sce.__file__)
model = RTDETR(
    "runs/detect/RTDETR_SCE_Weather/train_sce_v3/weights/best.pt"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.model.to(device)
model.model.eval()

print("✓ Model Loaded")
print("Device:", device)

# -------------------------------------------------------
# Load dataset yaml
# -------------------------------------------------------

with open("acdc_output/chokam_dataset/dataset.yaml", "r") as f:
    data = yaml.safe_load(f)

base = Path(data["path"])

data["train"] = str(base / data["train"])
data["val"] = str(base / data["val"])

print("Train:", data["train"])
print("Val  :", data["val"])

# -------------------------------------------------------
# Validator
# -------------------------------------------------------

args = get_cfg()

args.data = "acdc_output/chokam_dataset/dataset.yaml"
args.imgsz = 640
args.batch = 4
args.workers = 0
args.cache = False
args.compile = False

validator = RTDETRValidator(args=args)
validator.data = data

loader = validator.get_dataloader(
    data["val"],
    batch_size=4,
)

print("\nStarting Weather Evaluation...\n")

all_preds = []
all_labels = []

with torch.no_grad():

    for batch in loader:

        imgs = batch["img"].to(device).float() / 255.0

        labels = batch["weather_id"]

        # Forward pass
        _ = model.model(imgs)

        # Read logits directly from head
        weather_logits = model.model.model[-1].weather_logits

        preds = weather_logits.argmax(dim=1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

print("\n==============================")
print("WEATHER EVALUATION RESULTS")
print("==============================")

print(f"Accuracy : {accuracy_score(all_labels, all_preds):.4f}")
print(f"Precision: {precision_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
print(f"Recall   : {recall_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")
print(f"F1 Score : {f1_score(all_labels, all_preds, average='macro', zero_division=0):.4f}")

print("\nClassification Report\n")
print(classification_report(all_labels, all_preds, zero_division=0))

print("\nConfusion Matrix\n")
print(confusion_matrix(all_labels, all_preds))