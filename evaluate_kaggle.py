import kagglehub
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys

from ultralytics import RTDETR
import ultralytics.nn.modules.sce as sce

print("Imported SCE from:", sce.__file__)

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
        self.weather_loss_weight = 1.0

    def loss(self, batch, preds=None):
        pass

def main():
    print("Downloading/Locating dataset from Kaggle...")
    import time
    for attempt in range(5):
        try:
            # Using the BDD100K Weather Classification subset
            dataset_path = kagglehub.dataset_download("marquis03/bdd100k-weather-classification")
            break
        except Exception as e:
            print(f"Download failed on attempt {attempt+1}: {e}")
            time.sleep(5)
    else:
        print("Failed to download dataset after 5 attempts.")
        return
    
    base_dir = Path(dataset_path)
    print(f"Dataset located at: {base_dir}")

    # Map kaggle classes to our model's classes (rain=0, fog=1, snow=2, night=3)
    class_map = {}
    for folder in base_dir.rglob('*'):
        if not folder.is_dir(): continue
        
        # Only consider folders that actually contain images
        images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
        if len(images) == 0: continue
        
        name = folder.name.lower()
        # Explicitly avoid 'rainbow' or any other distracting classes
        if "rain" in name and "rainbow" not in name:
            class_map[folder] = 0
        elif "fog" in name or "smog" in name:
            class_map[folder] = 1
        elif "snow" in name:
            class_map[folder] = 2
        elif "night" in name:
            class_map[folder] = 3
            
    print("\nClass mapping found:")
    for k, v in class_map.items():
        print(f"  {k.name} -> Class {v}")
        
    if not class_map:
        print("Could not find rain, fog, or snow folders!")
        return

    print("\nLoading model...")
    model = RTDETR("runs/detect/RTDETR_SCE_Weather/train_sce_v3/weights/best.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    model.model.eval()
    
    transform = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor()
    ])

    all_preds = []
    all_labels = []

    print("\nEvaluating...")
    with torch.no_grad():
        for folder, label_idx in class_map.items():
            images = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
            print(f"Processing {folder.name} ({len(images)} images)...")
            
            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    
                    _ = model.model(img_tensor)
                    weather_logits = model.model.model[-1].weather_logits
                    pred = weather_logits.argmax(dim=1).item()
                    
                    all_preds.append(pred)
                    all_labels.append(label_idx)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    print("\n==============================")
    print("KAGGLE DATASET EVALUATION")
    print("==============================")
    print(f"Accuracy : {accuracy_score(all_labels, all_preds):.4f}")
    
    print("\nClassification Report\n")
    print(classification_report(all_labels, all_preds, zero_division=0))
    
    print("\nConfusion Matrix\n")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()
