"""
Weather-Aware RT-DETR Dataset
c:/Users/kavya_05/OneDrive/Desktop/detr/weather_dataset.py

Extends Ultralytics RTDETRDataset to inject weather_id into sample dictionaries
during batch loading for multi-weather scene condition training.
"""

from pathlib import Path
import os
import pandas as pd

from ultralytics.models.rtdetr.val import RTDETRDataset


WEATHER_PREFIX_MAP = {
    "rain": 0,
    "fog": 1,
    "snow": 2,
    "night": 3,
}


class WeatherRTDETRDataset(RTDETRDataset):
    """
    Dataset class that attaches weather_id labels to each sample.
    """

    def __init__(self, *args, data=None, **kwargs):
        super().__init__(*args, data=data, **kwargs)

        self.weather_dict = {}
        # Try to find weather_labels.csv from dataset root or parent directories
        img_dir = Path(self.img_path)
        possible_csv_paths = [
            img_dir.parent / "weather_labels.csv",
            img_dir.parent.parent / "weather_labels.csv",
            Path("acdc_output/chokam_dataset/weather_labels.csv").resolve(),
        ]

        csv_found = False
        for csv_path in possible_csv_paths:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    self.weather_dict = dict(zip(df["image"], df["weather_id"]))
                    csv_found = True
                    print(f"[WeatherRTDETRDataset] Loaded {len(self.weather_dict)} labels from {csv_path}")
                    break
                except Exception as e:
                    print(f"[WeatherRTDETRDataset] Warning reading {csv_path}: {e}")

        if not csv_found:
            print("[WeatherRTDETRDataset] weather_labels.csv not found; falling back to filename prefixes.")

    def get_image_and_label(self, index: int) -> dict:
        """Fetch image and label dict, injecting weather_id."""
        label = super().get_image_and_label(index)

        img_file = os.path.basename(self.im_files[index])

        # 1. Lookup in CSV dictionary
        if img_file in self.weather_dict:
            weather_id = self.weather_dict[img_file]
        else:
            # 2. Fallback: Parse prefix
            weather_id = 0  # default rain
            for prefix, wid in WEATHER_PREFIX_MAP.items():
                if img_file.lower().startswith(prefix):
                    weather_id = wid
                    break

        label["weather_id"] = weather_id
        return label
