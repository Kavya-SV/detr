from pathlib import Path
import shutil
import csv

# -----------------------------
# CONFIGURATION
# -----------------------------
SOURCE = Path("acdc_output")

DEST = SOURCE / "chokam_dataset"

WEATHERS = ["rain", "fog", "snow", "night"]

WEATHER_ID = {
    "rain": 0,
    "fog": 1,
    "snow": 2,
    "night": 3,
}

# -----------------------------
# Create folders
# -----------------------------
for split in ["train", "val"]:
    (DEST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DEST / "labels" / split).mkdir(parents=True, exist_ok=True)

csv_rows = []

duplicates = 0
copied = 0

# -----------------------------
# Merge Dataset
# -----------------------------
for weather in WEATHERS:

    for split in ["train", "val"]:

        img_dir = SOURCE / weather / "images" / split
        lbl_dir = SOURCE / weather / "labels" / split

        for img_path in img_dir.iterdir():

            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            label_path = lbl_dir / (img_path.stem + ".txt")

            new_name = f"{weather}_{img_path.name}"

            dest_img = DEST / "images" / split / new_name

            dest_lbl = DEST / "labels" / split / f"{weather}_{label_path.name}"

            if dest_img.exists():
                duplicates += 1
                print(f"Duplicate skipped: {img_path.name}")
                continue

            shutil.copy2(img_path, dest_img)

            if label_path.exists():
                shutil.copy2(label_path, dest_lbl)

            csv_rows.append([
                new_name,
                weather,
                WEATHER_ID[weather]
            ])
            copied += 1

# -----------------------------
# Save CSV
# -----------------------------
csv_path = DEST / "weather_labels.csv"

with open(csv_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "image",
        "weather",
        "weather_id"
    ])

    writer.writerows(csv_rows)

print("=" * 50)
print("Finished!")
print(f"Copied Images : {copied}")
print(f"Duplicates    : {duplicates}")
print(f"CSV Saved At  : {csv_path}")
print("=" * 50)