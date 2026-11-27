import os
import shutil
import random
from pathlib import Path

random.seed(42)

dataset_dir = Path("acdc_output/chokam_dataset")
images_dir = dataset_dir / "images"
labels_dir = dataset_dir / "labels"

all_images = []
for split in ["train", "val", "test", "temp_images"]:
    split_dir = dataset_dir / "images" / split if split != "temp_images" else dataset_dir / "temp_images"
    if split_dir.exists():
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            all_images.extend(list(split_dir.glob(ext)))

if len(all_images) == 0:
    print("No images found! Dataset might be missing or already deleted.")
    exit(1)

temp_images = dataset_dir / "temp_images"
temp_labels = dataset_dir / "temp_labels"
temp_images.mkdir(exist_ok=True)
temp_labels.mkdir(exist_ok=True)

print(f"Gathering {len(all_images)} total images for splitting...")
for img_path in all_images:
    label_path = labels_dir / img_path.parent.name / (img_path.stem + ".txt")
    
    # Try to move them, handle if already in temp somehow
    dest_img = temp_images / img_path.name
    if not dest_img.exists():
        shutil.move(str(img_path), str(dest_img))
        
    dest_lbl = temp_labels / (img_path.stem + ".txt")
    if label_path.exists() and not dest_lbl.exists():
        shutil.move(str(label_path), str(dest_lbl))

for split in ["train", "val", "test"]:
    if (images_dir / split).exists():
        shutil.rmtree(images_dir / split, ignore_errors=True)
    if (labels_dir / split).exists():
        shutil.rmtree(labels_dir / split, ignore_errors=True)

temp_all_images = list(temp_images.glob("*.*"))
random.shuffle(temp_all_images)

# 70% Train, 20% Val, 10% Test
num_total = len(temp_all_images)
num_train = int(num_total * 0.70)
num_val = int(num_total * 0.20)
num_test = num_total - num_train - num_val

train_imgs = temp_all_images[:num_train]
val_imgs = temp_all_images[num_train:num_train+num_val]
test_imgs = temp_all_images[num_train+num_val:]

def move_split(imgs, split_name):
    img_split_dir = images_dir / split_name
    lbl_split_dir = labels_dir / split_name
    img_split_dir.mkdir(parents=True, exist_ok=True)
    lbl_split_dir.mkdir(parents=True, exist_ok=True)
    
    for img in imgs:
        shutil.move(str(img), str(img_split_dir / img.name))
        lbl_path = temp_labels / (img.stem + ".txt")
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(lbl_split_dir / lbl_path.name))

move_split(train_imgs, "train")
move_split(val_imgs, "val")
move_split(test_imgs, "test")

# Cleanup temp dirs
shutil.rmtree(temp_images)
shutil.rmtree(temp_labels)

print(f"Successfully split dataset: {len(train_imgs)} Train, {len(val_imgs)} Val, {len(test_imgs)} Test")
