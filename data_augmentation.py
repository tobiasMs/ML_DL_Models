import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# =========================================================
# KONFIGURASI
# =========================================================

# Folder dataset asli
SOURCE_DIR = Path("TomatoDataset")

# Folder output binary dataset
OUTPUT_DIR = Path("TomatoBinaryDataset")

# Nama class output
HEALTHY_CLASS = "healthy"
DISEASED_CLASS = "diseased"

# Target jumlah healthy setelah augmentasi
TARGET_HEALTHY_IMAGES = 3000

# Format file yang didukung
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# Seed random
random.seed(42)

# =========================================================
# BUAT FOLDER OUTPUT
# =========================================================

healthy_output = OUTPUT_DIR / HEALTHY_CLASS
diseased_output = OUTPUT_DIR / DISEASED_CLASS

healthy_output.mkdir(parents=True, exist_ok=True)
diseased_output.mkdir(parents=True, exist_ok=True)

# =========================================================
# AUGMENTATION PIPELINE
# =========================================================

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=25, p=0.7),
    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.7
    ),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=15,
        p=0.7
    ),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=15,
        val_shift_limit=10,
        p=0.5
    ),
])

# =========================================================
# FUNGSI BANTU
# =========================================================

def is_image_file(file_path):
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def save_image(image, save_path):
    cv2.imwrite(str(save_path), image)


# =========================================================
# KUMPULKAN SEMUA GAMBAR
# =========================================================

healthy_images = []
diseased_images = []

print("Scanning dataset...")

for class_dir in SOURCE_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name.lower()

    image_files = [
        x for x in class_dir.iterdir()
        if is_image_file(x)
    ]

    # cek healthy
    if "healthy" in class_name:
        healthy_images.extend(image_files)

    else:
        diseased_images.extend(image_files)

print(f"\nHealthy images  : {len(healthy_images)}")
print(f"Diseased images : {len(diseased_images)}")

# =========================================================
# COPY DATA DISEASED
# =========================================================

print("\nCopying diseased images...")

disease_counter = 0

for img_path in tqdm(diseased_images):

    image = cv2.imread(str(img_path))

    if image is None:
        continue

    save_path = diseased_output / f"diseased_{disease_counter}.jpg"

    save_image(image, save_path)

    disease_counter += 1

# =========================================================
# COPY DATA HEALTHY ASLI
# =========================================================

print("\nCopying original healthy images...")

healthy_counter = 0

healthy_loaded_images = []

for img_path in tqdm(healthy_images):

    image = cv2.imread(str(img_path))

    if image is None:
        continue

    healthy_loaded_images.append(image)

    save_path = healthy_output / f"healthy_{healthy_counter}.jpg"

    save_image(image, save_path)

    healthy_counter += 1

# =========================================================
# AUGMENTASI HEALTHY
# =========================================================

print("\nGenerating augmented healthy images...")

while healthy_counter < TARGET_HEALTHY_IMAGES:

    # pilih random gambar healthy
    image = random.choice(healthy_loaded_images)

    # augmentasi
    augmented = augment(image=image)["image"]

    save_path = healthy_output / f"healthy_aug_{healthy_counter}.jpg"

    save_image(augmented, save_path)

    healthy_counter += 1

print("\nDONE!")
print(f"Final healthy images  : {healthy_counter}")
print(f"Final diseased images : {disease_counter}")

print("\nOutput dataset:")
print(OUTPUT_DIR.resolve())