import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm
import albumentations as A

# =========================================================
# KONFIGURASI
# =========================================================

# Dataset asli (multiclass)
SOURCE_DIR = Path("TomatoDataset")

# Output dataset binary
OUTPUT_DIR = Path("TomatoBinaryDataset")

# Nama class output
HEALTHY_CLASS = "healthy"
DISEASED_CLASS = "diseased"

# Target jumlah final tiap class
TARGET_HEALTHY_IMAGES = 3000
TARGET_DISEASED_IMAGES = 3000

# Format gambar yang didukung
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# Seed
random.seed(42)

# =========================================================
# RESET OUTPUT FOLDER
# =========================================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

healthy_output = OUTPUT_DIR / HEALTHY_CLASS
diseased_output = OUTPUT_DIR / DISEASED_CLASS

healthy_output.mkdir(parents=True, exist_ok=True)
diseased_output.mkdir(parents=True, exist_ok=True)

# =========================================================
# AUGMENTATION PIPELINE
# =========================================================

augment = A.Compose([

    A.HorizontalFlip(p=0.5),

    A.Rotate(
        limit=20,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.7
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.5
    ),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.10,
        rotate_limit=10,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.5
    ),

    A.GaussianBlur(
        blur_limit=(3, 5),
        p=0.15
    ),

    A.HueSaturationValue(
        hue_shift_limit=8,
        sat_shift_limit=10,
        val_shift_limit=8,
        p=0.3
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
# SCAN DATASET
# =========================================================

healthy_images = []
diseased_images = []

print("\nScanning dataset...")

for class_dir in SOURCE_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name.lower()

    image_files = [
        x for x in class_dir.iterdir()
        if is_image_file(x)
    ]

    if "healthy" in class_name:
        healthy_images.extend(image_files)
    else:
        diseased_images.extend(image_files)

print(f"\nOriginal Healthy Images  : {len(healthy_images)}")
print(f"Original Diseased Images : {len(diseased_images)}")

# =========================================================
# COPY ORIGINAL HEALTHY
# =========================================================

print("\nCopying original healthy images...")

healthy_loaded_images = []
healthy_counter = 0

for img_path in tqdm(healthy_images):

    image = cv2.imread(str(img_path))

    if image is None:
        continue

    healthy_loaded_images.append(image)

    save_path = healthy_output / f"healthy_{healthy_counter}.jpg"

    save_image(image, save_path)

    healthy_counter += 1

# =========================================================
# AUGMENT HEALTHY
# =========================================================

print("\nGenerating augmented healthy images...")

while healthy_counter < TARGET_HEALTHY_IMAGES:

    image = random.choice(healthy_loaded_images)

    augmented = augment(image=image)["image"]

    save_path = healthy_output / f"healthy_aug_{healthy_counter}.jpg"

    save_image(augmented, save_path)

    healthy_counter += 1

# =========================================================
# COPY ORIGINAL DISEASED
# =========================================================

print("\nCopying original diseased images...")

diseased_loaded_images = []
disease_counter = 0

for img_path in tqdm(diseased_images):

    image = cv2.imread(str(img_path))

    if image is None:
        continue

    diseased_loaded_images.append(image)

    save_path = diseased_output / f"diseased_{disease_counter}.jpg"

    save_image(image, save_path)

    disease_counter += 1

# =========================================================
# AUGMENT DISEASED
# =========================================================

print("\nGenerating augmented diseased images...")

while disease_counter < TARGET_DISEASED_IMAGES:

    image = random.choice(diseased_loaded_images)

    augmented = augment(image=image)["image"]

    save_path = diseased_output / f"diseased_aug_{disease_counter}.jpg"

    save_image(augmented, save_path)

    disease_counter += 1

# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n========================================")
print("DATASET GENERATION FINISHED")
print("========================================")

print(f"\nFinal Healthy Images  : {healthy_counter}")
print(f"Final Diseased Images : {disease_counter}")

print(f"\nDataset saved to:")
print(OUTPUT_DIR.resolve())