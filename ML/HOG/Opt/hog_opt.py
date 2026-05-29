import os
import cv2
import datetime
import platform
import joblib
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from skimage.feature import hog


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
base_dir = BASE_DIR.parent.parent.parent / "TomatoBinaryDataset"

IMAGE_SIZE = (300, 300)
SEED = 123
VALIDATION_SPLIT = 0.2

MODEL_NAME = "Optimized HOG + SVM"

class_names = ["diseased", "healthy"]

class_to_label = {
    "diseased": 0,
    "healthy": 1
}

valid_ext = [".jpg", ".jpeg", ".png"]

OUTPUT_DIR = BASE_DIR / "hog_svm_opt_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "hog_svm_opt_model.joblib"
REPORT_TXT_PATH = OUTPUT_DIR / "hog_svm_opt_report.txt"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "hog_svm_opt_confusion_matrix.png"

np.random.seed(SEED)


# =========================================================
# 2. LOAD IMAGE PATHS
# =========================================================

image_paths = []
labels = []

for class_name in class_names:
    class_dir = base_dir / class_name

    if not class_dir.exists():
        raise FileNotFoundError(f"Folder tidak ditemukan: {class_dir}")

    for img_path in class_dir.iterdir():
        if img_path.suffix.lower() in valid_ext:
            image_paths.append(str(img_path))
            labels.append(class_to_label[class_name])

image_paths = np.array(image_paths)
labels = np.array(labels)

total_diseased = int(np.sum(labels == 0))
total_healthy = int(np.sum(labels == 1))
total_dataset = len(labels)

print("\nTotal Dataset:")
print(f"diseased : {total_diseased}")
print(f"healthy  : {total_healthy}")
print(f"total    : {total_dataset}")


# =========================================================
# 3. STRATIFIED SPLIT
# =========================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=VALIDATION_SPLIT,
    random_state=SEED,
    stratify=labels
)

train_diseased = int(np.sum(train_labels == 0))
train_healthy = int(np.sum(train_labels == 1))

val_diseased = int(np.sum(val_labels == 0))
val_healthy = int(np.sum(val_labels == 1))

print("\nTraining Dataset:")
print(f"diseased : {train_diseased}")
print(f"healthy  : {train_healthy}")
print(f"total    : {len(train_labels)}")

print("\nValidation Dataset:")
print(f"diseased : {val_diseased}")
print(f"healthy  : {val_healthy}")
print(f"total    : {len(val_labels)}")


# =========================================================
# 4. IMAGE PREPROCESSING
# =========================================================

def load_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise ValueError(f"Gagal membaca gambar: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        IMAGE_SIZE
    )

    return image


# =========================================================
# 5. DATA AUGMENTATION
# Same concept as deep learning:
# one image remains one training sample
# =========================================================

def augment_image(image):
    augmented = image.copy()

    # Random horizontal flip
    if np.random.rand() < 0.5:
        augmented = cv2.flip(augmented, 1)

    # Random rotation, similar to RandomRotation(0.05)
    if np.random.rand() < 0.5:
        angle = np.random.uniform(-18, 18)

        h, w = augmented.shape[:2]
        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        augmented = cv2.warpAffine(
            augmented,
            matrix,
            (w, h),
            borderMode=cv2.BORDER_REFLECT
        )

    # Random zoom, similar to RandomZoom(0.05)
    if np.random.rand() < 0.5:
        zoom_factor = np.random.uniform(1.0, 1.05)

        h, w = augmented.shape[:2]

        new_h = int(h / zoom_factor)
        new_w = int(w / zoom_factor)

        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2

        cropped = augmented[
            start_y:start_y + new_h,
            start_x:start_x + new_w
        ]

        augmented = cv2.resize(
            cropped,
            (w, h)
        )

    # Random contrast, similar to RandomContrast(0.05)
    if np.random.rand() < 0.5:
        alpha = np.random.uniform(0.95, 1.05)

        augmented = cv2.convertScaleAbs(
            augmented,
            alpha=alpha,
            beta=0
        )

    return augmented


# =========================================================
# 6. VISUALIZE AUGMENTATION
# =========================================================

def visualize_augmentation(paths, labels):
    plt.figure(figsize=(10, 10))

    sample_indices = np.random.choice(
        len(paths),
        size=3,
        replace=False
    )

    for row_idx, sample_idx in enumerate(sample_indices):
        image = load_image(paths[sample_idx])
        label_name = class_names[int(labels[sample_idx])]

        augmented = augment_image(image)

        plt.subplot(3, 2, 2 * row_idx + 1)
        plt.imshow(image)
        plt.title(f"Original: {label_name}")
        plt.axis("off")

        plt.subplot(3, 2, 2 * row_idx + 2)
        plt.imshow(augmented)
        plt.title("After Augmentation")
        plt.axis("off")

    plt.suptitle("Sample Data Augmentation Before vs After (Opt)")
    plt.tight_layout()

    augmentation_path = OUTPUT_DIR / "hog_svm_augmentation_sample.png"

    plt.savefig(
        augmentation_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show(block=False)
    plt.pause(0.1)

    return augmentation_path


augmentation_path = visualize_augmentation(
    train_paths,
    train_labels
)


# =========================================================
# 7. HOG FEATURE EXTRACTION
# =========================================================

def extract_hog_features(image):
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    return features


def build_feature_dataset(paths, labels, augment=False):
    X = []
    y = []

    total = len(paths)

    for idx, (path, label) in enumerate(zip(paths, labels), start=1):
        image = load_image(path)

        if augment:
            image = augment_image(image)

        features = extract_hog_features(image)

        X.append(features)
        y.append(label)

        if idx % 100 == 0:
            print(f"Processed {idx}/{total} images")

    return np.array(X), np.array(y)


# =========================================================
# 8. BUILD FEATURE DATASET
# =========================================================

print("\nExtracting training features with augmentation...")
X_train, y_train = build_feature_dataset(
    train_paths,
    train_labels,
    augment=True
)

print("\nExtracting validation features without augmentation...")
X_val, y_val = build_feature_dataset(
    val_paths,
    val_labels,
    augment=False
)

print("\nFeature Dataset:")
print(f"Train feature shape: {X_train.shape}")
print(f"Val feature shape  : {X_val.shape}")


# =========================================================
# 9. MODEL OPTIMIZATION (GridSearchCV)
# =========================================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        probability=True,
        random_state=SEED,
        verbose=False
    ))
])

param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': ['scale', 'auto', 0.1, 0.01],
    'classifier__kernel': ['rbf', 'poly']
}

print("\nStarting Hyperparameter Optimization with GridSearchCV...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nOptimization completed.")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

model = grid_search.best_estimator_


# =========================================================
# 10. MODEL EVALUATION
# =========================================================

print("\nEvaluating best model on validation set...")

y_pred = model.predict(X_val)

accuracy = accuracy_score(
    y_val,
    y_pred
)

cm = confusion_matrix(
    y_val,
    y_pred
)

report_text = classification_report(
    y_val,
    y_pred,
    target_names=class_names
)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report_text)


# =========================================================
# 11. SAVE CONFUSION MATRIX
# =========================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title(f"Confusion Matrix - {MODEL_NAME}")
plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show(block=False)
plt.pause(0.1)


# =========================================================
# 12. SAVE MODEL
# =========================================================

joblib.dump(
    model,
    MODEL_PATH
)


# =========================================================
# 13. EXPORT JOURNAL-FRIENDLY REPORT
# =========================================================

def export_report():
    now = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("JOURNAL-FRIENDLY OPTIMIZED MACHINE LEARNING REPORT\n")
        f.write("==================================================\n\n")

        f.write("1. Experiment Information\n")
        f.write("-------------------------\n")
        f.write(f"Report generated at       : {now}\n")
        f.write(f"Python version            : {platform.python_version()}\n")
        f.write(f"OpenCV version            : {cv2.__version__}\n")
        f.write(f"Random seed               : {SEED}\n\n")

        f.write("2. Dataset Description\n")
        f.write("----------------------\n")
        f.write("Dataset name              : Tomato Binary Dataset\n")
        f.write("Dataset task              : Binary image classification\n")
        f.write("Classes                   : diseased, healthy\n")
        f.write(f"Dataset directory         : {base_dir.resolve()}\n")
        f.write(f"Total images              : {total_dataset}\n")
        f.write(f"Total diseased images     : {total_diseased}\n")
        f.write(f"Total healthy images      : {total_healthy}\n\n")

        f.write("3. Train-Validation Split\n")
        f.write("-------------------------\n")
        f.write("Split method              : Stratified train-validation split\n")
        f.write(f"Validation ratio          : {VALIDATION_SPLIT}\n")
        f.write(f"Training images           : {len(train_labels)}\n")
        f.write(f"Validation images         : {len(val_labels)}\n")
        f.write(f"Training diseased images  : {train_diseased}\n")
        f.write(f"Training healthy images   : {train_healthy}\n")
        f.write(f"Validation diseased images: {val_diseased}\n")
        f.write(f"Validation healthy images : {val_healthy}\n\n")

        f.write("4. Image Preprocessing\n")
        f.write("----------------------\n")
        f.write(f"Input image size          : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}\n")
        f.write("Color format              : RGB\n")
        f.write("Image resizing            : OpenCV resize operation\n\n")

        f.write("5. Data Augmentation\n")
        f.write("--------------------\n")
        f.write("Augmentation applied      : Yes, training set only\n")
        f.write("Augmentation strategy     : One input image produces one augmented image, so the number of training samples remains unchanged.\n")
        f.write("Augmentation techniques   :\n")
        f.write("- Random horizontal flip\n")
        f.write("- Random rotation\n")
        f.write("- Random zoom\n")
        f.write("- Random contrast\n\n")

        f.write("6. Machine Learning Model & Optimization\n")
        f.write("----------------------------------------\n")
        f.write(f"Model                     : {MODEL_NAME}\n")
        f.write("Feature extraction        : Histogram of Oriented Gradients (HOG)\n")
        f.write("Classifier                : Support Vector Machine (SVM)\n")
        f.write("Optimization Method       : GridSearchCV (5-fold CV)\n")
        f.write(f"Parameter Grid            : {param_grid}\n")
        f.write(f"Best Parameters Found     : {grid_search.best_params_}\n")
        f.write(f"Best CV Accuracy          : {grid_search.best_score_:.4f}\n\n")

        f.write("7. Feature Dataset Shape\n")
        f.write("------------------------\n")
        f.write(f"Training feature shape    : {X_train.shape}\n")
        f.write(f"Validation feature shape  : {X_val.shape}\n\n")

        f.write("8. Evaluation Metrics\n")
        f.write("---------------------\n")
        f.write("Metrics used              : Accuracy, Precision, Recall, F1-score, Confusion Matrix\n\n")

        f.write("9. Final Classification Report (Best Model)\n")
        f.write("-------------------------------------------\n")
        f.write(report_text)
        f.write("\n\n")

        f.write("10. Confusion Matrix\n")
        f.write("--------------------\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("11. Final Summary\n")
        f.write("-----------------\n")
        f.write(f"Validation accuracy       : {accuracy:.4f}\n\n")

        f.write("12. Saved Outputs\n")
        f.write("-----------------\n")
        f.write(f"Saved model path          : {MODEL_PATH.resolve()}\n")
        f.write(f"Confusion matrix image    : {CONFUSION_MATRIX_PATH.resolve()}\n")
        f.write(f"Augmentation sample image : {augmentation_path.resolve()}\n")
        f.write(f"Text report path          : {REPORT_TXT_PATH.resolve()}\n")


export_report()


print(f"\n{MODEL_NAME} training completed.")
print(f"Model saved to             : {MODEL_PATH}")
print(f"Confusion matrix saved to  : {CONFUSION_MATRIX_PATH}")
print(f"Report saved to            : {REPORT_TXT_PATH}")
print(f"Output directory           : {OUTPUT_DIR}")

plt.show()