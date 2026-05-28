import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import datetime
import platform

# =========================================================
# 1. KONFIGURASI
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
base_dir = BASE_DIR.parent.parent.parent / "TomatoBinaryDataset"

IMAGE_SIZE = (300, 300)
BATCH_SIZE = 32
SEED = 123
VALIDATION_SPLIT = 0.2

# Dibuat besar supaya model tidak terlalu confident
LABEL_SMOOTHING = 0.20

BASE_MODEL_NAME = "MobileNetV3Large"
TRANSFER_LEARNING_WEIGHTS = "ImageNet"

BEST_MODEL_PATH = BASE_DIR / "best_model_binary_mobilenetv3.keras"
FINAL_MODEL_PATH = BASE_DIR / "mobilenet_v3_binary_final.keras"

REPORT_TXT_PATH = BASE_DIR / "training_report_journal_friendly.txt"

CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"
TRAINING_CURVE_PATH = BASE_DIR / "training_curve.png"

tf.keras.utils.set_random_seed(SEED)

# =========================================================
# 2. LOAD PATH DATASET
# =========================================================

class_names = ["diseased", "healthy"]

class_to_label = {
    "diseased": 0,
    "healthy": 1
}

valid_ext = [".jpg", ".jpeg", ".png"]

image_paths = []
labels = []

for class_name in class_names:

    class_dir = base_dir / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {class_dir}"
        )

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
# 4. LOAD IMAGE
# =========================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(
        image,
        IMAGE_SIZE
    )

    image = tf.cast(
        image,
        tf.float32
    )

    label = tf.cast(
        label,
        tf.float32
    )

    return image, label


AUTOTUNE = tf.data.AUTOTUNE

# =========================================================
# 5. DATASET PIPELINE
# =========================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

train_ds = train_ds.shuffle(
    buffer_size=len(train_paths),
    seed=SEED
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.batch(BATCH_SIZE)

train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_paths, val_labels)
)

val_ds = val_ds.map(
    load_image,
    num_parallel_calls=AUTOTUNE
)

val_ds = val_ds.batch(BATCH_SIZE)

val_ds = val_ds.prefetch(AUTOTUNE)

# =========================================================
# 6. AUGMENTASI
# =========================================================

data_augmentation = tf.keras.Sequential([

    layers.RandomFlip("horizontal"),

    layers.RandomRotation(0.05),

    layers.RandomZoom(0.05),

    layers.RandomContrast(0.05),

], name="data_augmentation")

# =========================================================
# 7. VISUALISASI AUGMENTASI
# =========================================================

def visualize_augmentation(ds, augmentation_layer):

    plt.figure(figsize=(10, 10))

    for images, labels_batch in ds.take(1):

        for i in range(3):

            label_name = class_names[
                int(labels_batch[i].numpy())
            ]

            # ORIGINAL
            plt.subplot(3, 2, 2 * i + 1)

            plt.imshow(
                images[i].numpy().astype("uint8")
            )

            plt.title(
                f"Original: {label_name}"
            )

            plt.axis("off")

            # AUGMENTED
            augmented_image = augmentation_layer(
                tf.expand_dims(images[i], 0),
                training=True
            )

            plt.subplot(3, 2, 2 * i + 2)

            plt.imshow(
                augmented_image[0].numpy().astype("uint8")
            )

            plt.title("After Augmentation")

            plt.axis("off")

    plt.suptitle(
        "Sample Data Augmentation Before vs After"
    )

    plt.tight_layout()

    plt.show(block=False)

    plt.pause(0.1)


visualize_augmentation(
    train_ds,
    data_augmentation
)

# =========================================================
# 8. MODEL
# =========================================================

base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze total backbone
base_model.trainable = False

inputs = layers.Input(
    shape=(*IMAGE_SIZE, 3)
)

x = data_augmentation(inputs)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

# =========================================================
# CLASSIFIER HEAD DIPERKECIL
# =========================================================

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(
    64,
    activation="relu"
)(x)

x = layers.Dropout(0.4)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = models.Model(
    inputs,
    outputs
)

model.summary()

# =========================================================
# 9. CALLBACKS
# =========================================================

def get_callbacks(patience_es):

    return [

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1
        )
    ]

# =========================================================
# 10. TRAINING
# =========================================================

print("\n--- Training Classifier Head ---")

LEARNING_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5

model.compile(

    optimizer=tf.keras.optimizers.SGD(
        learning_rate=LEARNING_RATE,
        momentum=0.8
    ),

    loss=tf.keras.losses.BinaryCrossentropy(
        label_smoothing=LABEL_SMOOTHING
    ),

    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=get_callbacks(PATIENCE)
)

# =========================================================
# 11. EVALUASI MODEL
# =========================================================

def evaluate_model(model, val_ds):

    y_true = []
    y_pred = []
    y_prob = []

    for images, labels_batch in val_ds:

        probs = model.predict(
            images,
            verbose=0
        )

        preds = (
            probs >= 0.5
        ).astype(int).reshape(-1)

        y_true.extend(
            labels_batch.numpy().astype(int)
        )

        y_pred.extend(preds)

        y_prob.extend(
            probs.reshape(-1)
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True
    )

    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )

    return (
        y_true,
        y_pred,
        y_prob,
        cm,
        report_dict,
        report_text
    )


(
    y_true,
    y_pred,
    y_prob,
    cm,
    report_dict,
    report_text
) = evaluate_model(model, val_ds)

# =========================================================
# 12. PLOT TRAINING CURVE
# =========================================================

def save_training_curve(
    history,
    save_path
):

    acc = history.history["accuracy"]

    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]

    val_loss = history.history["val_loss"]

    plt.figure(figsize=(12, 5))

    # ACCURACY
    plt.subplot(1, 2, 1)

    plt.plot(
        acc,
        label="Training Accuracy"
    )

    plt.plot(
        val_acc,
        label="Validation Accuracy"
    )

    plt.title(
        "Training and Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    # LOSS
    plt.subplot(1, 2, 2)

    plt.plot(
        loss,
        label="Training Loss"
    )

    plt.plot(
        val_loss,
        label="Validation Loss"
    )

    plt.title(
        "Training and Validation Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show(block=False)

    plt.pause(0.1)


save_training_curve(
    history,
    TRAINING_CURVE_PATH
)

# =========================================================
# 13. CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(
    cm,
    class_names,
    save_path
):

    plt.figure(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show(block=False)

    plt.pause(0.1)


save_confusion_matrix(
    cm,
    class_names,
    CONFUSION_MATRIX_PATH
)

print("\nClassification Report:")
print(report_text)

# =========================================================
# 14. SAVE MODEL
# =========================================================

model.save(
    str(FINAL_MODEL_PATH)
)

# =========================================================
# 15. EXPORT TXT REPORT
# =========================================================

def export_journal_friendly_report():

    now = datetime.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    total_params = model.count_params()

    trainable_params = int(np.sum([
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    ]))

    non_trainable_params = int(np.sum([
        tf.keras.backend.count_params(w)
        for w in model.non_trainable_weights
    ]))

    final_train_acc = (
        history.history["accuracy"][-1]
    )

    final_train_loss = (
        history.history["loss"][-1]
    )

    final_val_acc = report_dict["accuracy"]

    best_val_acc = max(
        history.history["val_accuracy"]
    )

    tn, fp, fn, tp = cm.ravel()

    with open(
        REPORT_TXT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "JOURNAL-FRIENDLY TRAINING REPORT\n"
        )

        f.write(
            "================================\n\n"
        )

        # =====================================================
        # EXPERIMENT INFO
        # =====================================================

        f.write("1. Experiment Information\n")
        f.write("-------------------------\n")

        f.write(
            f"Report generated at       : {now}\n"
        )

        f.write(
            f"Python version            : "
            f"{platform.python_version()}\n"
        )

        f.write(
            f"TensorFlow version        : "
            f"{tf.__version__}\n"
        )

        f.write(
            f"Random seed               : "
            f"{SEED}\n\n"
        )

        # =====================================================
        # DATASET
        # =====================================================

        f.write("2. Dataset Description\n")
        f.write("----------------------\n")

        f.write(
            "Dataset name              : "
            "Tomato Binary Dataset\n"
        )

        f.write(
            "Dataset task              : "
            "Binary image classification\n"
        )

        f.write(
            "Classes                   : "
            "diseased, healthy\n"
        )

        f.write(
            f"Dataset directory         : "
            f"{base_dir.resolve()}\n"
        )

        f.write(
            f"Total images              : "
            f"{total_dataset}\n"
        )

        f.write(
            f"Total diseased images     : "
            f"{total_diseased}\n"
        )

        f.write(
            f"Total healthy images      : "
            f"{total_healthy}\n\n"
        )

        # =====================================================
        # SPLIT
        # =====================================================

        f.write("3. Train-Validation Split\n")
        f.write("-------------------------\n")

        f.write(
            "Split method              : "
            "Stratified train-validation split\n"
        )

        f.write(
            f"Validation ratio          : "
            f"{VALIDATION_SPLIT}\n"
        )

        f.write(
            f"Training images           : "
            f"{len(train_labels)}\n"
        )

        f.write(
            f"Validation images         : "
            f"{len(val_labels)}\n"
        )

        f.write(
            f"Validation diseased images: "
            f"{val_diseased}\n"
        )

        f.write(
            f"Validation healthy images : "
            f"{val_healthy}\n\n"
        )

        # =====================================================
        # PREPROCESSING
        # =====================================================

        f.write("4. Image Preprocessing\n")
        f.write("----------------------\n")

        f.write(
            f"Input image size          : "
            f"{IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}\n"
        )

        f.write(
            f"Batch size                : "
            f"{BATCH_SIZE}\n"
        )

        f.write(
            "Color format              : RGB\n"
        )

        f.write(
            "Image resizing            : "
            "TensorFlow resize operation\n\n"
        )

        # =====================================================
        # AUGMENTATION
        # =====================================================

        f.write("5. Data Augmentation\n")
        f.write("--------------------\n")

        f.write(
            "Augmentation applied      : Yes\n"
        )

        f.write(
            "Augmentation techniques   :\n"
        )

        f.write("- Random horizontal flip\n")
        f.write("- Random rotation\n")
        f.write("- Random zoom\n")
        f.write("- Random contrast\n\n")

        # =====================================================
        # MODEL
        # =====================================================

        f.write("6. Model Architecture\n")
        f.write("---------------------\n")

        f.write(
            f"Base model                : "
            f"{BASE_MODEL_NAME}\n"
        )

        f.write(
            f"Pretrained weights        : "
            f"{TRANSFER_LEARNING_WEIGHTS}\n"
        )

        f.write(
            "Classifier head           : "
            "Dense(128, ReLU) -> "
            "Dropout(0.5) -> "
            "Dense(64, ReLU) -> "
            "Dropout(0.4) -> "
            "Dense(1, Sigmoid)\n"
        )

        f.write(
            "Backbone status           : Frozen\n"
        )

        f.write(
            "Fine-tuning               : Disabled\n"
        )

        f.write(
            f"Total parameters          : "
            f"{total_params}\n"
        )

        f.write(
            f"Trainable parameters      : "
            f"{trainable_params}\n"
        )

        f.write(
            f"Non-trainable parameters  : "
            f"{non_trainable_params}\n\n"
        )

        # =====================================================
        # FLOW DIAGRAM
        # =====================================================

        f.write("6A. Model Flow Diagram\n")
        f.write("----------------------\n\n")

        f.write(
            "Input Image "
            "(300x300x3 RGB)\n"
        )

        f.write("        |\n")
        f.write("        v\n")

        f.write("Data Augmentation\n")

        f.write("        |\n")
        f.write("        v\n")

        f.write("MobileNetV3Large Backbone\n")

        f.write("        |\n")
        f.write("        v\n")

        f.write("GlobalAveragePooling2D\n")

        f.write("        |\n")
        f.write("        v\n")

        f.write("Dense(128, ReLU)\n")

        f.write("        |\n")
        f.write("        v\n")

        f.write("Dropout(0.5)\n")

        f.write("        |\n")
        f.write("        v\n")

        f.write("Dense(64, ReLU)\n")

        f.write("        |\n")
        f.write("        v\n")

        f.write("Dropout(0.4)\n")

        f.write("        |\n")
        f.write("        v\n")

        f.write("Dense(1, Sigmoid)\n")

        f.write("        |\n")
        f.write("        v\n")

        f.write("Binary Prediction\n\n")

        # =====================================================
        # TRAINING CONFIG
        # =====================================================

        f.write("7. Training Configuration\n")
        f.write("-------------------------\n")

        f.write(
            "Loss function             : "
            "Binary Crossentropy\n"
        )

        f.write(
            f"Label smoothing           : "
            f"{LABEL_SMOOTHING}\n"
        )

        f.write(
            "Optimizer                 : SGD\n"
        )

        f.write(
            f"Learning rate             : "
            f"{LEARNING_RATE}\n"
        )

        f.write(
            f"Epochs                    : "
            f"{EPOCHS}\n\n"
        )

        # =====================================================
        # EVALUATION
        # =====================================================

        f.write("8. Evaluation Metrics\n")
        f.write("---------------------\n")

        f.write(
            "Metrics used              : "
            "Accuracy, Precision, Recall, "
            "AUC, F1-score\n\n"
        )

        # =====================================================
        # CLASSIFICATION REPORT
        # =====================================================

        f.write("9. Final Classification Report\n")
        f.write("------------------------------\n")

        f.write(report_text)

        f.write("\n\n")

        # =====================================================
        # CONFUSION MATRIX
        # =====================================================

        f.write("10. Confusion Matrix\n")
        f.write("--------------------\n")

        f.write(str(cm))

        f.write("\n\n")

        f.write(
            f"True Diseased predicted Diseased : "
            f"{tn}\n"
        )

        f.write(
            f"True Diseased predicted Healthy  : "
            f"{fp}\n"
        )

        f.write(
            f"True Healthy predicted Diseased  : "
            f"{fn}\n"
        )

        f.write(
            f"True Healthy predicted Healthy   : "
            f"{tp}\n\n"
        )

        # =====================================================
        # FINAL SUMMARY
        # =====================================================

        f.write("11. Final Training Summary\n")
        f.write("--------------------------\n")

        f.write(
            f"Final training accuracy   : "
            f"{final_train_acc:.4f}\n"
        )

        f.write(
            f"Final training loss       : "
            f"{final_train_loss:.4f}\n"
        )

        f.write(
            f"Final validation accuracy : "
            f"{final_val_acc:.4f}\n"
        )

        f.write(
            f"Best validation accuracy  : "
            f"{best_val_acc:.4f}\n\n"
        )

        # =====================================================
        # SAVED OUTPUTS
        # =====================================================

        f.write("12. Saved Outputs\n")
        f.write("-----------------\n")

        f.write(
            f"Best model path           : "
            f"{BEST_MODEL_PATH.resolve()}\n"
        )

        f.write(
            f"Final model path          : "
            f"{FINAL_MODEL_PATH.resolve()}\n"
        )

        f.write(
            f"Training curve image      : "
            f"{TRAINING_CURVE_PATH.resolve()}\n"
        )

        f.write(
            f"Confusion matrix image    : "
            f"{CONFUSION_MATRIX_PATH.resolve()}\n"
        )

        f.write(
            f"Text report path          : "
            f"{REPORT_TXT_PATH.resolve()}\n"
        )


export_journal_friendly_report()

print("\nTraining selesai.")
print(f"Best model saved to          : {BEST_MODEL_PATH}")
print(f"Final model saved to         : {FINAL_MODEL_PATH}")
print(f"Training curve saved to      : {TRAINING_CURVE_PATH}")
print(f"Confusion matrix saved to    : {CONFUSION_MATRIX_PATH}")
print(f"Journal-friendly TXT saved to: {REPORT_TXT_PATH}")

plt.show()