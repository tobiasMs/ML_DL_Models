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

LABEL_SMOOTHING = 0.05

BASE_MODEL_NAME = "ResNet50"
TRANSFER_LEARNING_WEIGHTS = "ImageNet"

BEST_MODEL_PATH = BASE_DIR / "best_model_binary_resnet50.keras"
FINAL_MODEL_PATH = BASE_DIR / "resnet50_binary_final.keras"
REPORT_TXT_PATH = BASE_DIR / "training_report_resnet50_journal_friendly.txt"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix_resnet50.png"
TRAINING_CURVE_PATH = BASE_DIR / "training_curve_resnet50.png"

tf.keras.utils.set_random_seed(SEED)

# =========================================================
# 2. LOAD PATH GAMBAR DAN LABEL
# =========================================================

# PENTING:
# Urutan ini harus tetap sama dengan model MobileNet sebelumnya.
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

print("\nTotal dataset:")
print(f"diseased: {total_diseased}")
print(f"healthy : {total_healthy}")
print(f"total   : {total_dataset}")

# =========================================================
# 3. STRATIFIED TRAIN-VALIDATION SPLIT
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

print("\nTrain dataset:")
print(f"diseased: {train_diseased}")
print(f"healthy : {train_healthy}")
print(f"total   : {len(train_labels)}")

print("\nValidation dataset:")
print(f"diseased: {val_diseased}")
print(f"healthy : {val_healthy}")
print(f"total   : {len(val_labels)}")

# =========================================================
# 4. TF.DATA PIPELINE
# =========================================================

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    label = tf.cast(label, tf.float32)

    return image, label


AUTOTUNE = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(buffer_size=len(train_paths), seed=SEED)
train_ds = train_ds.map(load_image, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(load_image, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =========================================================
# 5. AUGMENTASI
# =========================================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.05),
    layers.RandomContrast(0.05),
], name="data_augmentation")

AUGMENTATION_DESCRIPTION = """
Data augmentation was applied only during the training stage using horizontal flipping,
random rotation, random zoom, and random contrast adjustment. The augmentation was used
to improve generalization and reduce overfitting.
"""

def visualize_augmentation(ds, augmentation_layer):
    plt.figure(figsize=(10, 10))

    for images, labels_batch in ds.take(1):
        for i in range(3):
            label_name = class_names[int(labels_batch[i].numpy())]

            plt.subplot(3, 2, 2 * i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"Original: {label_name}")
            plt.axis("off")

            augmented_image = augmentation_layer(
                tf.expand_dims(images[i], 0),
                training=True
            )

            plt.subplot(3, 2, 2 * i + 2)
            plt.imshow(augmented_image[0].numpy().astype("uint8"))
            plt.title("After Augmentation")
            plt.axis("off")

    plt.suptitle("Sample Data Augmentation Before vs After")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


visualize_augmentation(train_ds, data_augmentation)

# =========================================================
# 6. MODEL BINARY CLASSIFICATION - RESNET50
# =========================================================

base_model = tf.keras.applications.ResNet50(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(*IMAGE_SIZE, 3))

x = data_augmentation(inputs)

# PENTING:
# ResNet50 membutuhkan preprocessing khusus dari Keras.
# Format input diubah sesuai standar pretrained ImageNet ResNet50.
x = tf.keras.applications.resnet50.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(
    256,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
)(x)

x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(
    128,
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(1e-4)
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# =========================================================
# 7. CALLBACKS
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
# 8. PHASE 1 - TRAINING CLASSIFIER HEAD
# =========================================================

print("\n--- Phase 1: Training Classifier Head ---")

PHASE_1_LEARNING_RATE = 1e-3
PHASE_1_EPOCHS = 25
PHASE_1_PATIENCE = 8

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE_1_LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
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
    epochs=PHASE_1_EPOCHS,
    callbacks=get_callbacks(PHASE_1_PATIENCE)
)

# =========================================================
# 9. PHASE 2 - FINE-TUNING BACKBONE
# =========================================================

print("\n--- Phase 2: Fine-Tuning ResNet50 Backbone ---")

PHASE_2_LEARNING_RATE = 1e-5
PHASE_2_EPOCHS = 60
PHASE_2_PATIENCE = 12
UNFROZEN_LAST_LAYERS = 50

base_model.trainable = True

for layer in base_model.layers[:-UNFROZEN_LAST_LAYERS]:
    layer.trainable = False

# BatchNormalization tetap freeze agar statistik pretrained tidak rusak
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE_2_LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=history.epoch[-1] + 1,
    epochs=PHASE_2_EPOCHS,
    callbacks=get_callbacks(PHASE_2_PATIENCE)
)

# =========================================================
# 10. EVALUASI MODEL
# =========================================================

def evaluate_model(model, val_ds):
    y_true = []
    y_pred = []
    y_prob = []

    for images, labels_batch in val_ds:
        probs = model.predict(images, verbose=0)
        preds = (probs >= 0.5).astype(int).reshape(-1)

        y_true.extend(labels_batch.numpy().astype(int))
        y_pred.extend(preds)
        y_prob.extend(probs.reshape(-1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names,
        output_dict=True
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=class_names
    )

    return y_true, y_pred, y_prob, cm, report_dict, report_text


y_true, y_pred, y_prob, cm, report_dict, report_text = evaluate_model(model, val_ds)

# =========================================================
# 11. PLOT TRAINING CURVE
# =========================================================

def save_training_curve(h1, h2, save_path):
    acc = h1.history["accuracy"] + h2.history["accuracy"]
    val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]

    loss = h1.history["loss"] + h2.history["loss"]
    val_loss = h1.history["val_loss"] + h2.history["val_loss"]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.1)


save_training_curve(history, fine_tune_history, TRAINING_CURVE_PATH)

# =========================================================
# 12. PLOT CONFUSION MATRIX
# =========================================================

def save_confusion_matrix(cm, class_names, save_path):
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
    plt.title("Confusion Matrix - ResNet50")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show(block=False)
    plt.pause(0.1)


save_confusion_matrix(cm, class_names, CONFUSION_MATRIX_PATH)

print("\nClassification Report:")
print(report_text)

# =========================================================
# 13. SAVE MODEL
# =========================================================

model.save(str(FINAL_MODEL_PATH))

# =========================================================
# 14. EXPORT TXT JURNAL-FRIENDLY
# =========================================================

def export_journal_friendly_report():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_params = model.count_params()
    trainable_params = int(np.sum([
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    ]))
    non_trainable_params = int(np.sum([
        tf.keras.backend.count_params(w)
        for w in model.non_trainable_weights
    ]))

    final_train_acc = history.history["accuracy"][-1]
    final_train_loss = history.history["loss"][-1]
    final_val_acc = report_dict["accuracy"]

    best_val_acc_phase_1 = max(history.history["val_accuracy"])
    best_val_acc_phase_2 = max(fine_tune_history.history["val_accuracy"])

    tn, fp, fn, tp = cm.ravel()

    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write("JOURNAL-FRIENDLY TRAINING REPORT\n")
        f.write("================================\n\n")

        f.write("1. Experiment Information\n")
        f.write("-------------------------\n")
        f.write(f"Report generated at       : {now}\n")
        f.write(f"Python version            : {platform.python_version()}\n")
        f.write(f"TensorFlow version        : {tf.__version__}\n")
        f.write(f"Random seed               : {SEED}\n\n")

        f.write("2. Dataset Description\n")
        f.write("----------------------\n")
        f.write("Dataset name              : Tomato Binary Dataset\n")
        f.write("Dataset task              : Binary image classification\n")
        f.write("Classes                   : diseased, healthy\n")
        f.write(f"Dataset directory         : {base_dir.resolve()}\n")
        f.write(f"Total images              : {total_dataset}\n")
        f.write(f"Total diseased images     : {total_diseased}\n")
        f.write(f"Total healthy images      : {total_healthy}\n")
        f.write(
            "Dataset construction      : Original tomato leaf disease classes were grouped into a single "
            "'diseased' class, while the healthy tomato leaf class was assigned to the 'healthy' class.\n"
        )
        f.write(
            "Class balancing strategy  : The dataset was balanced to contain an equal number of images "
            "for each class.\n\n"
        )

        f.write("3. Train-Validation Split\n")
        f.write("-------------------------\n")
        f.write("Split method              : Stratified train-validation split\n")
        f.write(f"Validation ratio          : {VALIDATION_SPLIT}\n")
        f.write(f"Training images           : {len(train_labels)}\n")
        f.write(f"Training diseased images  : {train_diseased}\n")
        f.write(f"Training healthy images   : {train_healthy}\n")
        f.write(f"Validation images         : {len(val_labels)}\n")
        f.write(f"Validation diseased images: {val_diseased}\n")
        f.write(f"Validation healthy images : {val_healthy}\n\n")

        f.write("4. Image Preprocessing\n")
        f.write("----------------------\n")
        f.write(f"Input image size          : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} pixels\n")
        f.write(f"Batch size                : {BATCH_SIZE}\n")
        f.write("Color format              : RGB\n")
        f.write("Image resizing            : TensorFlow resize operation\n")
        f.write(
            "Preprocessing method      : tf.keras.applications.resnet50.preprocess_input was applied "
            "to match the input format expected by ResNet50 pretrained on ImageNet.\n\n"
        )

        f.write("5. Data Augmentation\n")
        f.write("--------------------\n")
        f.write("Augmentation applied      : Yes\n")
        f.write("Augmentation stage        : Training pipeline only\n")
        f.write("Augmentation techniques   :\n")
        f.write("- Random horizontal flip\n")
        f.write("- Random rotation with factor 0.05\n")
        f.write("- Random zoom with factor 0.05\n")
        f.write("- Random contrast with factor 0.05\n\n")

        f.write("6. Model Architecture\n")
        f.write("---------------------\n")
        f.write(f"Base model                : {BASE_MODEL_NAME}\n")
        f.write(f"Pretrained weights        : {TRANSFER_LEARNING_WEIGHTS}\n")
        f.write("Include top               : False\n")
        f.write("Pooling layer             : GlobalAveragePooling2D\n")
        f.write(
            "Classifier head           : BatchNormalization -> Dense(256, ReLU, L2=1e-4) -> "
            "BatchNormalization -> Dropout(0.4) -> Dense(128, ReLU, L2=1e-4) -> "
            "Dropout(0.3) -> Dense(1, Sigmoid)\n"
        )
        f.write("Output activation         : Sigmoid\n")
        f.write("Classification type       : Binary classification\n")
        f.write(f"Total parameters          : {total_params}\n")
        f.write(f"Trainable parameters      : {trainable_params}\n")
        f.write(f"Non-trainable parameters  : {non_trainable_params}\n\n")

        f.write("6A. Model Flow Diagram\n")
        f.write("----------------------\n")
        f.write("The proposed binary classification model follows the architecture flow below:\n\n")

        f.write("+-----------------------------+\n")
        f.write("| Input Image                 |\n")
        f.write(f"| {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} x 3 RGB           |\n")
        f.write("+-------------+---------------+\n")
        f.write("              |\n")
        f.write("              v\n")
        f.write("+-----------------------------+\n")
        f.write("| Data Augmentation           |\n")
        f.write("| - Random horizontal flip    |\n")
        f.write("| - Random rotation           |\n")
        f.write("| - Random zoom               |\n")
        f.write("| - Random contrast           |\n")
        f.write("+-------------+---------------+\n")
        f.write("              |\n")
        f.write("              v\n")
        f.write("+-----------------------------+\n")
        f.write("| ResNet50 Preprocessing      |\n")
        f.write("| resnet50.preprocess_input   |\n")
        f.write("+-------------+---------------+\n")
        f.write("              |\n")
        f.write("              v\n")
        f.write("+-----------------------------+\n")
        f.write("| ResNet50 Backbone           |\n")
        f.write("| Pretrained on ImageNet      |\n")
        f.write("| include_top = False         |\n")
        f.write("+-------------+---------------+\n")
        f.write("              |\n")
        f.write("              v\n")
        f.write("+-----------------------------+\n")
        f.write("| GlobalAveragePooling2D      |\n")
        f.write("+-------------+---------------+\n")
        f.write("              |\n")
        f.write("              v\n")
        f.write("+-----------------------------+\n")
        f.write("| BatchNormalization          |\n")
        f.write("+-------------+---------------+\n")
        f.write("              |\n")
        f.write("              v\n")
        f.write("+-----------------------------+\n")
        f.write("| Dense Layer                 |\n")
        f.write("| 256 neurons, ReLU           |\n")
        f.write("| L2 regularization = 1e-4    |\n")
        f.write("+-------------+---------------+\n")
        f.write("              |\n")
        f.write("              v\n")
        f.write("+-----------------------------+\n")
        f.write("| BatchNormalization          |\n")
        f.write("+-------------+---------------+\n")
        f.write("              |\n")
        f.write("              v\n")
        f.write("+-----------------------------+\n")
        f.write("| Dropout                     |\n")
        f.write("| rate = 0.4                  |\n")
        f.write("+-------------+---------------+\n")
        f.write("              |\n")
        f.write("              v\n")
        f.write("+-----------------------------+\n")
        f.write("| Dense Layer                 |\n")
        f.write("| 128 neurons, ReLU           |\n")
        f.write("| L2 regularization = 1e-4    |\n")
        f.write("+-------------+---------------+\n")
        f.write("              |\n")
        f.write("              v\n")
        f.write("+-----------------------------+\n")
        f.write("| Dropout                     |\n")
        f.write("| rate = 0.3                  |\n")
        f.write("+-------------+---------------+\n")
        f.write("              |\n")
        f.write("              v\n")
        f.write("+-----------------------------+\n")
        f.write("| Output Layer                |\n")
        f.write("| Dense(1), Sigmoid           |\n")
        f.write("+-------------+---------------+\n")
        f.write("              |\n")
        f.write("              v\n")
        f.write("+-----------------------------+\n")
        f.write("| Prediction Output           |\n")
        f.write("| 0 = Diseased                |\n")
        f.write("| 1 = Healthy                 |\n")
        f.write("| threshold = 0.5             |\n")
        f.write("+-----------------------------+\n\n")

        f.write("Model flow summary:\n")
        f.write(
            "Input images were resized to 224 x 224 x 3 and augmented during training. "
            "The augmented images were preprocessed using the ResNet50 preprocessing function "
            "and then passed into a ResNet50 backbone pretrained on ImageNet for feature extraction. "
            "The extracted feature maps were reduced using global average pooling, followed by "
            "batch normalization, dense layers with ReLU activation, dropout regularization, and "
            "a sigmoid output layer for binary classification into diseased or healthy classes.\n\n"
        )

        f.write("7. Training Configuration\n")
        f.write("-------------------------\n")
        f.write("Loss function             : Binary Crossentropy\n")
        f.write(f"Label smoothing           : {LABEL_SMOOTHING}\n")
        f.write("Optimizer phase 1         : Adam\n")
        f.write(f"Learning rate phase 1     : {PHASE_1_LEARNING_RATE}\n")
        f.write(f"Epochs phase 1            : {PHASE_1_EPOCHS}\n")
        f.write(f"Early stopping phase 1    : patience={PHASE_1_PATIENCE}, monitor=val_accuracy\n")
        f.write("Optimizer phase 2         : Adam\n")
        f.write(f"Learning rate phase 2     : {PHASE_2_LEARNING_RATE}\n")
        f.write(f"Epochs phase 2            : {PHASE_2_EPOCHS}\n")
        f.write(f"Fine-tuned layers         : Last {UNFROZEN_LAST_LAYERS} layers of the ResNet50 base model\n")
        f.write("BatchNormalization layers : Frozen during fine-tuning\n")
        f.write(f"Early stopping phase 2    : patience={PHASE_2_PATIENCE}, monitor=val_accuracy\n")
        f.write("Learning rate scheduler   : ReduceLROnPlateau, factor=0.2, patience=3, min_lr=1e-6\n\n")

        f.write("8. Evaluation Metrics\n")
        f.write("---------------------\n")
        f.write("Metrics used              : Accuracy, Precision, Recall, AUC, F1-score\n")
        f.write("Decision threshold        : 0.5\n")
        f.write("Evaluation set            : Validation set\n\n")

        f.write("9. Final Classification Report\n")
        f.write("------------------------------\n")
        f.write(report_text)
        f.write("\n\n")

        f.write("10. Confusion Matrix\n")
        f.write("--------------------\n")
        f.write("Class order               : diseased, healthy\n")
        f.write("Matrix format             : rows=true labels, columns=predicted labels\n\n")
        f.write(str(cm))
        f.write("\n\n")
        f.write(f"True Diseased predicted Diseased : {tn}\n")
        f.write(f"True Diseased predicted Healthy  : {fp}\n")
        f.write(f"True Healthy predicted Diseased  : {fn}\n")
        f.write(f"True Healthy predicted Healthy   : {tp}\n\n")

        f.write("11. Final Training Summary\n")
        f.write("--------------------------\n")
        f.write(f"Final training accuracy   : {final_train_acc:.4f}\n")
        f.write(f"Final training loss       : {final_train_loss:.4f}\n")
        f.write(f"Final validation accuracy : {final_val_acc:.4f}\n")
        f.write(f"Best val accuracy phase 1 : {best_val_acc_phase_1:.4f}\n")
        f.write(f"Best val accuracy phase 2 : {best_val_acc_phase_2:.4f}\n\n")

        f.write("12. Saved Outputs\n")
        f.write("-----------------\n")
        f.write(f"Best model path           : {BEST_MODEL_PATH.resolve()}\n")
        f.write(f"Final model path          : {FINAL_MODEL_PATH.resolve()}\n")
        f.write(f"Training curve image      : {TRAINING_CURVE_PATH.resolve()}\n")
        f.write(f"Confusion matrix image    : {CONFUSION_MATRIX_PATH.resolve()}\n")
        f.write(f"Text report path          : {REPORT_TXT_PATH.resolve()}\n\n")

        f.write("13. Suggested Journal Method Description\n")
        f.write("----------------------------------------\n")
        f.write(
            "This study performed binary classification of tomato leaf images into diseased and healthy classes. "
            "The dataset consisted of 6,000 images, with 3,000 images assigned to each class. "
            "A stratified train-validation split was applied with an 80:20 ratio, resulting in 4,800 training images "
            "and 1,200 validation images. The validation set contained 600 diseased and 600 healthy images. "
            "Images were resized to 224 x 224 pixels and processed using a TensorFlow data pipeline. "
            "Data augmentation, including random horizontal flipping, rotation, zooming, and contrast adjustment, "
            "was applied during training to improve generalization. The model used ResNet50 pretrained on ImageNet "
            "as the feature extractor, followed by a custom classifier head consisting of global average pooling, "
            "batch normalization, dense layers, dropout, and a sigmoid output layer. Training was conducted in two phases: "
            "first, only the classifier head was trained while the base model was frozen; second, the last 50 layers "
            "of the ResNet50 base model were fine-tuned with a lower learning rate. The model was optimized using Adam "
            "and binary cross-entropy loss. Performance was evaluated using accuracy, precision, recall, F1-score, AUC, "
            "and confusion matrix analysis."
        )
        f.write("\n\n")

        f.write("14. Important Methodological Note\n")
        f.write("---------------------------------\n")
        f.write(
            "If augmented images were generated before splitting the dataset, there is a possibility that visually similar "
            "augmented variants of the same original image may appear in both training and validation sets. "
            "For stricter experimental validity, the recommended approach is to split the original images first and "
            "apply augmentation only to the training subset."
        )
        f.write("\n")


export_journal_friendly_report()

print("\nTraining selesai.")
print(f"Best model saved to          : {BEST_MODEL_PATH}")
print(f"Final model saved to         : {FINAL_MODEL_PATH}")
print(f"Training curve saved to      : {TRAINING_CURVE_PATH}")
print(f"Confusion matrix saved to    : {CONFUSION_MATRIX_PATH}")
print(f"Journal-friendly TXT saved to: {REPORT_TXT_PATH}")

plt.show()