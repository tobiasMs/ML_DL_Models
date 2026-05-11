import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# ============================================================
# 1. KONFIGURASI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
base_dir = BASE_DIR.parent.parent.parent / "TomatoDataset"

# Resolusi sedikit lebih besar supaya detail bercak daun lebih mudah ditangkap
IMAGE_SIZE = (256, 256)

# Lebih stabil untuk fine-tuning
BATCH_SIZE = 16

SEED = 123
VALIDATION_SPLIT = 0.2

# Regularisasi yang terlalu kuat sering menahan akurasi di bawah target
USE_MIXUP = False
MIXUP_ALPHA = 0.1

LABEL_SMOOTHING = 0.05

# Focal loss dipertahankan sebagai opsi, tetapi default-nya dimatikan
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 1.5
FOCAL_ALPHA = 0.25

# Class Weight
CLASS_WEIGHT_MODE = "sqrt"

BEST_MODEL_PATH = BASE_DIR / "best_model_mobilenetv3_v2.keras"
FINAL_MODEL_PATH = BASE_DIR / "mobilenetv3_v2.keras"

tf.keras.utils.set_random_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# 2. LOAD DATASET PATH
# ============================================================

if not base_dir.exists():
    raise FileNotFoundError(f"Dataset tidak ditemukan: {base_dir}")

valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

class_names = sorted([
    folder.name for folder in base_dir.iterdir()
    if folder.is_dir()
])

num_classes = len(class_names)

if num_classes == 0:
    raise ValueError("Folder class tidak ditemukan.")

class_to_index = {
    class_name: idx
    for idx, class_name in enumerate(class_names)
}

print("\nClass Names:")
for idx, name in enumerate(class_names):
    print(f"{idx}: {name}")

image_paths = []
labels = []

for class_name in class_names:
    class_dir = base_dir / class_name

    for img_path in class_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
            image_paths.append(str(img_path))
            labels.append(class_to_index[class_name])

image_paths = np.array(image_paths)
labels = np.array(labels)

if len(image_paths) == 0:
    raise ValueError("Tidak ada gambar ditemukan.")

print(f"\nTotal gambar: {len(image_paths)}")


# ============================================================
# 3. DISTRIBUSI DATASET
# ============================================================

print("\nDistribusi Dataset:")
print("=" * 90)

for i, class_name in enumerate(class_names):
    count = np.sum(labels == i)

    print(
        f"{i:2d} | {class_name:55s} | total: {count:5d}"
    )


# ============================================================
# 4. STRATIFIED SPLIT
# ============================================================

train_paths, val_paths, train_labels, val_labels = train_test_split(
    image_paths,
    labels,
    test_size=VALIDATION_SPLIT,
    random_state=SEED,
    stratify=labels
)

print("\nDistribusi Setelah Stratified Split:")
print("=" * 100)

for i, class_name in enumerate(class_names):
    train_count = np.sum(train_labels == i)
    val_count = np.sum(val_labels == i)

    print(
        f"{i:2d} | {class_name:55s} | "
        f"train: {train_count:5d} | val: {val_count:5d}"
    )


# ============================================================
# 5. LOAD IMAGE
# ============================================================

def load_image(path, label):
    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)

    image = tf.cast(image, tf.float32)

    label = tf.one_hot(label, depth=num_classes)

    return image, label


# ============================================================
# 6. DATA AUGMENTATION
# ============================================================

# AUGMENTASI DIPERINGAN
# supaya model tidak underfit

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.12),
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomContrast(0.1),
], name="data_augmentation")


# ============================================================
# 7. VISUALISASI AUGMENTASI
# ============================================================

def visualize_augmentation(ds, augmentation_layer):
    plt.figure(figsize=(10, 10))

    for images, labels in ds.take(1):

        for i in range(min(3, images.shape[0])):

            # ORIGINAL
            ax = plt.subplot(3, 2, 2 * i + 1)

            image = tf.clip_by_value(images[i], 0, 255)

            plt.imshow(image.numpy().astype("uint8"))

            plt.title(
                f"Original: "
                f"{class_names[np.argmax(labels[i])]}"
            )

            plt.axis("off")

            # AUGMENTED
            augmented_image = augmentation_layer(
                tf.expand_dims(images[i], 0),
                training=True
            )

            augmented_image = tf.clip_by_value(
                augmented_image[0],
                0,
                255
            )

            ax = plt.subplot(3, 2, 2 * i + 2)

            plt.imshow(
                augmented_image.numpy().astype("uint8")
            )

            plt.title("After Augmentation")

            plt.axis("off")

    plt.suptitle(
        "Sample Data Augmentation Before vs After"
    )

    plt.tight_layout()

    plt.show(block=False)

    plt.pause(0.1)


# ============================================================
# 8. MIXUP
# ============================================================

def sample_beta_distribution(size, alpha):
    gamma_1 = tf.random.gamma(
        shape=[size],
        alpha=alpha
    )

    gamma_2 = tf.random.gamma(
        shape=[size],
        alpha=alpha
    )

    return gamma_1 / (gamma_1 + gamma_2)


def mixup(images, labels, alpha=0.1):
    batch_size = tf.shape(images)[0]

    lam = sample_beta_distribution(
        batch_size,
        alpha
    )

    lam_x = tf.reshape(
        lam,
        (batch_size, 1, 1, 1)
    )

    lam_y = tf.reshape(
        lam,
        (batch_size, 1)
    )

    index = tf.random.shuffle(
        tf.range(batch_size)
    )

    mixed_images = (
        images * lam_x +
        tf.gather(images, index) * (1 - lam_x)
    )

    mixed_labels = (
        labels * lam_y +
        tf.gather(labels, index) * (1 - lam_y)
    )

    return mixed_images, mixed_labels


# ============================================================
# 9. BUILD DATASET
# ============================================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

train_ds = train_ds.shuffle(
    buffer_size=len(train_paths),
    seed=SEED,
    reshuffle_each_iteration=True
)

train_ds = train_ds.map(
    load_image,
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.batch(BATCH_SIZE)

# VISUALISASI AUGMENTASI
visualize_augmentation(
    train_ds,
    data_augmentation
)

# MIXUP
if USE_MIXUP:
    train_ds = train_ds.map(
        lambda x, y: mixup(
            x,
            y,
            MIXUP_ALPHA
        ),
        num_parallel_calls=AUTOTUNE
    )

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


# ============================================================
# 10. CLASS WEIGHT
# ============================================================

weights = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=train_labels
)

if CLASS_WEIGHT_MODE == "sqrt":

    selected_weights = np.sqrt(weights)

    selected_weights = (
        selected_weights /
        np.mean(selected_weights)
    )

    class_weights = {
        int(i): float(w)
        for i, w in zip(
            np.arange(num_classes),
            selected_weights
        )
    }

elif CLASS_WEIGHT_MODE == "balanced":

    class_weights = {
        int(i): float(w)
        for i, w in zip(
            np.arange(num_classes),
            weights
        )
    }

else:
    class_weights = None

print("\nClass Weights:")
print(class_weights)


# ============================================================
# 11. LOSS FUNCTION
# ============================================================

def categorical_focal_loss(
    gamma=2.0,
    alpha=0.5
):

    def loss(y_true, y_pred):

        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        cross_entropy = (
            -y_true *
            tf.math.log(y_pred)
        )

        focal_weight = (
            alpha *
            tf.pow(
                1.0 - y_pred,
                gamma
            )
        )

        focal_loss = (
            focal_weight *
            cross_entropy
        )

        return tf.reduce_sum(
            focal_loss,
            axis=1
        )

    return loss


loss_function = categorical_focal_loss(
    gamma=FOCAL_GAMMA,
    alpha=FOCAL_ALPHA
)


def get_loss_function():

    if USE_FOCAL_LOSS:
        return loss_function

    return tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=LABEL_SMOOTHING
    )


# ============================================================
# 12. ATTENTION BLOCK
# ============================================================

def se_block(x, ratio=8):
    channels = x.shape[-1]

    se = layers.GlobalAveragePooling2D()(x)

    se = layers.Dense(
        channels // ratio,
        activation="relu",
        kernel_initializer="he_normal"
    )(se)

    se = layers.Dense(
        channels,
        activation="sigmoid",
        kernel_initializer="he_normal"
    )(se)

    se = layers.Reshape(
        (1, 1, channels)
    )(se)

    x = layers.Multiply()([x, se])

    return x


def spatial_attention_block(x):

    avg_pool = layers.Lambda(
        lambda t: tf.reduce_mean(
            t,
            axis=-1,
            keepdims=True
        )
    )(x)

    max_pool = layers.Lambda(
        lambda t: tf.reduce_max(
            t,
            axis=-1,
            keepdims=True
        )
    )(x)

    concat = layers.Concatenate(
        axis=-1
    )([avg_pool, max_pool])

    attention = layers.Conv2D(
        filters=1,
        kernel_size=7,
        padding="same",
        activation="sigmoid"
    )(concat)

    x = layers.Multiply()([x, attention])

    return x


# ============================================================
# 13. BUILD MODEL
# ============================================================

def build_model():

    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True
    )

    base_model.trainable = False

    inputs = layers.Input(
        shape=(*IMAGE_SIZE, 3),
        name="input_image"
    )

    x = data_augmentation(inputs)

    # BACKBONE
    x = base_model(
        x,
        training=False
    )

    # Head yang lebih sederhana biasanya lebih stabil untuk fine-tuning
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])

    x = layers.BatchNormalization()(x)

    x = layers.Dense(
        384,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.35)(x)

    x = layers.Dense(
        192,
        activation="swish",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dropout(0.25)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(
        inputs,
        outputs,
        name="MobileNetV3Large_V2"
    )

    return model, base_model


model, base_model = build_model()

model.summary()


# ============================================================
# 14. CALLBACKS
# ============================================================

def get_callbacks(patience):

    return [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
    ]

# ============================================================
# 15. PHASE 1 TRAINING
# ============================================================

print("\n" + "=" * 80)
print("PHASE 1 TRAINING")
print("=" * 80)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=1e-3,
        weight_decay=1e-4
    ),
    loss=get_loss_function(),
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    class_weight=class_weights,
    callbacks=get_callbacks(8)
)


# ============================================================
# 16. PHASE 2 FINE TUNING
# ============================================================

print("\n" + "=" * 80)
print("PHASE 2 FINE TUNING")
print("=" * 80)

base_model.trainable = True

# Buka lebih banyak layer, tetapi tetap simpan bagian awal backbone
for layer in base_model.layers[:-80]:
    layer.trainable = False

# FREEZE BN
for layer in base_model.layers:
    if isinstance(
        layer,
        layers.BatchNormalization
    ):
        layer.trainable = False

trainable_layers = sum([
    1 for layer in base_model.layers
    if layer.trainable
])

print(
    f"\nTrainable backbone layer: "
    f"{trainable_layers}"
)

# COSINE DECAY
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=5e-5,
    decay_steps=1500
)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-5
)

model.compile(
    optimizer=optimizer,
    loss=get_loss_function(),
    metrics=["accuracy"]
)

fine_tune_history = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=len(history.history["accuracy"]),
    epochs=80,
    class_weight=class_weights,
    callbacks=get_callbacks(12)
)


# ============================================================
# 17. PLOT TRAINING HISTORY
# ============================================================

def plot_training_history(h1, h2):

    acc = (
        h1.history["accuracy"] +
        h2.history["accuracy"]
    )

    val_acc = (
        h1.history["val_accuracy"] +
        h2.history["val_accuracy"]
    )

    loss = (
        h1.history["loss"] +
        h2.history["loss"]
    )

    val_loss = (
        h1.history["val_loss"] +
        h2.history["val_loss"]
    )

    plt.figure(figsize=(14, 5))

    # ACCURACY
    plt.subplot(1, 2, 1)

    plt.plot(
        acc,
        label="Train Accuracy"
    )

    plt.plot(
        val_acc,
        label="Validation Accuracy"
    )

    plt.axvline(
        x=len(h1.history["accuracy"]) - 1,
        linestyle="--",
        label="Start Fine Tuning"
    )

    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # LOSS
    plt.subplot(1, 2, 2)

    plt.plot(
        loss,
        label="Train Loss"
    )

    plt.plot(
        val_loss,
        label="Validation Loss"
    )

    plt.axvline(
        x=len(h1.history["loss"]) - 1,
        linestyle="--",
        label="Start Fine Tuning"
    )

    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    plt.show(block=False)

    plt.pause(0.1)


plot_training_history(
    history,
    fine_tune_history
)


# ============================================================
# 18. EVALUASI MODEL
# ============================================================

def evaluate_model(
    model,
    val_ds,
    class_names
):

    print("\n" + "=" * 80)
    print("EVALUASI MODEL")
    print("=" * 80)

    val_loss, val_acc = model.evaluate(
        val_ds,
        verbose=1
    )

    print(f"\nValidation Loss     : {val_loss:.4f}")
    print(f"Validation Accuracy : {val_acc:.4f}")
    print(f"Validation Accuracy : {val_acc * 100:.2f}%")

    y_true = []
    y_pred = []

    for images, labels_batch in val_ds:

        preds = model.predict(
            images,
            verbose=0
        )

        y_true.extend(
            np.argmax(
                labels_batch.numpy(),
                axis=1
            )
        )

        y_pred.extend(
            np.argmax(
                preds,
                axis=1
            )
        )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels_range = np.arange(
        len(class_names)
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels_range,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels_range
    )

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted Label")

    plt.ylabel("True Label")

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.show(block=False)

    plt.pause(0.1)

    return (
        val_loss,
        val_acc,
        y_true,
        y_pred
    )


val_loss, val_acc, y_true, y_pred = evaluate_model(
    model,
    val_ds,
    class_names
)


# ============================================================
# 19. SAVE MODEL
# ============================================================

model.save(str(FINAL_MODEL_PATH))

print("\n" + "=" * 80)
print("TRAINING SELESAI")
print("=" * 80)

print(
    f"Best model tersimpan di : "
    f"{BEST_MODEL_PATH}"
)

print(
    f"Final model tersimpan di: "
    f"{FINAL_MODEL_PATH}"
)

print(
    f"Final validation accuracy: "
    f"{val_acc * 100:.2f}%"
)

print(
    "\nTutup semua jendela grafik untuk keluar."
)

plt.show()