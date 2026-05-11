import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras import layers, models
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- 1. KONFIGURASI ---
BASE_DIR = Path(__file__).resolve().parent
base_dir = BASE_DIR.parent.parent.parent / "TomatoDataset"

IMAGE_SIZE = (300, 300)
BATCH_SIZE = 32
SEED = 123
CLASS_WEIGHT_MODE = "sqrt"
LABEL_SMOOTHING = 0.05
BEST_MODEL_PATH = BASE_DIR / "best_model_base.keras"
FINAL_MODEL_PATH = BASE_DIR / "mobilenet_v3_base.keras"

tf.keras.utils.set_random_seed(SEED)

# --- 2. LOAD DATASET ---
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
num_classes = len(class_names)

# --- 3. AUGMENTASI & VISUALISASI BEFORE-AFTER ---
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomBrightness(0.1),
])

def visualize_augmentation(ds, augmentation_layer):
    plt.figure(figsize=(10, 10))
    # Ambil 1 batch
    for images, labels in ds.take(1):
        for i in range(3): # Tampilkan 3 contoh
            # Original
            ax = plt.subplot(3, 2, 2*i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(f"Original: {class_names[np.argmax(labels[i])]}")
            plt.axis("off")
            
            # Augmented
            augmented_image = augmentation_layer(tf.expand_dims(images[i], 0))
            ax = plt.subplot(3, 2, 2*i + 2)
            plt.imshow(augmented_image[0].numpy().astype("uint8"))
            plt.title("After Augmentation")
            plt.axis("off")
    
    plt.suptitle("Sample Data Augmentation Before vs After")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

visualize_augmentation(train_ds, data_augmentation)

# --- 4. ARSITEKTUR MODEL (Dioptimalkan untuk Overfitting) ---
base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False 

inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

# Sederhanakan Head: 1 Layer Dense dengan Dropout lebih tinggi (0.5)
x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x) 

outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

# --- 5. CLASS WEIGHTS & PREFETCH ---
y_train = np.concatenate([y for x, y in train_ds], axis=0)
y_train_integers = np.argmax(y_train, axis=1)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_integers), y=y_train_integers)

if CLASS_WEIGHT_MODE == "sqrt":
    selected_weights = np.sqrt(weights)
    selected_weights = selected_weights / np.mean(selected_weights)
    class_weights = {int(i): float(w) for i, w in zip(np.unique(y_train_integers), selected_weights)}
else:
    class_weights = None

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# --- 6. TRAINING ---
def get_callbacks(patience_es):
    return [
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(str(BEST_MODEL_PATH), monitor='val_accuracy', save_best_only=True, mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience_es, restore_best_weights=True, verbose=1)
    ]

# Phase 1: Training Head
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), 
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
              metrics=['accuracy'])

history = model.fit(train_ds, validation_data=val_ds, epochs=25, class_weight=class_weights, callbacks=get_callbacks(8))

# Phase 2: Fine-Tuning
base_model.trainable = True
# Freeze awal, buka 50 layer terakhir saja
for layer in base_model.layers[:-50]:
    layer.trainable = False
# Penting: BN tetap freeze agar tidak merusak statistik ImageNet
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
              metrics=['accuracy'])

fine_tune_history = model.fit(train_ds, validation_data=val_ds, initial_epoch=history.epoch[-1], 
                              epochs=60, class_weight=class_weights, callbacks=get_callbacks(12))

# --- 7. EVALUASI & VISUALISASI AKHIR ---
def plot_final_results(h1, h2, model, val_ds, class_names):
    acc = h1.history['accuracy'] + h2.history['accuracy']
    val_acc = h1.history['val_accuracy'] + h2.history['val_accuracy']
    loss = h1.history['loss'] + h2.history['loss']
    val_loss = h1.history['val_loss'] + h2.history['val_loss']
    
    # Plot Accuracy & Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Acc')
    plt.plot(val_acc, label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.show(block=False)
    plt.pause(0.1)

    # Confusion Matrix
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.show(block=False)
    plt.pause(0.1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

# Tampilkan hasil akhir
plot_final_results(history, fine_tune_history, model, val_ds, class_names)

# Save Final
model.save(str(FINAL_MODEL_PATH))

print("\nSemua proses selesai. Tutup semua jendela grafik untuk keluar.")
plt.show() # Baris ini menahan semua figure agar tidak tertutup