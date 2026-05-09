import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import os

# Konfigurasi Path
base_dir = r'D:\Projects\Web\smart-farming\ML_DL_Models\TomatoDataset'

# 1. Load Dataset dengan Shuffle yang lebih baik
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',
    shuffle=True # Pastikan data diacak agar model tidak belajar urutan folder
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    base_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

class_names = train_ds.class_names

# 2. Penambahan Augmentasi Sederhana & Arsitektur Base yang Lebih Kuat
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

base_model = tf.keras.applications.MobileNetV3Large(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False 

model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    data_augmentation,
    layers.Rescaling(1./255), 
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    
    # Satu layer Dense yang cukup kuat tanpa L2
    layers.Dense(256, activation='relu'), 
    layers.BatchNormalization(),
    layers.Dropout(0.4), 
    
    layers.Dense(10, activation='softmax')
])

# Menggunakan Learning Rate yang sedikit lebih rendah agar stabil
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=3, 
    min_lr=1e-6,
    verbose=1
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    decay_rate=0.9
)

initial_lr = 1e-3

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# 3. Training dengan Epoch lebih banyak
# Kita butuh minimal 20 epoch untuk melihat perkembangan dari 25% ke 60%


# Ambil label dari dataset training
y_train = np.concatenate([y for x, y in train_ds], axis=0)
y_train_integers = np.argmax(y_train, axis=1)

# Hitung bobot
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_integers),
    y=y_train_integers
)
class_weights = dict(enumerate(weights))

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=6, 
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'DL/MobileNetV3/Based/best_model_base.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

epochs = 50 # Tambah epoch, biarkan EarlyStopping atau LR Scheduler yang bekerja
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    class_weight=class_weights,
    callbacks=[reduce_lr, checkpoint, early_stop]
)

# 4. Visualisasi Graph Loss & Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Gunakan panjang data acc untuk sumbu X
actual_epochs = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(actual_epochs, acc, label='Training Accuracy')
plt.plot(actual_epochs, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(actual_epochs, loss, label='Training Loss')
plt.plot(actual_epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Simpan Model Base
model.save('DL/MobileNetV3/Based/mobilenet_v3_base.h5')


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

def plot_evaluation(model, val_ds, class_names):
    print("\nMenghitung evaluasi model...")
    
    # 1. Mengambil label asli dan prediksi
    y_true = []
    y_pred = []
    
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 2. Membuat Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Deteksi Penyakit Tomat')
    plt.ylabel('Label Sebenarnya')
    plt.xlabel('Prediksi Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # 3. Mencetak Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

# Panggil fungsi setelah model.fit()
plot_evaluation(model, val_ds, class_names)