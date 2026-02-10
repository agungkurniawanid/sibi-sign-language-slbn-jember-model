import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import (
    Input, LSTM, Dense, Dropout,
    BatchNormalization, Attention,
    GlobalAveragePooling1D, Bidirectional, LayerNormalization,
    Flatten, Concatenate
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2
import matplotlib.pyplot as plt

# ===============================
# IMPORT CONFIG
# ===============================
from Config import DATA_PATH, actions, sequence_length
# Asumsi: actions adalah list 10 kelas, sequence_length=30, DATA_PATH adalah path dataset

# ===============================
# 1. LOAD DATASET + AUGMENTASI CERDAS (Diperbaiki)
# ===============================
print("🔄 Memuat dataset & Melakukan Augmentasi...")
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

# Fungsi augmentasi baru: Lebih variatif untuk mengurangi kebingungan antar class serupa
# - Noise Gaussian
# - Scaling
# - Time shifting (geser sequence sedikit untuk variasi temporal)
# - Mirroring (flip horizontal untuk landmark tangan, asumsi x-coords di index genap)
def augment_sequence(original):
    augmented = original.copy()
    
    # Augmentasi 1: Noise
    noise = np.random.normal(0, 0.02, augmented.shape)
    augmented += noise
    
    # Augmentasi 2: Scaling
    scale = np.random.uniform(0.9, 1.1)  # Lebih variatif dari sebelumnya
    augmented *= scale
    
    # Augmentasi 3: Time shifting (roll sequence)
    shift = np.random.randint(-3, 3)  # Geser max 3 frames
    augmented = np.roll(augmented, shift, axis=0)
    
    # Augmentasi 4: Mirroring (flip x-coordinates untuk simulasi tangan kiri/kanan)
    # Asumsi landmark: [x1,y1,z1, x2,y2,z2, ...] -> flip semua x (index 0,3,6,... mod 3==0)
    if np.random.rand() > 0.5:  # 50% chance
        augmented[:, ::3] = -augmented[:, ::3]  # Flip x-coords
    
    return augmented

# Target augmentasi: 1 original + 2 augmented per sequence -> 3x data (dari 200 menjadi ~600 per class)
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    sequence_folders = sorted(
        [f for f in os.listdir(action_path) if f.isdigit()],
        key=int
    )
    for sequence in sequence_folders:
        window = []
        try:
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(action_path, sequence, f"{frame_num}.npy"))
                window.append(res)
            
            original_window = np.array(window)
            
            # Normalisasi data: Subtract mean, divide std (per sequence, untuk stability)
            mean = np.mean(original_window, axis=0)
            std = np.std(original_window, axis=0) + 1e-6  # Avoid div by zero
            original_window = (original_window - mean) / std
            
            # DATA 1: ORIGINAL
            sequences.append(original_window)
            labels.append(label_map[action])
            
            # DATA 2 & 3: AUGMENTED (2 variasi)
            for _ in range(2):
                augmented_window = augment_sequence(original_window)
                sequences.append(augmented_window)
                labels.append(label_map[action])
        
        except Exception as e:
            print(f"⚠️ Skip {action}/{sequence}: {e}")

X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions))
print(f"✅ Total Dataset (Original + Augmentasi): {X.shape}")
# Ekspektasi: ~6000 samples (10 kelas x 200 seq x 3) , (6000, 30, 1662)
print("🎯 Jumlah kelas:", len(actions))

# ===============================
# 2. SPLIT DATA (Stratified)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 20% test
    random_state=42,
    stratify=labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,  # 10% dari train untuk val
    random_state=42,
    stratify=y_train
)

# ===============================
# 3. MODEL ARCHITECTURE (Diperbaiki untuk mengurangi kebingungan)
# ===============================
# Tambah: Lebih dalam tapi dengan regularisasi kuat
# Tambah: Multi-head attention sederhana (via Concatenate)
inputs = Input(shape=(sequence_length, X.shape[2]))

# Layer 1: Bidirectional LSTM
x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)))(inputs)  # Naikkan units dari 64 ke 128
x = LayerNormalization()(x)  # Ganti BatchNorm ke LayerNorm untuk sequence stability
x = Dropout(0.5)(x)  # Naikkan dropout

# Layer 2: Bidirectional LSTM kedua
x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.001)))(x)
x = LayerNormalization()(x)
x = Dropout(0.5)(x)

# Attention: Multi-query attention sederhana (2 heads)
attn1 = Attention()([x, x])
attn2 = Attention()([x, x])
attn = Concatenate()([attn1, attn2])  # Combine attentions
x = GlobalAveragePooling1D()(attn)

# Classifier Head
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)  # Naikkan units
x = LayerNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.4)(x)
outputs = Dense(len(actions), activation='softmax')(x)

model = Model(inputs, outputs)

# Optimizer dengan clipnorm untuk stability
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',  # Bisa ganti ke focal loss jika perlu (lihat bawah)
    metrics=['accuracy']
)
model.summary()

# Optional: Focal Loss untuk handle class sulit (kebingungan spesifik)
# def focal_loss(gamma=2.0, alpha=0.25):
#     def loss(y_true, y_pred):
#         ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
#         pt = tf.exp(-ce)
#         return alpha * (1 - pt)**gamma * ce
#     return loss
# model.compile(optimizer=optimizer, loss=focal_loss(), metrics=['accuracy'])

# ===============================
# 4. CALLBACKS (Diperbaiki)
# ===============================
checkpoint = ModelCheckpoint(
    'model_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=40,  # Lebih sabar
    restore_best_weights=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,  # Lebih agresif reduce
    patience=8,
    min_lr=1e-7,
    verbose=1
)

# ===============================
# 5. TRAINING
# ===============================
print("\n🚀 Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,  # Naikkan epochs, early stop akan handle
    batch_size=16,  # Kurangi batch size untuk generalisasi lebih baik pada data kecil
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ===============================
# 6. EVALUASI AKHIR (Ditambah Confusion Matrix)
# ===============================
print("\n🧪 Evaluasi Test (Menggunakan model terbaik):")
best_model = tf.keras.models.load_model('model_best.keras')
loss, acc = best_model.evaluate(X_test, y_test)
print(f"🏆 Test Accuracy: {acc * 100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")

# Confusion Matrix untuk analisis kebingungan (e.g., IBU vs AGAR, BUAH vs SAYA/SABAR)
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=actions)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()  # Tampilkan plot (di environment yang support, seperti Jupyter)

# Simpan final model
best_model.save("model_final.h5")
print("💾 Model terbaik disimpan sebagai model_final.h5")