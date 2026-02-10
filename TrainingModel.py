import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import (
    Input, LSTM, Dense, Dropout,
    BatchNormalization, Attention,
    GlobalAveragePooling1D, Bidirectional, LayerNormalization,
    Flatten
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2

# ===============================
# IMPORT CONFIG
# ===============================
from Config import DATA_PATH, actions, sequence_length

# ===============================
# 1. LOAD DATASET + AUGMENTASI CERDAS
# ===============================
print("🔄 Memuat dataset & Melakukan Augmentasi...")

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

# Target augmentasi: Kita buat 1 data asli + 1 data noise = 2x lipat data
for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    sequence_folders = sorted(
        [f for f in os.listdir(action_path) if f.isdigit()],
        key=int
    )

    for sequence in sequence_folders:
        window = []
        try:
            # Load satu sequence utuh
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(action_path, sequence, f"{frame_num}.npy"))
                window.append(res)
            
            original_window = np.array(window)
            
            # -----------------------------------------------
            # DATA 1: ORIGINAL (Murni)
            # -----------------------------------------------
            sequences.append(original_window)
            labels.append(label_map[action])

            # -----------------------------------------------
            # DATA 2: AUGMENTED (Noise + Scale Kecil)
            # -----------------------------------------------
            # Tambah noise
            noise = np.random.normal(0, 0.02, original_window.shape)
            # Scaling acak (sedikit membesar/mengecil)
            scale = np.random.uniform(0.95, 1.05)
            
            augmented_window = (original_window + noise) * scale
            
            sequences.append(augmented_window)
            labels.append(label_map[action])

        except Exception as e:
            print(f"⚠️ Skip {action}/{sequence}: {e}")

X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions))

print(f"✅ Total Dataset (Original + Augmentasi): {X.shape}") 
# Ekspektasi: (3200, 30, 1662) jika sequence 30 frames
print("🎯 Jumlah kelas:", len(actions))

# ===============================
# 2. SPLIT DATA
# ===============================
# Split Stratified agar proporsi kelas seimbang di training dan test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2, # 20% untuk test (640 data)
    random_state=42,
    stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1, # 10% dari train untuk validasi
    random_state=42,
    stratify=y_train
)

# ===============================
# 3. MODEL ARCHITECTURE (Optimized for Small Data)
# ===============================
inputs = Input(shape=(sequence_length, X.shape[2]))

# Layer 1: LSTM untuk menangkap pola waktu
# Menggunakan L2 Regularizer (kernel_regularizer) untuk mencegah overfitting pada data kecil
x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(inputs)
x = BatchNormalization()(x) # Menstabilkan learning
x = Dropout(0.4)(x)

# Layer 2: LSTM kedua
x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001)))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

# Attention Mechanism (Membantu fokus pada frame penting)
attn = Attention()([x, x])
x = GlobalAveragePooling1D()(attn)

# Classifier Head
# Dense layer dikurangi tapi diberi regularisasi
x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)

x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x)

outputs = Dense(len(actions), activation='softmax')(x)

model = Model(inputs, outputs)

# Optimizer dengan learning rate sedikit lebih tinggi di awal karena ada BatchNormalization
optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# 4. CALLBACKS
# ===============================
checkpoint = ModelCheckpoint(
    'model_best.keras',  # Format baru keras direkomendasikan
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30, # Bersabar lebih lama
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# ===============================
# 5. TRAINING
# ===============================
print("\n🚀 Training model...")

# Batch size 16 atau 32 ideal untuk dataset kecil
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200, 
    batch_size=32, 
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ===============================
# 6. EVALUASI AKHIR
# ===============================
print("\n🧪 Evaluasi Test (Menggunakan model terbaik):")
# Load model terbaik yang disimpan checkpoint
best_model = tf.keras.models.load_model('model_best.keras')
loss, acc = best_model.evaluate(X_test, y_test)

print(f"🏆 Test Accuracy: {acc * 100:.2f}%")
print(f"📉 Test Loss: {loss:.4f}")

# Simpan final model (opsional, karena model_best.keras sudah yang terbaik)
best_model.save("model_final.h5")
print("💾 Model terbaik disimpan sebagai model_final.h5")