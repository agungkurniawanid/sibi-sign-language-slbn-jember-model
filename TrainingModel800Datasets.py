import os
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ===============================
# IMPORT CONFIG
# ===============================
from Config import DATA_PATH, actions, sequence_length

# ===============================
# 1. LOAD DATASET DENGAN AUGMENTASI
# ===============================
print("🔄 Memuat dataset landmark dengan augmentasi...")

label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

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
                res = np.load(
                    os.path.join(action_path, sequence, f"{frame_num}.npy")
                )

                # 1️⃣ Augmentasi noise kecil
                res = res + np.random.normal(0, 0.001, size=res.shape)

                # 2️⃣ Augmentasi scaling (0.9 - 1.1)
                scale_factor = np.random.uniform(0.9, 1.1)
                res = res * scale_factor

                window.append(res)

            sequences.append(window)
            labels.append(label_map[action])

        except Exception as e:
            print(f"⚠️ Skip {action}/{sequence}: {e}")

X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions))

print("✅ Total data:", X.shape)
print("🎯 Kelas:", actions)

# ===============================
# 2. SPLIT DATA (TRAIN / VAL / TEST)
# ===============================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42
)

print("📊 Train:", X_train.shape)
print("📊 Val  :", X_val.shape)
print("📊 Test :", X_test.shape)

# ===============================
# 3. MODEL LSTM (RINGAN & STABIL)
# ===============================
model = Sequential([
    # LSTM 1 - ekstraksi pola kasar
    LSTM(64, return_sequences=True, input_shape=(sequence_length, X.shape[2])),
    BatchNormalization(),
    Dropout(0.4),

    # LSTM 2 - refinement temporal
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.4),

    # Dense classifier
    Dense(64, activation='relu'),
    Dropout(0.4),
    Dense(len(actions), activation='softmax')
])

optimizer = Adam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# 4. CALLBACKS
# ===============================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=30,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_lstm_action.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# ===============================
# 5. TRAINING
# ===============================
print("\n🚀 Memulai training model...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=300,
    batch_size=16,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# ===============================
# 6. EVALUASI FINAL
# ===============================
print("\n🧪 Evaluasi pada data TEST:")
loss, acc = model.evaluate(X_test, y_test)
print(f"🏆 Akurasi Test: {acc * 100:.2f}%")

# ===============================
# 7. SIMPAN MODEL FINAL
# ===============================
model.save("final_lstm_action.h5")
print("💾 Model disimpan sebagai final_lstm_action.h5")
