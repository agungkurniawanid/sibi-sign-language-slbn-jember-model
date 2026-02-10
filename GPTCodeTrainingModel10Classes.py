# =========================================================
# FINAL TRAINING CODE - GESTURE RECOGNITION LSTM + ATTENTION
# =========================================================

import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from keras.utils import to_categorical
from keras.models import Model
from keras.layers import (
    Input, LSTM, Dense, Dropout,
    Bidirectional, LayerNormalization,
    GlobalAveragePooling1D,
    MultiHeadAttention
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l2

# =========================================================
# CONFIG
# =========================================================
from Config import DATA_PATH, actions, sequence_length

NUM_CLASSES = len(actions)
RANDOM_STATE = 42

# =========================================================
# TEMPORAL AUGMENTATION
# =========================================================
def temporal_augment(sequence, drop_prob=0.1):
    seq = sequence.copy()
    for i in range(seq.shape[0]):
        if np.random.rand() < drop_prob:
            seq[i] = seq[max(i - 1, 0)]
    return seq

# =========================================================
# LOAD DATASET + AUGMENTATION
# =========================================================
print("🔄 Loading dataset & augmentation...")

label_map = {label: idx for idx, label in enumerate(actions)}
sequences, labels = [], []

for action in actions:
    action_path = os.path.join(DATA_PATH, action)
    sequence_folders = sorted(
        [f for f in os.listdir(action_path) if f.isdigit()],
        key=int
    )

    for seq_folder in sequence_folders:
        try:
            window = []
            for frame_num in range(sequence_length):
                frame_path = os.path.join(
                    action_path, seq_folder, f"{frame_num}.npy"
                )
                window.append(np.load(frame_path))

            original_window = np.array(window)

            # -------------------------------
            # ORIGINAL DATA
            # -------------------------------
            sequences.append(original_window)
            labels.append(label_map[action])

            # -------------------------------
            # AUGMENTED DATA
            # -------------------------------
            noise = np.random.normal(0, 0.02, original_window.shape)
            scale = np.random.uniform(0.95, 1.05)

            augmented_window = temporal_augment(original_window)
            augmented_window = (augmented_window + noise) * scale

            sequences.append(augmented_window)
            labels.append(label_map[action])

        except Exception as e:
            print(f"⚠️ Skip {action}/{seq_folder}: {e}")

X = np.array(sequences)
y = to_categorical(labels, num_classes=NUM_CLASSES)

print(f"✅ Dataset shape : {X.shape}")
print(f"🎯 Classes       : {NUM_CLASSES}")

# =========================================================
# SPLIT DATA (STRATIFIED)
# =========================================================
X_train, X_test, y_train, y_test, labels_train, labels_test = train_test_split(
    X, y, labels,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=np.argmax(y_train, axis=1)
)

# =========================================================
# MODEL ARCHITECTURE
# =========================================================
inputs = Input(shape=(sequence_length, X.shape[2]))

# -------- LSTM BLOCK 1 --------
x = Bidirectional(
    LSTM(
        96,
        return_sequences=True,
        kernel_regularizer=l2(0.001)
    )
)(inputs)
x = LayerNormalization()(x)
x = Dropout(0.4)(x)

# -------- LSTM BLOCK 2 --------
x = Bidirectional(
    LSTM(
        96,
        return_sequences=True,
        kernel_regularizer=l2(0.001)
    )
)(x)
x = LayerNormalization()(x)
x = Dropout(0.4)(x)

# -------- MULTI-HEAD SELF ATTENTION --------
attn = MultiHeadAttention(
    num_heads=4,
    key_dim=64,
    dropout=0.3
)(x, x)

x = LayerNormalization()(x + attn)
x = GlobalAveragePooling1D()(x)

# -------- CLASSIFIER HEAD --------
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = LayerNormalization()(x)
x = Dropout(0.5)(x)

x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.4)(x)

outputs = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs, outputs)

# =========================================================
# COMPILE
# =========================================================
optimizer = Adam(learning_rate=5e-4)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================================================
# CALLBACKS
# =========================================================
checkpoint = ModelCheckpoint(
    "model_best.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=30,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# =========================================================
# TRAINING
# =========================================================
print("\n🚀 Training model...")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=250,
    batch_size=32,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# =========================================================
# EVALUATION
# =========================================================
print("\n🧪 Evaluating best model...")
best_model = tf.keras.models.load_model("model_best.keras")

loss, acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"🏆 Test Accuracy : {acc * 100:.2f}%")
print(f"📉 Test Loss     : {loss:.4f}")

# =========================================================
# CONFUSION MATRIX
# =========================================================
y_pred = np.argmax(best_model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=actions))

cm = confusion_matrix(y_true, y_pred)
print("\n🧩 Confusion Matrix:")
print(cm)

# =========================================================
# SAVE FINAL MODEL
# =========================================================
best_model.save("model_final.h5")
print("💾 Model saved as model_final.h5")
