# =============================================================================
# TestingRealtimeModel.py
# =============================================================================
# Tujuan:
#   Menjalankan deteksi bahasa isyarat SIBI secara realtime menggunakan model
#   AI (LSTM) yang telah dilatih. File ini membaca input dari kamera, mengekstrak
#   keypoints dengan MediaPipe Holistic, lalu memprediksi kata yang diperagakan
#   dan menampilkan hasilnya di layar.
#
#   Fitur utama:
#     - Deteksi realtime berbasis urutan 30 frame keypoints
#     - Mekanisme stabilisasi: prediksi hanya diterima jika konsisten selama
#       12 frame berturut-turut (mencegah hasil "gonta-ganti")
#     - Threshold kepercayaan minimum 80% sebelum kata diterima
#     - Mode mirror (flip horizontal) yang bisa di-toggle dengan tombol 'M'
#     - Visualisasi bar probabilitas per kelas di layar
#     - Auto-reset buffer saat tangan tidak terdeteksi
#
# Cara Penggunaan:
#   Pastikan file 'model_best.keras' tersedia di direktori yang sama, lalu:
#   python TestingRealtimeModel.py
#   Kontrol keyboard: 'Q' = keluar, 'M' = toggle mirror
#
# PENTING - Konsistensi dengan Dataset Collector:
#   - Resolusi kamera     : 1280x720 (sama dengan pengambilan dataset)
#   - Fitur per frame     : 258 (pose=132, lh=63, rh=63) — TANPA face landmark
#   - refine_face_landmarks: False
# =============================================================================

# ==================== IMPORT ====================
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import keras
from keras.layers import Layer
import os
import math
from Config import actions, sequence_length

# ==================== CUSTOM CLASSES (harus sama persis dengan saat training) ====================

@keras.saving.register_keras_serializable()
class AttentionLayer(Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()


def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        ce     = -y_true * tf.math.log(y_pred)
        pt     = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        return tf.reduce_mean(alpha * tf.pow(1.0 - pt, gamma) * ce)
    return loss_fn

# ==================== CONFIGURATION ====================
ACTIONS   = actions
COLORS    = [(245, 117, 16), (117, 245, 16)]
THRESHOLD = 0.80   # Akurasi minimal 80%
MODEL_PATH = 'TrainingModel/Training_SIBI_3Class/Models/model_sibi.keras'

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model '{MODEL_PATH}' tidak ditemukan!")
    exit()

print("🔄 Memuat Model AI...")
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'AttentionLayer': AttentionLayer,
        'loss_fn': focal_loss(),
    }
)
print("✅ Model Siap! Tekan 'M' untuk Mirror, 'Q' untuk Keluar.")

# ==================== MEDIAPIPE SETUP ====================
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

# ==================== FUNCTIONS ====================

def mediapipe_detection(image, model, is_mirrored):
    if is_mirrored:
        image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):
    """
    Gambar landmark POSE, TANGAN KIRI, TANGAN KANAN.
    FACE LANDMARK TIDAK DIGAMBAR — konsisten dengan pengambilan dataset.
    """
    # Pose
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10),   thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 44, 121),  thickness=1, circle_radius=1))
    # Tangan Kiri
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76),  thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # Tangan Kanan
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
    # FACE: TIDAK DIGAMBAR (refine_face_landmarks=False, tidak ada di results)


def extract_keypoints(results):
    """
    Ekstrak keypoints TANPA face landmark — identik dengan dataset collector.
    Total fitur: pose(33*4=132) + left_hand(21*3=63) + right_hand(21*3=63) = 258
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 4)

    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(21 * 3)

    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])  # shape: (258,)


def kalibrasi_focal_length(image, results, jarak_nyata_cm=60):
    """
    Hitung focal length kamera secara otomatis.
    User berdiri tepat di jarak_nyata_cm dari kamera saat kalibrasi.
    Referensi: lebar bahu (pose landmark 11=kiri, 12=kanan)
    """
    if results.pose_landmarks:
        h, w, _ = image.shape
        bl = results.pose_landmarks.landmark[11]
        br = results.pose_landmarks.landmark[12]

        x1, y1 = bl.x * w, bl.y * h
        x2, y2 = br.x * w, br.y * h
        w_pixel = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if w_pixel < 10:
            return None

        W_BAHU = 40  # lebar bahu rata-rata ~40 cm
        return (w_pixel * jarak_nyata_cm) / W_BAHU

    return None


def hitung_jarak_bahu(image, results, focal_length):
    """Estimasi jarak user ke kamera berdasarkan lebar bahu (cm)."""
    if results.pose_landmarks and focal_length:
        h, w, _ = image.shape
        bl = results.pose_landmarks.landmark[11]
        br = results.pose_landmarks.landmark[12]

        x1, y1 = bl.x * w, bl.y * h
        x2, y2 = br.x * w, br.y * h
        w_pixel = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if w_pixel == 0:
            return 0

        W_BAHU = 40
        return int((W_BAHU * focal_length) / w_pixel)

    return 0


def tampilkan_info_jarak(image, jarak_cm, y_offset=30):
    """
    Tampilkan indikator jarak dengan:
    - Angka jarak (cm)
    - Bar horizontal berwarna
    - Pesan panduan
    """
    JARAK_MIN = 50
    JARAK_MAX = 100
    BAR_X     = 20
    BAR_Y     = y_offset + 45
    BAR_W     = 260
    BAR_H     = 14

    # Tentukan warna & pesan
    if jarak_cm == 0:
        warna = (128, 128, 128)
        pesan = "Bahu tidak terdeteksi"
        persen = 0
    elif jarak_cm < JARAK_MIN:
        warna = (0, 60, 255)
        pesan = "<< MUNDUR!"
        persen = max(0, jarak_cm / JARAK_MIN)
    elif jarak_cm > JARAK_MAX:
        warna = (0, 60, 255)
        pesan = "MAJU! >>"
        persen = 1.0
    else:
        warna = (0, 210, 0)
        pesan = "POSISI OK"
        persen = (jarak_cm - JARAK_MIN) / (JARAK_MAX - JARAK_MIN)

    # Teks jarak
    cv2.putText(image, f"Jarak: {jarak_cm} cm", (BAR_X, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, warna, 2)
    cv2.putText(image, pesan, (BAR_X + 155, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna, 1)

    # Zona aman (hijau muda tipis)
    cv2.rectangle(image, (BAR_X, BAR_Y), (BAR_X + BAR_W, BAR_Y + BAR_H),
                  (50, 80, 50), -1)
    # Outline bar
    cv2.rectangle(image, (BAR_X, BAR_Y), (BAR_X + BAR_W, BAR_Y + BAR_H),
                  (180, 180, 180), 1)
    # Isi bar
    fill_w = int(BAR_W * min(persen, 1.0))
    if fill_w > 0:
        cv2.rectangle(image, (BAR_X, BAR_Y),
                      (BAR_X + fill_w, BAR_Y + BAR_H), warna, -1)

    # Label zona
    cv2.putText(image, f"{JARAK_MIN}cm", (BAR_X, BAR_Y + BAR_H + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    cv2.putText(image, f"{JARAK_MAX}cm",
                (BAR_X + BAR_W - 38, BAR_Y + BAR_H + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        color = colors[num] if num < len(colors) else (255, 255, 255)
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        text = f"{actions[num]}: {prob * 100:.2f}%"
        cv2.putText(output_frame, text, (5, 85 + num * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return output_frame


# ==================== REALTIME LOGIC ====================
sequence    = []
sentence    = []
predictions = []
mirror_mode = True

STABILITY_FRAMES = 12
JARAK_KALIBRASI  = 60   # cm — jarak saat kalibrasi bahu
focal_length     = None

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
# ✅ DIPERBAIKI: resolusi sama dengan pengambilan dataset (1280x720)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ✅ DIPERBAIKI: refine_face_landmarks=False & enable_segmentation=False
#    identik dengan dataset collector → tidak ada face mesh di-proses
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=False,   # ← face mesh dimatikan
        enable_segmentation=False) as holistic:

    # -----------------------------------------------
    # PHASE 0: KALIBRASI FOCAL LENGTH
    # -----------------------------------------------
    print(f"\n[KALIBRASI] Berdiri tepat {JARAK_KALIBRASI}cm dari kamera, lalu tekan SPASI.")
    print("Tekan [M] untuk toggle mirror, [Q] untuk keluar.")

    while focal_length is None:
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic, mirror_mode)
        draw_styled_landmarks(image, results)

        h, w = image.shape[:2]

        # Overlay instruksi kalibrasi
        cv2.rectangle(image, (0, 0), (w, 105), (0, 0, 0), -1)
        cv2.putText(image, "FASE KALIBRASI", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(image, f"Berdiri tepat {JARAK_KALIBRASI}cm dari kamera",
                    (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(image, "SPASI = kalibrasi  |  M = mirror  |  Q = keluar",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        bahu_ok    = results.pose_landmarks is not None
        st_bahu    = "Bahu terdeteksi OK" if bahu_ok else "Bahu belum terdeteksi..."
        warna_bahu = (0, 220, 0) if bahu_ok else (0, 0, 220)
        cv2.putText(image, st_bahu, (20, h - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_bahu, 2)

        mode_text = "M: ON" if mirror_mode else "M: OFF"
        cv2.putText(image, mode_text, (w - 90, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('SIBI Test', image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if bahu_ok:
                fl = kalibrasi_focal_length(image, results, JARAK_KALIBRASI)
                if fl:
                    focal_length = fl
                    print(f"Kalibrasi berhasil! Focal length: {focal_length:.1f}px")
                else:
                    print("Kalibrasi gagal, lebar bahu terlalu kecil. Coba lagi.")
            else:
                print("Bahu tidak terdeteksi.")
        elif key == ord('m'):
            mirror_mode = not mirror_mode
            sequence = []
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # -----------------------------------------------
    # PHASE 1: DETEKSI REALTIME
    # -----------------------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic, mirror_mode)
        draw_styled_landmarks(image, results)

        # --- HITUNG JARAK BAHU ---
        jarak_cm = hitung_jarak_bahu(image, results, focal_length)

        # --- CEK APAKAH ADA TANGAN? ---
        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

        if hand_detected:
            keypoints = extract_keypoints(results)   # shape: (258,) ✅
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]
        else:
            # Reset jika tangan hilang agar tidak memprediksi sisa data lama
            sequence    = []
            predictions = []

        # Default display
        live_status = "Menunggu Tangan..." if not hand_detected else "Menganalisa..."
        live_prob   = 0.0
        text_color  = (0, 0, 255)

        if len(sequence) == sequence_length:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]

            best_class_index = np.argmax(res)
            confidence       = res[best_class_index]

            predictions.append(best_class_index)
            recent_predictions = predictions[-STABILITY_FRAMES:]

            live_status = ACTIONS[best_class_index]
            live_prob   = confidence

            # --- STABILISASI (ANTI GONTA-GANTI) ---
            if (len(recent_predictions) == STABILITY_FRAMES
                    and np.unique(recent_predictions)[0] == best_class_index):

                if confidence > THRESHOLD:
                    if len(sentence) > 0:
                        if ACTIONS[best_class_index] != sentence[-1]:
                            sentence.append(ACTIONS[best_class_index])
                    else:
                        sentence.append(ACTIONS[best_class_index])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            image = prob_viz(res, ACTIONS, image, COLORS)

            if confidence > THRESHOLD:
                text_color = (0, 255, 0)

        # --- VISUALISASI UI ---
        h, w = image.shape[:2]

        # 1. Kotak Kalimat
        cv2.rectangle(image, (0, 0), (w, 40), (245, 117, 16), -1)
        cv2.putText(image, 'Kalimat: ' + ' '.join(sentence), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # 2. Status Real-time
        if hand_detected:
            display_text = f"Deteksi: {live_status} ({live_prob * 100:.1f}%)"
        else:
            display_text = "TANGAN TIDAK TERDETEKSI"

        cv2.putText(image, display_text, (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # 3. Mirror Info
        mode_text = "M: ON" if mirror_mode else "M: OFF"
        cv2.putText(image, mode_text, (w - 90, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 4. Info Jarak
        tampilkan_info_jarak(image, jarak_cm, y_offset=530)

        cv2.imshow('SIBI Test', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('m'):
            mirror_mode = not mirror_mode
            sequence = []   # Reset buffer saat mode berubah

cap.release()
cv2.destroyAllWindows()