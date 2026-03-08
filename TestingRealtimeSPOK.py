# =============================================================================
# TestingRealtimeSPOK.py
# =============================================================================
# Tujuan:
#   Menjalankan deteksi bahasa isyarat SIBI secara realtime dengan validasi
#   grammar SPOK (Subjek-Predikat-Objek-Keterangan). Setiap kata yang terdeteksi
#   oleh model AI akan diperiksa apakah urutannya sesuai kaidah grammar bahasa
#   Indonesia sebelum ditambahkan ke kalimat.
#
#   Fitur utama:
#     - Semua fitur dari TestingRealtimeModel.py (deteksi realtime, stabilisasi, mirror)
#     - Validasi grammar SPOK: kata hanya diterima jika transisi antar kategori valid
#       (contoh: SUBJEK -> PREDIKAT -> OBJEK -> KETERANGAN)
#     - Feedback visual grammar real-time (hijau = valid, merah = tidak valid)
#     - Deteksi jarak pengguna ke kamera menggunakan landmark mata (dalam cm)
#     - Panduan jarak: MAJU / OK / MUNDUR ditampilkan di layar
#     - Logika suffix "-an": kata seperti "Makanan" dibentuk otomatis dari "Makan" + "An"
#
#   Kategori grammar yang didukung:
#     SUBJECT    : Saya, Ibu
#     PREDICATE  : Makan
#     OBJECT     : Buah, Sayur
#     HUBUNG     : Agar
#     KETERANGAN : Kuat
#     SUFFIX     : An (disambungkan ke kata sebelumnya)
#
# Cara Penggunaan:
#   Pastikan file 'model_best.keras' tersedia, lalu:
#   python TestingRealtimeSPOK.py
#   Kontrol: 'Q' = keluar, 'C' = reset kalimat, 'M' = toggle mirror
# =============================================================================

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import math
from Config import actions

# ==================== KONFIGURASI ====================
# Pastikan urutan ACTIONS sama persis dengan saat training!
ACTIONS = actions

# Nama Model
MODEL_PATH = 'model_best.keras' 

# Konfigurasi Deteksi
THRESHOLD = 0.85        # Ambang batas keyakinan
SEQUENCE_LENGTH = 30    # Jumlah frame per input
STABILITY_FRAMES = 12   # Harus stabil selama sekian frame

# Konfigurasi Jarak (cm)
MIN_DIST = 50
MAX_DIST = 90

# ==================== GRAMMAR RULES (SPOK STRICT) ====================
CATEGORY_MAP = {
    "SUBJECT":   ["Saya", "Ibu"],
    "PREDICATE": ["Makan"],
    "OBJECT":    ["Buah", "Sayur"],
    "HUBUNG":    ["Agar"],
    "KETERANGAN":["Kuat"],
    "SUFFIX":    ["An"]
}

VALID_TRANSITIONS = {
    "START":     ["SUBJECT"],                     
    "SUBJECT":   ["PREDICATE"],                   
    "PREDICATE": ["OBJECT", "KETERANGAN", "HUBUNG", "SUFFIX"], 
    "OBJECT":    ["HUBUNG", "KETERANGAN", "SUFFIX"],
    "SUFFIX":    ["HUBUNG", "KETERANGAN"],        
    "HUBUNG":    ["KETERANGAN", "PREDICATE"],     
    "KETERANGAN":[]                               
}

# ==================== FUNGSI LOAD MODEL ====================
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model '{MODEL_PATH}' tidak ditemukan!")
    exit()

print("🔄 Memuat Model AI...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model siap! Mendeteksi {len(ACTIONS)} kata.")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==================== LOGIKA GRAMMAR ====================
def get_category_of_word(word):
    for cat, words in CATEGORY_MAP.items():
        if word in words: return cat
    
    if word.lower().endswith("an"):
        root = word[:-2] 
        for cat, words in CATEGORY_MAP.items():
            if root in words: return cat 
            
    return "UNKNOWN"

def check_grammar_validity(current_sentence, new_word):
    new_cat = get_category_of_word(new_word)
    
    # 1. KONDISI AWAL (Kalimat Kosong)
    if not current_sentence:
        if new_cat in VALID_TRANSITIONS["START"]:
            return [new_word], True, f"✅ Awal Kalimat: {new_word} ({new_cat})"
        else:
            return [], False, f"❌ Awal harus SUBJEK, bukan {new_word}"

    # 2. CEK KATA TERAKHIR
    last_word = current_sentence[-1]
    
    if last_word == new_word:
        return current_sentence, False, "⚠️ Kata sudah ada (Duplikat)"

    last_cat = get_category_of_word(last_word)

    # 3. LOGIKA SUFFIX (-an)
    if new_cat == "SUFFIX":
        if last_cat in ["PREDICATE", "OBJECT"]:
            merged_word = last_word + "an" 
            current_sentence[-1] = merged_word
            return current_sentence, True, f"✅ Digabung: {merged_word}"
        else:
            return current_sentence, False, f"❌ '{last_word}' tidak bisa diberi akhiran -an"

    # 4. LOGIKA TRANSISI UMUM
    if last_cat in VALID_TRANSITIONS:
        allowed_next = VALID_TRANSITIONS[last_cat]
        if new_cat in allowed_next:
            current_sentence.append(new_word)
            return current_sentence, True, f"✅ {last_cat} -> {new_cat}: {new_word}"
        else:
            return current_sentence, False, f"❌ Salah Urutan: Habis {last_cat} tidak boleh {new_cat}"
            
    return current_sentence, False, "❌ Pola tidak dikenali."

# ==================== FUNGSI UTILITIES (MEDIPIPE & JARAK) ====================
def mediapipe_detection(image, model, mirror=False):
    if mirror:
        image = cv2.flip(image, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# 🔹 FUNGSI HITUNG JARAK
def hitung_jarak_user(image, results):
    if results.face_landmarks:
        image_height, image_width, _ = image.shape
        
        # Landmark Mata: 33 (Kiri) dan 263 (Kanan)
        point_left = results.face_landmarks.landmark[33]
        point_right = results.face_landmarks.landmark[263]

        x1, y1 = point_left.x * image_width, point_left.y * image_height
        x2, y2 = point_right.x * image_width, point_right.y * image_height

        w_pixel = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        W = 14   # Lebar wajah rata-rata (cm)
        f = 300  # Focal length
        
        if w_pixel == 0: return 0
        
        distance = (W * f) / w_pixel
        return int(distance)
    return 0

# ==================== MAIN LOOP ====================
sequence = []
sentence = []
predictions = []

mirror_mode = True 
feedback_text = "Menunggu Subjek..."
feedback_color = (255, 255, 255) 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Deteksi (Include Mirroring logic inside detection function call if needed, 
        # or flip explicitly here)
        if mirror_mode:
            frame = cv2.flip(frame, 1)

        image, results = mediapipe_detection(frame, holistic, mirror=False) 
        # Note: mirror=False disini karena kita sudah manual flip diatas ^
        
        # 2. Gambar Skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # 🔹 3. HITUNG JARAK
        jarak_cm = hitung_jarak_user(image, results)
        
        dist_color = (0, 0, 255) # Merah
        dist_msg = "?"
        
        if jarak_cm > 0:
            if jarak_cm < MIN_DIST:
                dist_msg = "MUNDUR"
            elif jarak_cm > MAX_DIST:
                dist_msg = "MAJU"
            else:
                dist_color = (0, 255, 0) # Hijau
                dist_msg = "OK"

        # 4. Logika Prediksi AI
        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:] 

            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                best_idx = np.argmax(res)
                confidence = res[best_idx]
                
                predictions.append(best_idx)
                predictions = predictions[-STABILITY_FRAMES:]

                if len(predictions) == STABILITY_FRAMES:
                    if np.unique(predictions)[0] == best_idx: 
                        if confidence > THRESHOLD:
                            detected_word = ACTIONS[best_idx]
                            
                            new_sentence, success, msg = check_grammar_validity(sentence.copy(), detected_word)
                            feedback_text = msg
                            
                            if success:
                                sentence = new_sentence
                                feedback_color = (0, 255, 0)
                                sequence = [] 
                                predictions = []
                                print(f"Valid Word Added: {detected_word}")
                            else:
                                feedback_color = (0, 0, 255)

        # ==================== UI DISPLAY ====================
        # Header Hitam (Atas)
        cv2.rectangle(image, (0,0), (640, 85), (20, 20, 20), -1)
        
        # Teks Kalimat
        text_sentence = ' '.join(sentence)
        cv2.putText(image, text_sentence, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Teks Feedback Grammar
        cv2.putText(image, feedback_text, (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, feedback_color, 2, cv2.LINE_AA)

        # 🔹 TAMPILKAN JARAK (POJOK KIRI BAWAH)
        # Background tipis untuk jarak agar terbaca
        cv2.putText(image, f"Jarak: {jarak_cm}cm ({dist_msg})", (10, 465), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, dist_color, 2, cv2.LINE_AA)

        # Footer Info (POJOK KANAN BAWAH)
        mirror_status = "ON" if mirror_mode else "OFF"
        cv2.putText(image, f"Mir: {mirror_status}(M) | Rst(C)", (400, 465), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('SIBI SPOK SYSTEM', image)

        # Kontrol Keyboard
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'): break
        if key == ord('c'): 
            sentence = []
            sequence = []
            feedback_text = "Kalimat Direset."
            feedback_color = (255, 255, 255)
        if key == ord('m'):
            mirror_mode = not mirror_mode
            sequence = [] 

    cap.release()
    cv2.destroyAllWindows()