# ==================== IMPORT ====================
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os

# ==================== KONFIGURASI PENTING ====================
ACTIONS = np.array(['Saya', 'Buah', 'Kuat', 'Agar', 'Makan', 'Sayur', 'Ibu', 'An'])

GRAMMAR_RULES = {
    "SUBJECT": ["Saya", "Ibu"],
    "PREDICATE": ["Makan"],
    "OBJECT": ["Buah", "Sayur"],
    "HUBUNG": ["Agar"],
    "KETERANGAN": ["Kuat"],
    "SUFFIX": ["An"] 
}

THRESHOLD = 0.8
SEQUENCE_LENGTH = 30
STABILITY_FRAMES = 10 

MODEL_PATH = 'final_lstm_action.h5'
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model '{MODEL_PATH}' tidak ditemukan!")
    exit()

print("🔄 Memuat Model AI...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model siap! Mendeteksi {len(ACTIONS)} kata.")

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==================== FUNGSI LOGIKA (GRAMMAR SPOK) ====================
def get_category(word):
    # 1. Cek Exact Match (Pencocokan Tepat)
    for cat, words in GRAMMAR_RULES.items():
        if word in words:
            return cat
            
    # 2. CEK CERDAS: Jika kata hasil suffix (berakhiran 'an')
    # Contoh: "Sayuran" tidak ada di list, tapi "Sayur" ada.
    word_lower = word.lower()
    if word_lower.endswith("an"):
        root_word = word[:-2] # Hapus 2 huruf terakhir ('an')
        # Cek apakah root_word (misal: Sayur) ada di kategori
        for cat, words in GRAMMAR_RULES.items():
            if root_word in words:
                # Sayur(Object) + an = Sayuran (Tetap dianggap Object untuk transisi)
                return cat
                
    return "UNKNOWN"

def merge_suffix(prev_word, suffix):
    return prev_word + suffix.lower()

def process_word_logic(sentence, word):
    cat = get_category(word)
    
    # --- LOGIKA SUFFIX ---
    if cat == "SUFFIX":
        if sentence:
            sentence[-1] = merge_suffix(sentence[-1], word)
            return sentence, True, f"✅ Digabung: {sentence[-1]}"
        return sentence, False, f"❌ '{word}' tidak bisa di awal"

    # --- LOGIKA AWAL KALIMAT ---
    if not sentence:
        if cat == "SUBJECT":
            sentence.append(word)
            return sentence, True, f"✅ Subjek '{word}' diterima."
        return sentence, False, f"❌ Awal harus SUBJEK, bukan {cat}"

    # --- LOGIKA TRANSISI (SPOK) ---
    # Kita ambil kategori kata terakhir di kalimat
    last_word = sentence[-1]
    last_cat = get_category(last_word) 
    
    # Debugging (Opsional: lihat di terminal)
    print(f"DEBUG: '{last_word}' ({last_cat}) -> '{word}' ({cat})")

    valid_transitions = {
        "SUBJECT": ["PREDICATE", "KETERANGAN"], 
        "PREDICATE": ["OBJECT", "HUBUNG", "KETERANGAN"], 
        # Object boleh lanjut ke Hubung (Sayur -> Agar)
        "OBJECT": ["HUBUNG", "KETERANGAN", "PREDICATE"], 
        "HUBUNG": ["KETERANGAN", "SUBJECT", "OBJECT", "PREDICATE"], 
        "KETERANGAN": ["HUBUNG"] 
    }

    if last_cat in valid_transitions:
        if cat in valid_transitions[last_cat]:
            sentence.append(word)
            return sentence, True, f"✅ {cat} '{word}' diterima."
        else:
            return sentence, False, f"❌ Habis {last_cat} tidak boleh {cat}."
    
    return sentence, False, "❌ Susunan tidak valid."

# ==================== FUNGSI DETEKSI ====================
def mediapipe_detection(image, model, is_mirrored):
    if is_mirrored:
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

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# ==================== MAIN PROGRAM ====================
sequence = []
sentence = []
predictions = []
mirror_mode = True

feedback_text = "Siap Deteksi..."
feedback_color = (255,255,255)
live_text = ""

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image, results = mediapipe_detection(frame, holistic, mirror_mode)
        draw_styled_landmarks(image, results)

        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

        if hand_detected:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_LENGTH:] 
        else:
            if len(sequence) > 0:
                print("⚠️ Tangan hilang - Reset buffer")
            sequence = []
            predictions = []

        if len(sequence) == SEQUENCE_LENGTH:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            best_idx = np.argmax(res)
            confidence = res[best_idx]
            predictions.append(best_idx)
            
            recent = predictions[-STABILITY_FRAMES:]

            if len(recent) == STABILITY_FRAMES:
                if len(np.unique(recent)) == 1 and recent[0] == best_idx:
                    if confidence > THRESHOLD:
                        detected_word = ACTIONS[best_idx]
                        
                        should_process = False
                        if len(sentence) == 0:
                            should_process = True
                        elif ACTIONS[best_idx] != sentence[-1]:
                             # Logika Debounce: Cegah spam, TAPI izinkan jika kata sebelumnya diubah oleh Suffix
                             # Misal: sentence[-1] = "Sayuran", detected = "Agar" -> Boleh
                             # Misal: sentence[-1] = "Sayuran", detected = "Sayur" -> Tahan (Kecuali user memang mau ulang)
                             should_process = True
                        elif get_category(ACTIONS[best_idx]) == "SUFFIX":
                            should_process = True

                        if should_process:
                            old_len = len(sentence)
                            old_word = sentence[-1] if sentence else ""
                            
                            sentence, success, msg = process_word_logic(sentence, detected_word)
                            
                            # Update feedback jika ada perubahan
                            if len(sentence) != old_len or (sentence and sentence[-1] != old_word):
                                feedback_text = msg
                                feedback_color = (0,255,0) if success else (0,0,255)
                                if success:
                                    sequence = [] 
                                    predictions = []
                            elif not success:
                                # Jika gagal validasi, tampilkan error tapi jangan reset sequence total
                                feedback_text = msg
                                feedback_color = (0,0,255)

            live_text = f"{ACTIONS[best_idx]} ({confidence*100:.0f}%)"
        else:
            live_text = "Menunggu..." if hand_detected else "Tangan tidak terdeteksi"

        # UI
        cv2.rectangle(image, (0,0), (640,110), (245,117,16), -1)
        cv2.putText(image, 'Kalimat: ' + ' '.join(sentence), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(image, 'Deteksi: ' + live_text, (10,75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(image, feedback_text, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, feedback_color, 2)
        cv2.putText(image, f"Mirror: {'ON' if mirror_mode else 'OFF'}", (480,460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        cv2.imshow("SIBI Smart Grammar (Final)", image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('c'):
            sentence = []
            feedback_text = "Kalimat Direset."
            feedback_color = (255,255,255)
        if key == ord('m'):
            mirror_mode = not mirror_mode
            sequence = []

cap.release()
cv2.destroyAllWindows()