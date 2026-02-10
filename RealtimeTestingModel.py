# ==================== IMPORT ====================
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
from Config import actions

# ==================== CONFIGURATION ====================
# PASTIKAN JUMLAH KATA SAMA DENGAN SAAT TRAINING!
# Jika training cuma ['Saya', 'Buah'], hapus 'Makan' dari sini.
ACTIONS = actions
COLORS = [(245,117,16), (117,245,16)] 

THRESHOLD = 0.8 # Akurasi minimal 80%

MODEL_PATH = 'final_lstm_action.h5' 
# MODEL_PATH = 'best_gesture_model.h5' 
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model '{MODEL_PATH}' tidak ditemukan!")
    exit()

print("🔄 Memuat Model AI...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Siap! Tekan 'M' untuk Mirror, 'Q' untuk Keluar.")

# ==================== MEDIAPIPE SETUP ====================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

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
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1)) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        color = colors[num] if num < len(colors) else (255, 255, 255)
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), color, -1)
        text = f"{actions[num]}: {prob*100:.2f}%"
        cv2.putText(output_frame, text, (5, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return output_frame

# ==================== REALTIME LOGIC ====================
sequence = []
sentence = []
predictions = []
mirror_mode = True 

# Setting Jendela Konsistensi (Semakin besar, semakin sulit gonta-ganti, tapi lebih lambat deteksinya)
STABILITY_FRAMES = 12 

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

        # --- LOGIKA 1: CEK APAKAH ADA TANGAN? ---
        # Karena kita pakai mirror, tangan kiri fisik jadi 'right_hand_landmarks'
        # Kita cek keduanya biar aman.
        hand_detected = results.left_hand_landmarks or results.right_hand_landmarks

        if hand_detected:
            # Jika ada tangan, lanjut proses AI
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:] # Ambil 30 frame terakhir
        else:
            # --- LOGIKA RESET: JIKA TANGAN HILANG ---
            # Jika tangan turun/hilang, reset history agar tidak memprediksi sisa data lama
            sequence = [] 
            predictions = [] 
            
        # Variabel display default
        live_status = "Menunggu Tangan..." if not hand_detected else "Menganalisa..."
        live_prob = 0.0
        text_color = (0, 0, 255) # Merah

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            
            best_class_index = np.argmax(res)
            confidence = res[best_class_index]
            
            predictions.append(best_class_index)
            
            # Ambil 12 prediksi terakhir untuk cek kestabilan
            recent_predictions = predictions[-STABILITY_FRAMES:]

            # Info Realtime
            live_status = ACTIONS[best_class_index]
            live_prob = confidence

            # --- LOGIKA 2: STABILISASI (ANTI GONTA-GANTI) ---
            # Syarat 1: Prediksi harus KONSISTEN selama 12 frame terakhir
            if len(recent_predictions) == STABILITY_FRAMES and np.unique(recent_predictions)[0] == best_class_index:
                
                # Syarat 2: Akurasi harus di atas Threshold (80%)
                if confidence > THRESHOLD: 
                    
                    # Syarat 3: Kata belum masuk (mencegah duplikat beruntun)
                    if len(sentence) > 0: 
                        if ACTIONS[best_class_index] != sentence[-1]:
                            sentence.append(ACTIONS[best_class_index])
                    else:
                        sentence.append(ACTIONS[best_class_index])

            if len(sentence) > 5: sentence = sentence[-5:]
            
            # Visualisasi Bar
            image = prob_viz(res, ACTIONS, image, COLORS)
            
            # Warna teks jadi hijau jika stabil & yakin
            if confidence > THRESHOLD:
                text_color = (0, 255, 0)

        # --- VISUALISASI UI ---
        
        # 1. Kotak Kalimat
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, 'Kalimat: ' + ' '.join(sentence), (10,30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # 2. Status Real-time
        if hand_detected:
            display_text = f"Deteksi: {live_status} ({live_prob*100:.1f}%)"
        else:
            display_text = "TANGAN TIDAK TERDETEKSI"
            
        cv2.putText(image, display_text, (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # 3. Mirror Info
        mode_text = "M: ON" if mirror_mode else "M: OFF"
        cv2.putText(image, mode_text, (550, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow('SIBI Test', image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('m'): 
            mirror_mode = not mirror_mode
            sequence = [] # Reset jika mode berubah

cap.release()
cv2.destroyAllWindows()