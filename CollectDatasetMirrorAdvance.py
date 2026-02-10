import cv2
import numpy as np
import os
import mediapipe as mp
import time
import math  # PERLU IMPORT INI

# IMPOR DARI CONFIG
from Config import DATA_PATH, actions, sequence_length

# ==========================================
# 1. SETUP MEDIAPIPE & FUNGSI BANTU
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    # Mirroring: Flip Horizontal
    image = cv2.flip(image, 1) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Gambar landmark seperti biasa
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
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

# === FUNGSI BARU: HITUNG JARAK (ESTIMASI) ===
# === FUNGSI BARU: HITUNG JARAK (SUDAH DIKALIBRASI) ===
def hitung_jarak_user(image, results):
    if results.face_landmarks:
        # Ambil lebar gambar
        image_height, image_width, _ = image.shape
        
        # Landmark 33: Ujung mata kiri (Outer)
        # Landmark 263: Ujung mata kanan (Outer)
        point_left = results.face_landmarks.landmark[33]
        point_right = results.face_landmarks.landmark[263]

        # Konversi ke pixel
        x1, y1 = point_left.x * image_width, point_left.y * image_height
        x2, y2 = point_right.x * image_width, point_right.y * image_height

        # Hitung jarak Euclidean antar mata dalam pixel
        w_pixel = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # RUMUS: Distance = (Real_Width * Focal_Length) / Apparent_Width_Pixel
        W = 14   # cm (lebar wajah rata-rata)
        
        # --- PERBAIKAN DISINI ---
        # Nilai f diturunkan dari 640 ke 300 agar sesuai dengan kamera Anda
        f = 300  
        
        if w_pixel == 0: return 0
        
        distance = (W * f) / w_pixel
        return int(distance)
    return 0

# ==========================================
# 2. LOGIKA UTAMA
# ==========================================
def main():
    JUMLAH_TAMBAHAN = 100  
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Pilih Kata
    print("\n=== MODE MENAMBAH DATASET + JARAK ===")
    for idx, name in enumerate(actions):
        print(f"[{idx}] {name}")
    try:
        choice = int(input("Pilih Nomor Kata: "))
        target_action = actions[choice]
    except:
        print("❌ Input salah!")
        return

    # Cek Folder
    start_folder = 0
    action_path = os.path.join(DATA_PATH, target_action)
    if os.path.exists(action_path):
        existing = [int(f) for f in os.listdir(action_path) if f.isdigit()]
        if existing:
            last_seq = max(existing)
            if len(os.listdir(os.path.join(action_path, str(last_seq)))) < sequence_length:
                start_folder = last_seq 
            else:
                start_folder = last_seq + 1 
    
    end_folder = start_folder + JUMLAH_TAMBAHAN
    
    print(f"🚀 Target Folder: {start_folder} sampai {end_folder - 1}")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # --- PHASE 1: STANDBY ---
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            # === HITUNG JARAK ===
            jarak_cm = hitung_jarak_user(image, results)
            
            # Tentukan Warna Jarak (Hijau jika ideal 50-80cm, Merah jika terlalu dekat/jauh)
            color_dist = (0, 255, 0) if 50 <= jarak_cm <= 80 else (0, 0, 255)
            
            # UI Info
            cv2.rectangle(image, (0,0), (640, 85), (0,0,0), -1)
            cv2.putText(image, f"ACTION: {target_action}", (20,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            # TAMPILKAN JARAK
            cv2.putText(image, f"JARAK: {jarak_cm} cm", (400, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_dist, 2)
            
            if jarak_cm < 50 and jarak_cm > 0:
                cv2.putText(image, "MUNDUR DIKIT", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            elif jarak_cm > 80:
                cv2.putText(image, "MAJU DIKIT", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            else:
                cv2.putText(image, "POSISI OK", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            cv2.putText(image, "TEKAN [SPASI] UNTUK MULAI", (100,450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(1) & 0xFF == ord(' '): break
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                cap.release()
                cv2.destroyAllWindows()
                return

        # --- PHASE 2: LOOPING PEREKAMAN ---
        for sequence in range(start_folder, end_folder):
            start_break = time.time()
            break_duration = 2.0 
            
            while True:
                elapsed = time.time() - start_break
                if elapsed > break_duration: break 
                
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                
                # Hitung Jarak saat Jeda
                jarak_cm = hitung_jarak_user(image, results)

                countdown = int(np.ceil(break_duration - elapsed))
                cv2.putText(image, f'MULAI: {countdown}', (220, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                
                # Tampilkan Jarak Kecil di Pojok
                cv2.putText(image, f"Jarak: {jarak_cm}cm", (450, 450), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(1)

            # Proses Merekam
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret: break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                
                # Hitung Jarak saat Merekam (opsional, biar user tau kalau goyang)
                jarak_cm = hitung_jarak_user(image, results)
                
                cv2.putText(image, f'REC: {target_action} | {sequence}', (20,40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(image, f"Dist: {jarak_cm}cm", (500, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

                cv2.imshow('OpenCV Feed', image)
                
                keypoints = extract_keypoints(results)
                save_path = os.path.join(DATA_PATH, target_action, str(sequence))
                os.makedirs(save_path, exist_ok=True)
                npy_path = os.path.join(save_path, str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\n🛑 Berhenti Paksa.")
                    return

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ Selesai! Data tersimpan di folder {start_folder} sampai {end_folder-1}.")

if __name__ == "__main__":
    main()