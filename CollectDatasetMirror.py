import cv2
import numpy as np
import os
import mediapipe as mp
import time

# IMPOR DARI CONFIG
from Config import DATA_PATH, actions, no_sequences, sequence_length

# ==========================================
# 1. SETUP MEDIAPIPE & FUNGSI BANTU
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    # --- FITUR MIRRORING ---
    # Tangan Kiri (Fisik) -> Tangan Kanan (Layar/Data)
    image = cv2.flip(image, 1) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    # Wajah
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    # Tubuh
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    # Tangan Kiri (Warna Ungu)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    # Tangan Kanan (Warna Orange - TARGET UTAMA HASIL MIRROR)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# ==========================================
# 2. LOGIKA UTAMA
# ==========================================
def main():
    # Setup Kamera (30 FPS Standar)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Pilih Kata
    for idx, name in enumerate(actions):
        print(f"[{idx}] {name}")
    try:
        choice = int(input("Pilih Nomor Kata: "))
        target_action = actions[choice]
    except:
        print("❌ Input salah!")
        return

    # Cek Resume (Start Folder)
    start_folder = 0
    action_path = os.path.join(DATA_PATH, target_action)
    if os.path.exists(action_path):
        existing = [int(f) for f in os.listdir(action_path) if f.isdigit()]
        if existing:
            last_seq = max(existing)
            # Cek kelengkapan file di folder terakhir
            if len(os.listdir(os.path.join(action_path, str(last_seq)))) < sequence_length:
                start_folder = last_seq # Ulangi folder yg rusak
            else:
                start_folder = last_seq + 1 # Lanjut folder baru

    print(f"🚀 Bersiap merekam '{target_action}'. Target: {no_sequences} video.")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # --- PHASE 1: STANDBY (CUKUP SEKALI DI AWAL) ---
        print("Tunggu Jendela Kamera terbuka...")
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            # UI Awal
            cv2.rectangle(image, (0,0), (640, 80), (0,0,0), -1)
            cv2.putText(image, f"TARGET: {target_action} (Total: {no_sequences})", (20,40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            cv2.putText(image, "TEKAN [SPASI] UNTUK MULAI (Otomatis Lanjut)", (20,70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            # Cek Tangan (Indikator Hijau/Merah)
            if results.right_hand_landmarks:
                cv2.putText(image, "TANGAN TERDETEKSI (OK)", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.putText(image, "POSISIKAN TANGAN...", (20,450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow('OpenCV Feed', image)
            
            # Tunggu Tombol SPASI
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # --- PHASE 2: LOOPING OTOMATIS (MEREKAM TERUS) ---
        for sequence in range(start_folder, no_sequences):
            
            # 1. Jeda Istirahat (Break) Antar Video (2 Detik)
            # Agar tidak langsung merekam saat tangan belum siap
            start_break = time.time()
            break_duration = 2.0 
            
            while True:
                elapsed = time.time() - start_break
                if elapsed > break_duration: 
                    break # Lanjut ke perekaman
                
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                
                # Hitung Mundur Visual
                countdown = int(np.ceil(break_duration - elapsed))
                cv2.putText(image, f'MULAI DALAM {countdown}', (180, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                cv2.putText(image, f'Video ke-{sequence}', (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(1)

            # 2. Proses Merekam (30 Frame)
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret: break

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                
                # UI Recording
                cv2.putText(image, f'MEREKAM [{target_action}]', (20,40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(image, f'Video: {sequence}/{no_sequences-1} | Frame: {frame_num}', (20,70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

                cv2.imshow('OpenCV Feed', image)
                
                # Simpan Data
                keypoints = extract_keypoints(results)
                
                save_path = os.path.join(DATA_PATH, target_action, str(sequence))
                os.makedirs(save_path, exist_ok=True)
                
                npy_path = os.path.join(save_path, str(frame_num))
                np.save(npy_path, keypoints)

                # Tombol Q Darurat
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\n🛑 Berhenti Paksa.")
                    return

    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Perekaman Selesai secara Otomatis!")

if __name__ == "__main__":
    main()