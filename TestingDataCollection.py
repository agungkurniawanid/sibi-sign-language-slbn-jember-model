# =============================================================================
# TestingDataCollection.py
# =============================================================================
# Tujuan:
#   Menguji pipeline ekstraksi keypoints sebelum proses pengumpulan dataset
#   dilakukan. File ini memverifikasi bahwa fungsi mediapipe_detection(),
#   draw_styled_landmarks(), dan extract_keypoints() bekerja dengan benar
#   dan menghasilkan array keypoints dengan panjang yang tepat (1662 nilai).
#
#   Rincian panjang array keypoints (1662 total):
#     - Pose  : 33 titik x 4 dimensi (x,y,z,visibility) = 132
#     - Face  : 468 titik x 3 dimensi (x,y,z)           = 1404
#     - Tangan Kiri  : 21 titik x 3 dimensi             = 63
#     - Tangan Kanan : 21 titik x 3 dimensi             = 63
#
# Cara Penggunaan:
#   python TestingDataCollection.py
#   Periksa output terminal: panjang array harus konsisten 1662.
#   Tekan 'q' untuk keluar.
#
# Urutan Eksekusi dalam Proyek:
#   1. TestingCameraHolistic.py  -> Uji kamera & landmark visual
#   2. TestingDataCollection.py  -> Uji ekstraksi keypoints  <-- FILE INI
#   3. SetupFolders.py           -> Buat folder dataset
#   4. (Collect Data)            -> Kumpulkan dataset
# =============================================================================

import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi Model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_styled_landmarks(image, results):
    # Menggambar landmark (sama seperti sebelumnya)
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
    """
    Fungsi ini mengubah hasil deteksi menjadi array numpy 1 dimensi.
    Jika tidak terdeteksi, diganti dengan array nol (np.zeros).
    """
    # 1. Pose (Tubuh): 33 titik x 4 dimensi (x,y,z,visibility) = 132 data
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose = np.zeros(33*4)

    # 2. Face (Wajah): 468 titik x 3 dimensi (x,y,z) = 1404 data
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468*3)

    # 3. Left Hand (Tangan Kiri): 21 titik x 3 dimensi (x,y,z) = 63 data
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21*3)

    # 4. Right Hand (Tangan Kanan): 21 titik x 3 dimensi (x,y,z) = 63 data
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21*3)

    # Gabungkan semua menjadi satu baris panjang
    return np.concatenate([pose, face, lh, rh])

def main():
    cap = cv2.VideoCapture(0)
    
    # Set model mediapipe
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 1. Deteksi
            image, results = mediapipe_detection(frame, holistic)
            
            # 2. Gambar Visual
            draw_styled_landmarks(image, results)
            
            # 3. Ekstraksi Data (Bagian Baru)
            keypoints = extract_keypoints(results)
            
            # Cek panjang data di terminal (harus konsisten 1662)
            # 132 (pose) + 1404 (face) + 63 (lh) + 63 (rh) = 1662
            print(f"Data terekstrak. Panjang Array: {len(keypoints)}")

            cv2.imshow('Data Collection Test', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
                
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()