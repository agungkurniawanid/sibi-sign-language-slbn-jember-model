# =============================================================================
# TestingCameraHolistic.py
# =============================================================================
# Tujuan:
#   Menguji apakah kamera dan model MediaPipe Holistic berjalan dengan benar
#   di sistem ini. File ini menampilkan feed kamera secara realtime dengan
#   landmark wajah, tubuh (pose), tangan kiri, dan tangan kanan yang digambar
#   dengan warna berbeda untuk memudahkan identifikasi.
#
#   Landmark yang divisualisasikan:
#     - Wajah (Face Mesh Tesselation)
#     - Tubuh/Pose (33 titik)
#     - Tangan Kiri (21 titik, warna ungu)
#     - Tangan Kanan (21 titik, warna oranye)
#
# Cara Penggunaan:
#   Jalankan file ini untuk memastikan kamera dan MediaPipe bekerja normal:
#   python TestingCameraHolistic.py
#   Tekan 'q' pada jendela kamera untuk keluar.
#
# Urutan Eksekusi dalam Proyek:
#   1. SystemInformation.py      -> Cek environment
#   2. TestingCameraHolistic.py  -> Uji kamera & MediaPipe  <-- FILE INI
#   3. TestingMirror.py          -> Uji mode mirror
#   4. TestingDataCollection.py  -> Uji ekstraksi keypoints
# =============================================================================

import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi Model MediaPipe
mp_holistic = mp.solutions.holistic # Model Holistic
mp_drawing = mp.solutions.drawing_utils # Utilitas Menggambar

def mediapipe_detection(image, model):
    """
    Fungsi untuk memproses gambar dan melakukan prediksi landmark.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Konversi warna BGR ke RGB
    image.flags.writeable = False                  # Kunci gambar (optimasi memori)
    results = model.process(image)                 # Lakukan prediksi/deteksi
    image.flags.writeable = True                   # Buka kunci gambar
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # Kembalikan warna ke BGR
    return image, results

def draw_styled_landmarks(image, results):
    """
    Fungsi untuk menggambar landmark ke layar dengan style (warna) khusus.
    Menggunakan FACEMESH_TESSELATION untuk kompatibilitas MediaPipe v0.10.9.
    """
    # 1. Menggambar Wajah
    mp_drawing.draw_landmarks(
        image, 
        results.face_landmarks, 
        mp_holistic.FACEMESH_TESSELATION, 
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    ) 
    
    # 2. Menggambar Pose (Tubuh)
    mp_drawing.draw_landmarks(
        image, 
        results.pose_landmarks, 
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    ) 
    
    # 3. Menggambar Tangan Kiri (Left Hand)
    # Perhatikan warna ungu/pink ini untuk memastikan tangan kiri terdeteksi
    mp_drawing.draw_landmarks(
        image, 
        results.left_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    ) 
    
    # 4. Menggambar Tangan Kanan (Right Hand)
    mp_drawing.draw_landmarks(
        image, 
        results.right_hand_landmarks, 
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )

def main():
    # Mengakses Webcam (Index 0 biasanya webcam default)
    cap = cv2.VideoCapture(0)
    
    # Set model mediapipe
    # min_detection_confidence & tracking diset 0.5 (bisa dinaikkan jika kurang akurat)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        print("Kamera dimulai. Tekan 'q' di keyboard untuk keluar.")
        
        while cap.isOpened():
            # Baca feed dari kamera
            ret, frame = cap.read()
            
            if not ret:
                print("Gagal mengambil frame dari kamera.")
                break

            # Lakukan deteksi
            image, results = mediapipe_detection(frame, holistic)
            
            # (Opsional) Print results untuk melihat data raw landmark di terminal
            # print(results) 
            
            # Gambar landmark di layar
            draw_styled_landmarks(image, results)

            # Tampilkan ke layar
            cv2.imshow('OpenCV Feed - Tekan q untuk keluar', image)

            # Break loop jika tombol 'q' ditekan
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        # Bersihkan resource
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()