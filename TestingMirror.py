# =============================================================================
# TestingMirror.py
# =============================================================================
# Tujuan:
#   Menguji dan memverifikasi perilaku mode mirror (flip horizontal) pada kamera
#   sebelum proses pengumpulan dataset. File ini membantu memastikan bahwa
#   mapping tangan kiri/kanan pada MediaPipe konsisten dengan gerakan fisik
#   pengguna ketika mode mirror diaktifkan.
#
#   Masalah yang diselesaikan:
#     Secara default, kamera tidak di-mirror sehingga tangan kanan fisik
#     terdeteksi sebagai "left_hand_landmarks" oleh MediaPipe (dan sebaliknya).
#     Dengan flip horizontal (mirror), tampilan jadi seperti cermin dan
#     mapping tangan menjadi lebih intuitif.
#
#   Visualisasi warna:
#     - MERAH  : right_hand_landmarks MediaPipe (= tangan kiri fisik saat mirror)
#     - BIRU   : left_hand_landmarks MediaPipe  (= tangan kanan fisik saat mirror)
#     - HIJAU  : sisa skeleton tubuh
#
#   Output terminal:
#     Koordinat pergelangan tangan (wrist X,Y) dicetak setiap frame saat
#     tangan terdeteksi, untuk verifikasi arah gerakan.
#
# Cara Penggunaan:
#   python TestingMirror.py
#   Angkat tangan kiri/kanan lalu perhatikan warna dan output terminal.
#   Tekan 'q' untuk keluar.
#
# Urutan Eksekusi dalam Proyek:
#   1. TestingCameraHolistic.py -> Uji kamera & landmark
#   2. TestingMirror.py         -> Uji mapping tangan mirror  <-- FILE INI
#   3. TestingDataCollection.py -> Uji ekstraksi keypoints
# =============================================================================

import cv2
import mediapipe as mp
import sys

# ==========================================
# SETUP & KONFIGURASI
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Definisi Warna (BGR)
COLOR_BODY_DEFAULT = (0, 255, 0)   # Hijau (Badan Tengah)
COLOR_LEFT_SIDE = (255, 0, 0)      # Biru (Sisi Kiri MP / Tangan Kanan Fisik)
COLOR_RIGHT_SIDE = (0, 0, 255)     # Merah (Sisi Kanan MP / Tangan Kiri Fisik)

def main():
    cap = cv2.VideoCapture(0)

    # Setup Model
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5) as holistic:
        
        print("\n=== MULAI TEST MIRRORING & COLOR ===")
        print("Pastikan terminal ini terlihat untuk cek output koordinat.")
        print("Tekan 'q' di jendela kamera untuk keluar.\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # 1. MIRROR FRAME (Balik Horizontal)
            frame = cv2.flip(frame, 1)

            # 2. PROSES MEDIAPIPE
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            
            # 3. KEMBALIKAN KE BGR UNTUK GAMBAR
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # ==========================================================
            # A. GAMBAR POSE (BADAN) DENGAN WARNA TERPISAH
            # ==========================================================
            if results.pose_landmarks:
                # -- 1. Gambar Koneksi Lengan KANAN (Bahu->Siku->Pergelangan) --
                # Di MediaPipe Pose: 12=Right Shoulder, 14=Right Elbow, 16=Right Wrist
                # Karena Mirroring: Ini akan aktif saat Anda pakai Tangan Kiri Fisik
                mp_drawing.draw_landmarks(
                    image_bgr, results.pose_landmarks,
                    connections=[(12, 14), (14, 16)], 
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_RIGHT_SIDE, thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_RIGHT_SIDE, thickness=3)
                )

                # -- 2. Gambar Koneksi Lengan KIRI --
                # 11=Left Shoulder, 13=Left Elbow, 15=Left Wrist
                mp_drawing.draw_landmarks(
                    image_bgr, results.pose_landmarks,
                    connections=[(11, 13), (13, 15)],
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_LEFT_SIDE, thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_LEFT_SIDE, thickness=3)
                )

                # -- 3. Gambar Sisa Tubuh (Hijau) --
                # Kita gambar titik-titik lainnya agar tetap terlihat posturnya
                mp_drawing.draw_landmarks(
                    image_bgr, results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS, # Gambar semua dulu (sebagai base)
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_BODY_DEFAULT, thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=COLOR_BODY_DEFAULT, thickness=1) 
                )
                # *Note: Garis hijau akan tertimpa garis merah/biru yang kita gambar sebelumnya/sesudahnya 
                # tergantung urutan, tapi intinya lengan akan berwarna khusus.

            # ==========================================================
            # B. GAMBAR TANGAN & OUTPUT TERMINAL
            # ==========================================================
            
            # Reset status print
            status_text = ""

            # --- CEK TANGAN KANAN (Right Hand Landmarks) ---
            # (Ini yang aktif saat Tangan KIRI Fisik Anda diangkat karena Mirror)
            if results.right_hand_landmarks:
                # Gambar Tangan MERAH
                mp_drawing.draw_landmarks(
                    image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLOR_RIGHT_SIDE, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=COLOR_RIGHT_SIDE, thickness=2, circle_radius=2)
                )
                
                # Ambil koordinat Wrist (Pergelangan) index 0
                wrist = results.right_hand_landmarks.landmark[0]
                info = f"🔴 RIGHT HAND DETECTED (Fisik Kiri) | X: {wrist.x:.2f}, Y: {wrist.y:.2f}"
                print(info) # Output ke Terminal
                status_text = "DETECTED: RIGHT (RED)"

            # --- CEK TANGAN KIRI (Left Hand Landmarks) ---
            if results.left_hand_landmarks:
                # Gambar Tangan BIRU
                mp_drawing.draw_landmarks(
                    image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLOR_LEFT_SIDE, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=COLOR_LEFT_SIDE, thickness=2, circle_radius=2)
                )
                
                wrist = results.left_hand_landmarks.landmark[0]
                info = f"🔵 LEFT HAND DETECTED (Fisik Kanan)| X: {wrist.x:.2f}, Y: {wrist.y:.2f}"
                print(info) # Output ke Terminal
                status_text = "DETECTED: LEFT (BLUE)"

            # --- UI INFO DI LAYAR ---
            cv2.rectangle(image_bgr, (0,0), (640, 80), (0,0,0), -1)
            cv2.putText(image_bgr, 'MIRROR MODE AKTIF', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if status_text:
                color = COLOR_RIGHT_SIDE if "RIGHT" in status_text else COLOR_LEFT_SIDE
                cv2.putText(image_bgr, status_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Mirror Test v2', image_bgr)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()