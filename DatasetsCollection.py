import cv2
import numpy as np
import os
import mediapipe as mp
import time
import math

# IMPORT DARI CONFIG
from Config import DATA_PATH, actions, sequence_length

# ==========================================
# 1. SETUP MEDIAPIPE & FUNGSI BANTU
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils

# State global mirror
mirror_on = True


def mediapipe_detection(image, model):
    """Deteksi tanpa flip otomatis — mirror dikendalikan dari luar."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def apply_mirror(image):
    """Flip frame secara horizontal (mirror)."""
    return cv2.flip(image, 1)


def draw_styled_landmarks(image, results):
    """
    Gambar landmark POSE, TANGAN KIRI, TANGAN KANAN.
    FACE LANDMARK TIDAK DIGAMBAR SAMA SEKALI.
    (Holistic diinisialisasi dengan refine_face_landmarks=False
     sehingga face landmark memang tidak ada di results.)
    """
    # Pose
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10),   thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121),  thickness=2, circle_radius=2))
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
    # FACE: TIDAK DIGAMBAR (results.face_landmarks tidak digunakan)


def extract_keypoints(results):
    """
    Ekstrak keypoints TANPA face landmark.
    Total fitur: pose(132) + left_hand(63) + right_hand(63) = 258
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


# ==========================================
# 2. FUNGSI JARAK BERBASIS BAHU
# ==========================================

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
        # Posisi relatif dalam zona aman (50-100cm)
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


def tampilkan_mirror_status(image, mirror_aktif):
    """Tampilkan status mirror di pojok kanan bawah."""
    h, w, _ = image.shape
    label = "[M] Mirror: ON" if mirror_aktif else "[M] Mirror: OFF"
    warna = (0, 255, 255) if mirror_aktif else (100, 100, 100)
    cv2.putText(image, label, (w - 210, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna, 1)


# ==========================================
# 3. LOGIKA UTAMA
# ==========================================
def main():
    global mirror_on

    JUMLAH_TAMBAHAN = 100
    JARAK_KALIBRASI = 60   # cm

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Pilih Kata
    print("\n=== MODE MENAMBAH DATASET ===")
    for idx, name in enumerate(actions):
        print(f"[{idx}] {name}")
    try:
        choice = int(input("Pilih Nomor Kata: "))
        target_action = actions[choice]
    except Exception:
        print("Input salah!")
        return

    # Cek Folder Lanjutan
    start_folder = 0
    action_path  = os.path.join(DATA_PATH, target_action)
    if os.path.exists(action_path):
        existing = [int(f) for f in os.listdir(action_path) if f.isdigit()]
        if existing:
            last_seq = max(existing)
            last_seq_path = os.path.join(action_path, str(last_seq))
            if len(os.listdir(last_seq_path)) < sequence_length:
                start_folder = last_seq
            else:
                start_folder = last_seq + 1

    end_folder = start_folder + JUMLAH_TAMBAHAN
    print(f"Target Folder: {start_folder} sampai {end_folder - 1}")

    # -------------------------------------------------------
    # PENTING: refine_face_landmarks=False → face mesh tidak
    # diproses sama sekali, sehingga tidak ada titik wajah
    # yang muncul di results maupun di layar.
    # -------------------------------------------------------
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=False,   # ← MATIKAN face mesh
            enable_segmentation=False) as holistic:

        # -----------------------------------------------
        # PHASE 0: KALIBRASI FOCAL LENGTH
        # -----------------------------------------------
        focal_length = None
        print(f"\n[KALIBRASI] Berdiri tepat {JARAK_KALIBRASI}cm dari kamera, lalu tekan SPASI.")
        print("Tekan [M] untuk toggle mirror, [Q] untuk keluar.")

        while focal_length is None:
            ret, frame = cap.read()
            if not ret:
                break

            if mirror_on:
                frame = apply_mirror(frame)

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Overlay instruksi kalibrasi
            cv2.rectangle(image, (0, 0), (image.shape[1], 100), (0, 0, 0), -1)
            cv2.putText(image, "FASE KALIBRASI", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            cv2.putText(image, f"Berdiri tepat {JARAK_KALIBRASI}cm dari kamera",
                        (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, "SPASI = kalibrasi  |  M = mirror  |  Q = keluar",
                        (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            bahu_ok    = results.pose_landmarks is not None
            st_bahu    = "Bahu terdeteksi OK" if bahu_ok else "Bahu belum terdeteksi..."
            warna_bahu = (0, 220, 0) if bahu_ok else (0, 0, 220)
            cv2.putText(image, st_bahu, (20, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, warna_bahu, 2)

            tampilkan_mirror_status(image, mirror_on)
            cv2.imshow('OpenCV Feed', image)
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
            elif key == ord('m') or key == ord('M'):
                mirror_on = not mirror_on
                print(f"Mirror: {'ON' if mirror_on else 'OFF'}")
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # -----------------------------------------------
        # PHASE 1: STANDBY
        # -----------------------------------------------
        print("\n[STANDBY] Tekan SPASI untuk mulai rekam. [M] mirror, [Q] keluar.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if mirror_on:
                frame = apply_mirror(frame)

            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            jarak_cm = hitung_jarak_bahu(image, results, focal_length)

            cv2.rectangle(image, (0, 0), (image.shape[1], 95), (0, 0, 0), -1)
            cv2.putText(image, f"ACTION: {target_action}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)
            cv2.putText(image, f"Folder: {start_folder} s/d {end_folder - 1}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
            cv2.putText(image, "SPASI = mulai rekam", (20, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

            tampilkan_info_jarak(image, jarak_cm, y_offset=115)
            tampilkan_mirror_status(image, mirror_on)

            cv2.imshow('OpenCV Feed', image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                break
            elif key == ord('m') or key == ord('M'):
                mirror_on = not mirror_on
                print(f"Mirror: {'ON' if mirror_on else 'OFF'}")
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # -----------------------------------------------
        # PHASE 2: LOOPING PEREKAMAN
        # -----------------------------------------------
        for sequence in range(start_folder, end_folder):

            # Countdown sebelum rekam
            start_break    = time.time()
            break_duration = 2.0

            while True:
                elapsed = time.time() - start_break
                if elapsed > break_duration:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if mirror_on:
                    frame = apply_mirror(frame)

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                jarak_cm  = hitung_jarak_bahu(image, results, focal_length)
                countdown = int(math.ceil(break_duration - elapsed))

                cv2.putText(image, f'SIAP: {countdown}', (220, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
                cv2.putText(image, f"Seq: {sequence}/{end_folder - 1} | [M] Mirror",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

                tampilkan_info_jarak(image, jarak_cm, y_offset=450)
                tampilkan_mirror_status(image, mirror_on)

                cv2.imshow('OpenCV Feed', image)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('m') or key == ord('M'):
                    mirror_on = not mirror_on

            # Rekam sequence_length frame
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break

                if mirror_on:
                    frame = apply_mirror(frame)

                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                jarak_cm = hitung_jarak_bahu(image, results, focal_length)

                # Indikator REC
                cv2.circle(image, (615, 25), 10, (0, 0, 255), -1)
                cv2.putText(image, 'REC', (575, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(
                    image,
                    f'{target_action} | Seq {sequence} | Frame {frame_num + 1}/{sequence_length}',
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

                tampilkan_info_jarak(image, jarak_cm, y_offset=450)
                tampilkan_mirror_status(image, mirror_on)

                cv2.imshow('OpenCV Feed', image)

                # Simpan keypoints (TANPA face)
                keypoints = extract_keypoints(results)
                save_path = os.path.join(DATA_PATH, target_action, str(sequence))
                os.makedirs(save_path, exist_ok=True)
                np.save(os.path.join(save_path, str(frame_num)), keypoints)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\nBerhenti paksa.")
                    return
                elif key == ord('m') or key == ord('M'):
                    mirror_on = not mirror_on

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSelesai! Data tersimpan di folder {start_folder} sampai {end_folder - 1}.")
    print(f"Total sequence baru: {JUMLAH_TAMBAHAN} | Fitur per frame: 258 (tanpa face)")


if __name__ == "__main__":
    main()