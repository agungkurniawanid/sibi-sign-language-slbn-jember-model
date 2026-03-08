# =============================================================================
# SetupFolders.py
# =============================================================================
# Tujuan:
#   Membuat struktur folder dataset secara otomatis sebelum proses pengumpulan
#   data dimulai. Folder dibuat berdasarkan konfigurasi di Config.py dengan
#   hierarki: DatasetSibiKeypoints / <NamaKata> / <NomorSequence>
#
#   Contoh hasil folder:
#     DatasetSibiKeypoints/
#       Saya/
#         0/
#         1/
#         ...
#         99/
#       Buah/
#         0/ ... 99/
#
# Cara Penggunaan:
#   Jalankan file ini SEKALI sebelum memulai pengumpulan data:
#   python SetupFolders.py
#
# Urutan Eksekusi dalam Proyek:
#   1. Config.py        -> Atur konfigurasi
#   2. SetupFolders.py  -> Buat struktur folder  <-- FILE INI
#   3. (Collect Data)   -> Kumpulkan data keypoints
#   4. Training         -> Latih model
# =============================================================================

import os
# IMPOR DARI CONFIG (Agar data selalu sama)
from Config import DATA_PATH, actions, no_sequences

def main():
    print(f"📂 Memulai pembuatan folder di: {DATA_PATH}")
    print(f"📝 Total Kata: {len(actions)}")
    print(f"🎬 Jumlah Video per Kata: {no_sequences}")
    
    for action in actions:
        # Loop untuk membuat folder sequence (0 sampai 99)
        for sequence in range(no_sequences):
            try:
                # Membuat path: SIBI_Dataset_Keypoints/Kata/NomorSequence
                target_path = os.path.join(DATA_PATH, action, str(sequence))
                os.makedirs(target_path, exist_ok=True)
                
            except Exception as e:
                print(f"❌ Error membuat folder {action}/{sequence}: {e}")
                
    print("\n✅ SUKSES! Semua folder telah siap.")

if __name__ == "__main__":
    main()