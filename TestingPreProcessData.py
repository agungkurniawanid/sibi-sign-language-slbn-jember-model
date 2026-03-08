# =============================================================================
# TestingPreProcessData.py
# =============================================================================
# Tujuan:
#   Memuat dan memvalidasi dataset keypoints (.npy) yang sudah dikumpulkan,
#   lalu menyiapkan data tersebut untuk proses training model. File ini juga
#   menampilkan statistik distribusi data per kelas dalam bentuk grafik bar.
#
#   Proses yang dilakukan:
#     1. Memuat file .npy dari folder DatasetSibiKeypoints
#     2. Membuat label mapping (kata -> angka)
#     3. Memvalidasi kelengkapan setiap sequence (30 frame penuh)
#     4. Membagi data menjadi training (95%) dan testing (5%)
#     5. Menampilkan statistik dan grafik distribusi data
#
# Cara Penggunaan:
#   Jalankan setelah dataset selesai dikumpulkan untuk memverifikasi kualitas data:
#   python TestingPreProcessData.py
#
# Output:
#   - Statistik jumlah data per kelas di terminal
#   - Grafik bar distribusi data (jendela pyplot)
#   - Objek X_train, X_test, y_train, y_test siap digunakan untuk training
#
# Urutan Eksekusi dalam Proyek:
#   1. (Collect Data)            -> Kumpulkan dataset
#   2. TestingPreProcessData.py  -> Validasi & pra-proses data  <-- FILE INI
#   3. (Training)                -> Latih model LSTM
# =============================================================================

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


# IMPOR CONFIG
from Config import DATA_PATH, actions, no_sequences, sequence_length

def main():
    print("🔄 Memulai Proses Loading Data...")
    
    # 1. SETUP LABEL MAPPING
    # Mengubah kata ('Saya', 'Buah') menjadi angka (0, 1)
    label_map = {label:num for num, label in enumerate(actions)}
    print(f"🏷️  Label Mapping: {label_map}")

    sequences, labels = [], []
    
    # Variabel untuk statistik
    action_counts = {}

    # 2. LOOPING LOAD DATA
    for action in actions:
        print(f"📂 Loading kelas: '{action}'...")
        
        # Ambil list folder sequence (hanya yang berupa angka)
        action_path = os.path.join(DATA_PATH, action)
        if not os.path.exists(action_path):
            print(f"⚠️  Folder {action} tidak ditemukan! Lewati.")
            continue
            
        # Filter hanya folder angka (untuk menghindari file sampah sistem)
        sequence_folders = [f for f in os.listdir(action_path) if f.isdigit()]
        
        # Hitung jumlah data valid
        valid_count = 0

        for sequence in sequence_folders:
            window = []
            is_sequence_complete = True
            
            # Loop setiap frame (0 sampai 29)
            for frame_num in range(sequence_length):
                npy_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
                
                # Cek apakah file ada
                if os.path.exists(npy_path):
                    res = np.load(npy_path)
                    window.append(res)
                else:
                    # Jika ada 1 frame saja yang hilang, sequence ini dianggap rusak
                    is_sequence_complete = False
                    # print(f"   ❌ Frame hilang: {action}/{sequence}/{frame_num}") # Uncomment untuk debug
                    break
            
            # Hanya masukkan ke dataset jika sequence FULL (30 frame lengkap)
            if is_sequence_complete and len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_map[action])
                valid_count += 1
        
        action_counts[action] = valid_count
        print(f"   ✅ Berhasil memuat {valid_count} video untuk '{action}'")

    # 3. KONVERSI KE NUMPY ARRAY
    print("\n📦 Mengemas data ke Numpy Array...")
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)

    # 4. SPLIT DATA (TRAIN & TEST)
    # Test size 0.05 artinya 5% data dipakai untuk ujian, 95% untuk latihan
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    # 5. CETAK HASIL AKHIR
    print("\n========================================")
    print("📊 STATISTIK DATASET")
    print("========================================")
    print(f"Total Sequences (X) : {X.shape}") # Harusnya (TotalVideo, 30, 1662)
    print(f"Total Labels (y)    : {y.shape}")
    print("----------------------------------------")
    print(f"Data Training       : {X_train.shape[0]} video")
    print(f"Data Testing        : {X_test.shape[0]} video")
    print("========================================")

    # 6. OUTPUT GRAFIK (VISUALISASI)
    # Membuat Bar Chart distribusi data
    plt.figure(figsize=(10, 6))
    plt.bar(action_counts.keys(), action_counts.values(), color=['blue', 'orange', 'green'][:len(actions)])
    plt.title('Jumlah Data per Kelas (Valid Sequences)')
    plt.xlabel('Kata / Kelas')
    plt.ylabel('Jumlah Video')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Menampilkan angka di atas batang
    for i, v in enumerate(action_counts.values()):
        plt.text(i, v + 1, str(v), ha='center', fontweight='bold')

    print("📈 Menampilkan grafik distribusi data...")
    plt.show() # Jendela grafik akan muncul

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Jalankan fungsi main
    X_train, X_test, y_train, y_test = main()