# =============================================================================
# Config.py
# =============================================================================
# Tujuan:
#   File konfigurasi pusat untuk seluruh proyek deteksi bahasa isyarat SIBI
#   (Sistem Isyarat Bahasa Indonesia). Semua nilai konfigurasi global seperti
#   path dataset, daftar kata/label, jumlah video, dan panjang sequence
#   didefinisikan di sini agar konsisten di seluruh file proyek.
#
# Cara Penggunaan:
#   Impor nilai yang dibutuhkan dari file ini di file lain, contoh:
#   from Config import DATA_PATH, actions, no_sequences, sequence_length
#
# Catatan:
#   Ubah nilai di file ini saja jika ingin mengubah konfigurasi global.
#   Jangan mendefinisikan ulang nilai yang sama di file lain.
# =============================================================================

import os
import numpy as np

# ==========================================
# KONFIGURASI PUSAT (Ubah hanya di sini)
# ==========================================

# 1. Path Penyimpanan Data
# Pastikan nama folder KONSISTEN. Sebelumnya Anda pakai dua nama berbeda.
# Kita tetapkan satu nama: 'SIBI_Dataset_Keypoints'
DATA_PATH = os.path.join('DatasetSibiKeypoints') 

# 2. Daftar Kata / Label SIBI
list_kata = np.array([
    "Saya",
    "Buah",
    "Makan",
    "Agar",
    "Kuat",
    "Sayur",
    "Ibu",
    "An",      # Cek kembali apakah ini typo "Dan"?
    "Sabar",
    "Siap",
    "Kan",     # Cek kembali apakah ini typo "Ikan"/"Akan"?
    "Kue",
    "Untuk",
    "Tamu",
    "Cerita",
    "Tentang",
    "Malam",
    "Keluarga",
    "Bapak",
    "Siram",
    "Harus",
    "Rajin",
    "Kakak",
    "Obat",
    "Gelas"
])

# Untuk Collect, Testing dan Training
# actions = np.array(['Saya', 'Buah', 'Kuat', 'Agar', 'Makan', 'Sayur', 'Ibu', 'An', 'Sabar', 'Siap', 'Gelas', 'Obat', 'Kakak', 'Rajin', 'Harus']);
actions = np.array(['Saya', 'Makan', 'Obat']);


# 3. Jumlah Video & Frame
# Sesuai request Anda: 100 video per kata
no_sequences = 100  

# 30 frame = 1 detik (jika webcam 30fps)
sequence_length = 45