# =============================================================================
# SystemInformation.py
# =============================================================================
# Tujuan:
#   Menampilkan informasi lengkap tentang spesifikasi sistem dan versi library
#   yang terinstall. File ini berguna untuk debugging kompatibilitas environment
#   sebelum menjalankan proses training maupun inferensi model.
#
#   Informasi yang ditampilkan:
#     - OS, arsitektur, dan prosesor
#     - Versi Python dan path executable
#     - Status virtual environment
#     - Versi NumPy, OpenCV, MediaPipe, TensorFlow
#     - Ketersediaan GPU/CUDA untuk TensorFlow
#     - Versi paket dari pip
#
# Cara Penggunaan:
#   Jalankan file ini untuk memverifikasi environment sebelum mulai coding:
#   python SystemInformation.py
# =============================================================================

import sys
import platform
import os
import subprocess

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except Exception:
        return "Tidak tersedia"

print("=" * 60)
print("🔍 SYSTEM INFORMATION")
print("=" * 60)

# ==================== OS & PYTHON ====================
print("\n🖥️ OS INFO")
print(f"OS              : {platform.system()} {platform.release()}")
print(f"OS Version      : {platform.version()}")
print(f"Architecture    : {platform.architecture()[0]}")
print(f"Processor       : {platform.processor()}")

print("\n🐍 PYTHON INFO")
print(f"Python Version  : {sys.version}")
print(f"Python Path     : {sys.executable}")

# ==================== ENV ====================
print("\n📦 VIRTUAL ENV")
print(f"Virtual Env     : {os.environ.get('VIRTUAL_ENV', 'Tidak menggunakan venv')}")

# ==================== NUMPY ====================
try:
    import numpy as np
    print("\n📊 NUMPY")
    print(f"NumPy Version   : {np.__version__}")
except:
    print("\n📊 NUMPY : Tidak terinstall")

# ==================== OPENCV ====================
try:
    import cv2
    print("\n📷 OPENCV")
    print(f"OpenCV Version  : {cv2.__version__}")
    print(f"Video Backends  : {cv2.videoio_registry.getBackends()}")
except:
    print("\n📷 OPENCV : Tidak terinstall")

# ==================== MEDIAPIPE ====================
try:
    import mediapipe as mp
    print("\n🖐️ MEDIAPIPE")
    print(f"MediaPipe Version : {mp.__version__}")
except:
    print("\n🖐️ MEDIAPIPE : Tidak terinstall")

# ==================== TENSORFLOW ====================
try:
    import tensorflow as tf
    print("\n🧠 TENSORFLOW")
    print(f"TensorFlow Version : {tf.__version__}")
    print(f"Built with CUDA    : {tf.test.is_built_with_cuda()}")
    print(f"Available Devices  :")
    for d in tf.config.list_physical_devices():
        print(f" - {d.device_type}: {d.name}")
except Exception as e:
    print("\n🧠 TENSORFLOW : Tidak terinstall / error")
    print(e)

# ==================== PIP LIST (IMPORTANT ONLY) ====================
print("\n📦 PACKAGE VERSIONS (pip)")
print("- numpy       :", run_cmd("pip show numpy | findstr Version"))
print("- tensorflow  :", run_cmd("pip show tensorflow | findstr Version"))
print("- mediapipe   :", run_cmd("pip show mediapipe | findstr Version"))
print("- opencv      :", run_cmd("pip show opencv-python | findstr Version"))
print("- protobuf    :", run_cmd("pip show protobuf | findstr Version"))

print("\n✅ SYSTEM CHECK SELESAI")
print("=" * 60)
