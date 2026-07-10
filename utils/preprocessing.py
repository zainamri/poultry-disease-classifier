import os
import cv2
import numpy as np
from PIL import Image
try:
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess_input
except ImportError:
    # Fallback jika tensorflow sedang diimpor / versi ringan
    def eff_preprocess_input(x):
        return x

def apply_clahe(rgb_img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Menerapkan Contrast Limited Adaptive Histogram Equalization (CLAHE)
    pada citra RGB melalui ruang warna LAB (khusus pada kanal L/Luminance).
    
    Parameter:
    - rgb_img: numpy array RGB dtypes uint8 ukuran (H, W, 3) atau PIL Image
    - clip_limit: batas amplifikasi kontras (default 2.0 sesuai training)
    - tile_grid_size: ukuran grid pemrosesan lokal (default 8x8 sesuai training)
    
    Return:
    - numpy array RGB dtypes uint8 setelah CLAHE
    """
    if isinstance(rgb_img, Image.Image):
        rgb_img = np.array(rgb_img.convert("RGB"))
    elif not isinstance(rgb_img, np.ndarray):
        raise ValueError("Input harus berupa PIL Image atau numpy array.")
        
    # Pastikan format uint8
    if rgb_img.dtype != np.uint8:
        rgb_img = np.clip(rgb_img, 0, 255).astype(np.uint8)
        
    # Konversi RGB ke LAB
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    
    # Inisialisasi dan terapkan CLAHE pada kanal L (indeks 0)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    
    # Konversi kembali dari LAB ke RGB
    clahe_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return clahe_rgb

def preprocess_image(image, target_size=(224, 224), use_clahe=True):
    """
    Menjalankan seluruh pipeline preprocessing yang PERSIS IDENTIK 100% dengan ChickenGenerator._load saat training:
    1. Terapkan CLAHE pada citra resolusi penuh RGB melalui ruang warna LAB (kanal L)
    2. Resize ke ukuran target (224x224) menggunakan interpolasi INTER_LINEAR
    3. Normalisasi menggunakan tf.keras.applications.efficientnet.preprocess_input()
    
    Parameter:
    - image: PIL Image atau numpy array dari upload user
    - target_size: ukuran citra target tuple (W, H)
    - use_clahe: boolean, apakah menerapkan CLAHE atau tidak
    
    Return:
    - tuple: (processed_batch, preview_original, preview_clahe)
      * processed_batch: tensor siap inference berukuran (1, 224, 224, 3) dtype float32
      * preview_original: citra asli setelah resize (numpy array uint8 RGB)
      * preview_clahe: citra hasil CLAHE setelah resize (numpy array uint8 RGB)
    """
    if isinstance(image, Image.Image):
        img_rgb = np.array(image.convert("RGB"))
    else:
        img_rgb = np.array(image)
        
    # Simpan preview citra asli setelah resize
    preview_original = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 1. Terapkan CLAHE TERLEBIH DAHULU pada resolusi penuh (sama persis dengan _load di training generator)
    if use_clahe:
        img_clahe_full = apply_clahe(img_rgb)
    else:
        img_clahe_full = img_rgb.copy()
        
    # 2. Resize ke 224x224 setelah CLAHE menggunakan INTER_LINEAR (bawaan cv2.resize di _load)
    preview_clahe = cv2.resize(img_clahe_full, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 3. Normalisasi dengan preprocess_input EfficientNet
    img_float = preview_clahe.astype(np.float32)
    img_normalized = eff_preprocess_input(img_float)
    
    # Tambahkan dimensi batch (1, 224, 224, 3)
    processed_batch = np.expand_dims(img_normalized, axis=0)
    
    return processed_batch, preview_original, preview_clahe
