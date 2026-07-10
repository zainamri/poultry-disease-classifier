import os
import json
import streamlit as st
import numpy as np
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None

@st.cache_resource
def load_class_names(json_path="model/class_names.json"):
    """
    Memuat mapping indeks kelas ke label dari file class_names.json.
    Diberi cache dengan @st.cache_resource agar hanya dibaca sekali.
    """
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        # Urutkan berdasarkan key string/integer indeks numerik (0, 1, 2, 3)
        sorted_keys = sorted(mapping.keys(), key=lambda x: int(x))
        class_names = [mapping[k] for k in sorted_keys]
        return class_names, mapping
    else:
        # Default fallback jika json belum ada
        default_classes = ["Coccidiosis", "Healthy", "New Castle Disease", "Salmonella"]
        default_map = {str(i): c for i, c in enumerate(default_classes)}
        return default_classes, default_map

@st.cache_resource
def load_disease_classifier(model_path="model/model_efficientnetb0_clahe.keras"):
    """
    Memuat model Keras (.keras) yang sudah dilatih dengan @st.cache_resource.
    Jika file model belum ditempatkan oleh user di folder model/,
    maka mengembalikan status 'missing' agar aplikasi tetap berjalan dengan UI edukatif.
    """
    if tf is None:
        return None, "tensorflow_missing"
        
    # Daftar jalur alternatif jika model dengan nama utama atau di lokasi alternatif
    possible_paths = [
        model_path,
        "model/best_EfficientNet-B0_CLAHE_ft.keras",
        "../best_EfficientNet-B0_CLAHE_ft.keras",
        "best_EfficientNet-B0_CLAHE_ft.keras",
        "model/efficientnetb0_clahe.keras"
    ]
    
    target_path = None
    for p in possible_paths:
        if os.path.exists(p):
            target_path = p
            break
            
    if target_path:
        try:
            model = load_model(target_path, compile=False)
            return model, "success"
        except Exception as e:
            return None, f"error: {str(e)}"
    else:
        return None, "missing_model"

def get_mock_prediction(processed_batch, class_names):
    """
    Fungsi simulasi/mock prediksi jika model_efficientnetb0_clahe.keras
    belum disalin ke folder model/. Berguna untuk demonstrasi dan verifikasi UI Streamlit.
    """
    # Gunakan statistik piksel untuk membuat variasi probabilitas yang rasional dan statis
    img_mean = np.mean(processed_batch)
    img_std = np.std(processed_batch)
    
    # Hitung skor probabilitas palsu (mock) bernilai realistis
    np.random.seed(int(img_mean * 100) % 10000)
    logits = np.random.uniform(0.5, 3.5, size=len(class_names))
    # Softmax
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx]) * 100.0
    
    return pred_class, confidence, probs
