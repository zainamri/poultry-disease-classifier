import os
import json
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

from utils.preprocessing import preprocess_image
from utils.model_loader import load_class_names, load_disease_classifier, get_mock_prediction

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Poultry Disease Classifier | EfficientNet-B0 + CLAHE",
    page_icon="🐔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi CSS untuk tampilan yang elegan dan modern
st.markdown("""
<style>
    .main-header {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        padding: 15px;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.95rem;
        margin-bottom: 25px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .prediction-card-warning {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .metric-badge {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Muat Daftar Kelas dan Model
class_names, class_map = load_class_names()
model, model_status = load_disease_classifier()

# ── SIDEBAR INFORMASI ────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/chicken.png", width=80)
    st.title("🐔 Informasi Skrining")
    st.markdown("---")
    
    st.subheader("📌 Arsitektur Model")
    st.markdown("""
    - **Backbone:** EfficientNet-B0 (Pretrained ImageNet)
    - **Preprocessing:** Resize $224\\times224$ px + **CLAHE** (LAB L-Channel, `clipLimit=2.0`, `tileGrid=8×8`)
    - **Head Klasifikasi:** GlobalAvgPooling → BatchNorm → Dense(512, ReLU, L2) → Dropout(0.4) → Dense(256, ReLU, L2) → Dropout(0.3) → Softmax(4)
    - **Akurasi Validasi:** **96.59%** (F1-Macro: **96.63%**)
    """)
    st.markdown("---")
    
    st.subheader("🦠 Panduan 4 Kelas Penyakit")
    st.markdown("""
    **1. Coccidiosis (Koksidiosis)**
    > Penyakit parasit intestinal akibat protozoa *Eimeria*. Gejala khas berupa feses berdarah atau lendir berwarna kecokelatan.
    
    **2. Salmonella (Salmonelosis)**
    > Infeksi bakteri *Salmonella*. Feses umumnya cair keputihan bergumpal seperti kapur atau hijau kekuningan.
    
    **3. New Castle Disease (NCD / Tetelo)**
    > Infeksi virus paramyxovirus akut. Feses berwarna hijau lumut pekat bercampur cairan putih lengket (gangguan pencernaan dan saraf pekat).
    
    **4. Healthy (Sehat)**
    > Feses ayam sehat umumnya berwarna abu-kecokelatan padat dengan endapan putih (asam urat) di bagian atasnya.
    """)
    st.markdown("---")
    st.caption("🚀 Dikembangkan untuk Tugas Pengembangan Machine Learning - Streamlit Community Cloud.")

# ── HEADER UTAMA ─────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-header'>🐔 Poultry Disease Classifier & Screening Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Sistem Klasifikasi Cerdas Penyakit Ayam Berbasis Citra Feses dengan Preprocessing CLAHE & EfficientNet-B0</div>", unsafe_allow_html=True)

st.markdown("""
<div class='disclaimer-box'>
    <b>⚠️ Disclaimer Medis / Dokter Hewan:</b><br>
    Aplikasi ini adalah <b>alat bantu skrining awal dan edukasi</b> berbasis Machine Learning (Computer Vision), <b>BUKAN</b> pengganti diagnosis klinis atau laboratorium dokter hewan profesional. Pastikan untuk selalu berkonsultasi dengan dokter hewan untuk kepastian diagnosis dan penanganan pengobatan ternak Anda.
</div>
""", unsafe_allow_html=True)

# ── NAVIGATION TABS ──────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Skrining & Diagnosis Citra", "📊 Tentang Model (Ablation Study & Metrik Transparansi)"])

# ── TAB 1: SKRINING & DIAGNOSIS ──────────────────────────────────────────────
with tab1:
    col_input, col_result = st.columns([1, 1.1], gap="large")
    
    with col_input:
        st.subheader("1️⃣ Upload Citra Feses Ayam")
        uploaded_file = st.file_uploader(
            "Pilih file gambar feses ayam (format: JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Pastikan citra fokus pada feses dengan pencahayaan yang memadai."
        )
        
        if uploaded_file is not None:
            try:
                # Muat gambar dengan PIL
                image = Image.open(uploaded_file)
                st.image(image, caption="📷 Preview Citra Asli (Original Input)", use_container_width=True)
                
                # Opsi Preprocessing Preview (Edukatif)
                st.markdown("---")
                st.subheader("2️⃣ Preprocessing & CLAHE Preview")
                st.write("Visualisasi perbaikan kontras adaptif yang direplikasi persis dari pipeline training:")
                
                # Jalankan preprocessing
                processed_batch, preview_orig, preview_clahe = preprocess_image(image, target_size=(224, 224), use_clahe=True)
                
                # Tampilkan Side-by-Side
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.image(preview_orig, caption="Resized (224×224) - Tanpa CLAHE", use_container_width=True)
                with col_p2:
                    st.image(preview_clahe, caption="Resized + CLAHE (LAB L-Channel)", use_container_width=True)
                    
                btn_predict = st.button("⚡ Jalankan Prediksi Klasifikasi", type="primary", use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ File gambar corrupt atau tidak dapat diproses: {str(e)}")
                uploaded_file = None
        else:
            st.info("💡 Silakan upload gambar citra feses pada kotak uploader di atas untuk memulai prediksi.")

    with col_result:
        st.subheader("3️⃣ Hasil Analisis & Prediksi")
        
        if uploaded_file is None:
            st.markdown("""
            <div style='border: 2px dashed #CBD5E1; padding: 50px; text-align: center; border-radius: 10px; color: #64748B;'>
                <h4>⏳ Menunggu Input Citra</h4>
                <p>Hasil prediksi, tingkat keyakinan (confidence score), dan probabilitas per kelas akan ditampilkan di sini setelah gambar diupload dan diproses.</p>
            </div>
            """, unsafe_allow_html=True)
        elif btn_predict:
            with st.spinner("🔄 Menganalisis fitur citra dengan model EfficientNet-B0 + CLAHE..."):
                if model_status == "success" and model is not None:
                    # Prediksi model asli Keras
                    preds = model.predict(processed_batch, verbose=0)[0]
                    pred_idx = int(np.argmax(preds))
                    pred_class = class_names[pred_idx]
                    confidence = float(preds[pred_idx]) * 100.0
                    probs = preds
                else:
                    # Prediksi simulasi / fallback jika model .keras belum disalin ke folder model/
                    pred_class, confidence, probs = get_mock_prediction(processed_batch, class_names)
                    if model_status == "missing_model":
                        st.warning("⚠️ **Mode Simulasi/Edukatif Aktif:** File bobot `model/model_efficientnetb0_clahe.keras` belum disalin ke server/folder `model/`. Menampilkan kalkulasi probabilitas simulasi statis (lihat petunjuk di tab *Tentang Model* atau `model/README_model.md`).")
                    elif model_status == "tensorflow_missing":
                        st.warning("⚠️ **Mode Simulasi Aktif:** Library TensorFlow belum terinstal di lingkungan ini. Menampilkan simulasi prediksi.")

                # Tampilkan Kartu Prediksi Utama
                if pred_class == "Healthy":
                    card_class = "prediction-card"
                    icon_status = "✅"
                else:
                    card_class = "prediction-card-warning"
                    icon_status = "🚨"
                    
                st.markdown(f"""
                <div class='{card_class}'>
                    <h4 style='margin:0; font-weight:400; opacity:0.9;'>Diagnosis Skrining Teratas</h4>
                    <h1 style='margin:10px 0; font-size:2.8rem;'>{icon_status} {pred_class.upper()}</h1>
                    <h3 style='margin:0; font-weight:600;'>Confidence Score: {confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Tampilkan Bar Chart Probabilitas
                st.markdown("#### 📊 Distribusi Probabilitas 4 Kelas")
                prob_df = pd.DataFrame({
                    "Kelas Penyakit": class_names,
                    "Probabilitas (%)": [float(p)*100.0 for p in probs]
                }).set_index("Kelas Penyakit")
                
                st.bar_chart(prob_df, color="#2E8B57", height=280)
                
                # Tampilkan Detail Probabilitas dalam Tabel
                st.markdown("#### 📋 Rincian Angka Probabilitas")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                cols_list = [col_m1, col_m2, col_m3, col_m4]
                for idx, cname in enumerate(class_names):
                    with cols_list[idx]:
                        st.metric(label=cname, value=f"{probs[idx]*100:.2f}%")

# ── TAB 2: TENTANG MODEL & ABLATION STUDY ────────────────────────────────────
with tab2:
    st.subheader("📈 Transparansi Model & Hasil Ablation Study")
    st.write("Metrik evaluasi lengkap di bawah ini bersumber dari eksperimen **Ablation Study** (6 konfigurasi model identik: EfficientNet-B0, InceptionV3, ResNet50 $\\times$ CLAHE vs No-CLAHE) pada dataset 8.067 citra feses ayam (split 80/20 stratified).")
    
    # Baca file training_history.json
    history_path = "model/training_history.json"
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            hist_data = json.load(f)
            
        final_m = hist_data.get("final_metrics", {})
        
        # Highlight Metrik Utama
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            st.metric("🏆 Accuracy (Validation)", f"{final_m.get('accuracy', 0.9659)*100:.2f}%", delta="+0.06% vs No-CLAHE")
        with col_t2:
            st.metric("🎯 F1-Macro Score", f"{final_m.get('f1_macro', 0.9663)*100:.2f}%")
        with col_t3:
            st.metric("⚖️ F1-Weighted Score", f"{final_m.get('f1_weighted', 0.9660)*100:.2f}%")
        with col_t4:
            st.metric("📦 Jumlah Kelas & Data", "4 Kelas | 8.067 Citra")
            
        st.markdown("---")
        
        # Tabel Ablation Study
        st.subheader("🔬 Tabel Perbandingan Ablation Study (6 Eksperimen)")
        ablation_list = hist_data.get("ablation_comparison", [])
        if ablation_list:
            df_ablation = pd.DataFrame(ablation_list)
            # Format persentase
            df_ablation["accuracy"] = df_ablation["accuracy"].apply(lambda x: f"{x:.2f}%")
            df_ablation["f1_macro"] = df_ablation["f1_macro"].apply(lambda x: f"{x:.2f}%")
            df_ablation["waktu_menit"] = df_ablation["waktu_menit"].apply(lambda x: f"{x:.1f} m")
            df_ablation.columns = ["Arsitektur CNN", "Preprocessing CLAHE", "Accuracy", "F1 Macro", "Waktu Training"]
            st.dataframe(df_ablation, use_container_width=True, hide_index=True)
            
        st.markdown("---")
        
        # Per-Class F1 Score & Kurva Training
        col_c1, col_c2 = st.columns([1, 1.2], gap="large")
        with col_c1:
            st.subheader("📑 F1-Score per Kelas (EfficientNet-B0 + CLAHE)")
            f1_pc = final_m.get("f1_per_class", {})
            if f1_pc:
                df_f1 = pd.DataFrame({
                    "Kelas Penyakit": list(f1_pc.keys()),
                    "F1-Score": [f"{v*100:.2f}%" for v in f1_pc.values()],
                    "Status Performansi": ["Sangat Tinggi (>97%)" if v>0.97 else "Tinggi (>95%)" for v in f1_pc.values()]
                })
                st.dataframe(df_f1, use_container_width=True, hide_index=True)
                
            st.markdown("""
            **Analisis & Catatan Peneliti:**
            - **Penanganan Class Imbalance:** Kelas *New Castle Disease (NCD)* hanya memiliki 562 citra (6.96% dari dataset). Dengan penerapan *scikit-learn `compute_class_weight` (max 3.0×)*, F1-Score NCD berhasil dipertahankan pada angka **96.86%**.
            - **Dampak CLAHE:** CLAHE secara adaptif menonjolkan tekstur halus dan gradasi warna pada kanal L (LAB space), meningkatkan akurasi EfficientNet-B0 dan ResNet50.
            """)
            
        with col_c2:
            st.subheader("📉 Kurva Training & Validasi (50 Epochs)")
            hist_epochs = hist_data.get("history", {})
            if hist_epochs and "accuracy" in hist_epochs:
                df_curve = pd.DataFrame({
                    "Epoch": hist_epochs["epochs"],
                    "Training Accuracy": hist_epochs["accuracy"],
                    "Validation Accuracy": hist_epochs["val_accuracy"]
                }).set_index("Epoch")
                st.line_chart(df_curve, color=["#3B82F6", "#10B981"])
                st.caption("Fase 1 (Epoch 1-30): Backbone Frozen. | Fase 2 (Epoch 31-50): Fine-Tuning 30 layer terakhir.")
    else:
        st.info("File metadata `model/training_history.json` belum ditemukan.")
