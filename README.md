# 🐔 Poultry Disease Classifier & Screening Dashboard

Aplikasi web interaktif berbasis **Streamlit** untuk skrining awal dan klasifikasi penyakit ayam berbasis citra feses. Aplikasi ini memanfaatkan model *Deep Learning* dengan arsitektur **EfficientNet-B0** dan teknik preprocessing **CLAHE (Contrast Limited Adaptive Histogram Equalization)** pada kanal L (LAB space).

---

## 📌 Fitur Utama
1. **Skrining Citra Feses:** Klasifikasi citra ke dalam 4 kelas penyakit (Coccidiosis, Salmonella, New Castle Disease/NCD, dan Healthy).
2. **Visualisasi Preprocessing CLAHE:** Side-by-side preview citra asli dengan citra hasil perbaikan kontras adaptif CLAHE.
3. **Analisis Keyakinan (Confidence Score):** Menampilkan probabilitas prediksi lengkap untuk keempat kelas dalam bentuk Bar Chart dan tabel statistik.
4. **Transparansi & Ablation Study:** Tab khusus menampilkan kurva training/validasi dan tabel perbandingan 6 model eksperimen (EfficientNet-B0, InceptionV3, dan ResNet50 dengan/tanpa CLAHE).

---

## 📊 Dataset & Model
- **Dataset Asli:** [Chicken Disease Dataset di Kaggle (Machuve dkk.)](https://www.kaggle.com/datasets/allandclive/chicken-disease-1) — 8.067 citra feses ayam terbagi dalam 4 kelas.
- **Arsitektur Model:** Transfer Learning dengan backbone `EfficientNet-B0` (Pretrained ImageNet) + Custom Head (GlobalAvgPool → BatchNorm → Dense(512) → Dropout(0.4) → Dense(256) → Dropout(0.3) → Softmax).
- **Performa Evaluasi:**
  - Akurasi Validasi: **96.59%**
  - F1-Macro Score: **96.63%**

---

## 🛠️ Cara Instalasi & Menjalankan secara Lokal

### 1. Prasyarat
Pastikan Python 3.9+ telah terinstal di komputer Anda. Disarankan menggunakan *virtual environment*.

```bash
# Buat virtual environment
python -m venv venv

# Aktifkan virtual environment
# Di Windows (Command Prompt):
venv\Scripts\activate.bat
# Di Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Di Linux/macOS:
source venv/bin/activate
```

### 2. Instal Library Dependensi
Jalankan perintah berikut untuk menginstal semua dependensi yang diperlukan:
```bash
pip install -r requirements.txt
```

### 3. Penempatan File Model (.keras)
Pastikan file bobot model hasil training (`model_efficientnetb0_clahe.keras`) telah diletakkan di dalam folder `model/`:
```
poultry-disease-app/
└── model/
    ├── model_efficientnetb0_clahe.keras
    ├── class_names.json
    └── training_history.json
```
*(Catatan: Jika model belum ada, aplikasi tetap berjalan dalam mode simulasi/edukatif).*

### 4. Jalankan Aplikasi Streamlit
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser Anda pada alamat `http://localhost:8501`.

---

## ☁️ Panduan Deployment ke Streamlit Community Cloud (Tahap 3)

### Langkah 1: Persiapan Repository GitHub
1. Pastikan seluruh struktur folder `poultry-disease-app/` sudah rapi.
2. Cek ukuran file `model/model_efficientnetb0_clahe.keras`:
   - **Jika ukuran file < 100 MB:** Anda dapat langsung melakukan `git add`, `git commit`, dan `git push` ke repository GitHub Anda.
   - **Jika ukuran file > 100 MB (Batasan GitHub):** Gunakan **Git LFS (Large File Storage)**:
     ```bash
     git lfs install
     git lfs track "*.keras"
     git add .gitattributes
     git add model/model_efficientnetb0_clahe.keras
     git commit -m "Add model with Git LFS"
     git push origin main
     ```
     *(Alternatif lain: Anda bisa menyimpan `.keras` di Google Drive / Hugging Face Hub lalu mengunduhnya secara otomatis menggunakan `gdown` atau `huggingface_hub` saat aplikasi pertama kali dijeda di cloud).*

### Langkah 2: Deploy di Streamlit Community Cloud
1. Buka [Streamlit Community Cloud (share.streamlit.io)](https://share.streamlit.io/) dan login menggunakan akun GitHub Anda.
2. Klik tombol **"New app"**.
3. Pilih repository GitHub Anda, branch `main`, dan arahkan **Main file path** ke `app.py` (atau `poultry-disease-app/app.py` jika folder berada di dalam repo).
4. Klik **"Deploy!"**.
5. Server Streamlit akan membaca `requirements.txt` dan `packages.txt` secara otomatis, menginstal library (termasuk `opencv-python-headless` & `tensorflow-cpu`), dan meluncurkan dashboard Anda ke publik dalam beberapa menit!
