# Petunjuk Penempatan File Model (`.keras`)

Folder ini (`poultry-disease-app/model/`) dirancang untuk menyimpan file bobot model hasil training beserta metadatanya.

## File yang Harus Ada di Folder Ini:
1. `class_names.json` (sudah tersedia) — Mapping indeks 0-3 ke nama kelas penyakit ayam.
2. `training_history.json` (sudah tersedia) — Riwayat akurasi, loss, dan tabel perbandingan ablation study dari notebook training.
3. `model_efficientnetb0_clahe.keras` — **File bobot model Keras final hasil training.**

---

## Cara Mendapatkan `model_efficientnetb0_clahe.keras`:

### 1. Jika Anda Menjalankan Training di Google Colab:
Di akhir notebook `Chicken_Comparison_AblationStudy_ini (1).ipynb`, jalankan perintah berikut untuk mengunduh model terbaik dari hasil fine-tuning:

```python
import shutil
from google.colab import files

# Salin file bobot terbaik ke nama standar deployment
shutil.copy('best_EfficientNet-B0_CLAHE_ft.keras', 'model_efficientnetb0_clahe.keras')

# Unduh ke komputer lokal
files.download('model_efficientnetb0_clahe.keras')
```

Setelah file `model_efficientnetb0_clahe.keras` terunduh ke komputer Anda, pindahkan/salin file tersebut ke folder ini (`poultry-disease-app/model/`).

### 2. Jika File Model Belum Disalin (Mode Demonstrasi/Edukatif):
Aplikasi Streamlit dirancang dengan **sistem toleransi error yang cerdas**. Jika file `model_efficientnetb0_clahe.keras` belum ada di folder ini, aplikasi **tetap dapat berjalan dengan normal** dalam mode *Simulasi/Demonstrasi Edukatif*. 
Semua fitur interaktif (upload gambar, visualisasi side-by-side CLAHE RGB vs LAB, grafik probabilitas, sidebar edukasi, dan dashboard "Tentang Model") tetap berfungsi penuh dan dapat diuji coba.
