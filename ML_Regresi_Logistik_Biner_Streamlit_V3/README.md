# Machine Learning: Regresi Logistik Biner — Versi 3

**Aplikasi Python Streamlit untuk buku praktikum 12 bab**  
Studi kasus: **Prediksi Nasabah Berlangganan Deposito Berjangka**  
Data bawaan: **1.000 nasabah simulasi**, bukan data pribadi nasabah sesungguhnya.

## Fitur aplikasi dan urutan bab

| Tab | Bab | Praktikum |
|---|---|---|
| 1. Dataset | 1 | Membaca seluruh dataset dan mengenal kolom, nilai, tipe data |
| 2. Variabel | 2 | Memilih target Y dan prediktor X |
| 3. Splitting | 3 | Training/testing, seed, stratifikasi, ekspor dua file Excel |
| 4. Probabilitas | 4 | Koefisien, persamaan lengkap, nilai z, sigmoid, klasifikasi satu nasabah |
| 5. Prediksi Training | 5 | **800 data training:** semua variabel asli, ID, Y aktual, z, P(Y=0), P(Y=1), Y prediksi, status benar/salah; unduh Excel |
| 6. Prediksi Testing | 6 | **200 data testing:** tabel prediksi lengkap tanpa melatih ulang model; unduh Excel |
| 7. Confusion Matrix | 7 | TN, FP, FN, TP dan interpretasi; unduh Excel |
| 8. Accuracy | 8 | Rumus, substitusi jumlah observasi, dan interpretasi |
| 9. Recall / Sensitivity | 9 | Rumus, substitusi, dan interpretasi |
| 10. Specificity | 10 | Rumus, substitusi, dan interpretasi |
| 11. Precision & F1-score | 11 | Dua metrik, rumus dan substitusi, interpretasi |
| 12. ROC & AUC | 12 | Kurva ROC, ROC-AUC, titik operasi threshold 0.5, unduh titik ROC ke Excel |

**Tema default:** Dark Mode, di `.streamlit/config.toml`. Tidak ada branding STATCAL ONLINE.

## Instalasi

Gunakan **Python 3.10 atau lebih baru**. Ekstrak ZIP, kemudian dari direktori folder aplikasi jalankan:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Aplikasi biasanya tersedia di `http://localhost:8501`.

## Pengaturan default sesuai buku

- Target: `Berlangganan_Deposito`, `0 = Tidak`, `1 = Ya`.
- Prediktor: `Usia`, `Penghasilan_Bulanan`, `Saldo_Tabungan`, `Pekerjaan`, `Memiliki_Pinjaman`, `Pernah_Dihubungi`.
- Training = 80% (800 observasi); testing = 20% (200 observasi).
- Random seed = **42**; stratified splitting = **aktif**.
- Model: `LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000)` dengan regularisasi L2.
- Encoding pekerjaan: Karyawan kategori referensi; 3 dummy lainnya. Ya=1 dan Tidak=0 untuk kedua prediktor biner.
- Penghasilan dan saldo diubah ke satuan **juta rupiah**. ID tidak menjadi prediktor.
- **Threshold evaluasi Bab 5–11 dan titik operasi Bab 12 adalah 0.50.** Slider threshold pada Bab 4 hanya untuk eksperimen prediksi satu nasabah.
- Testing hanya digunakan setelah training selesai, bukan untuk `fit()` model.

### Angka yang menjadi acuan pada pembagian default

- Distribusi training: 549 kelas 0, 251 kelas 1.
- Distribusi testing: 137 kelas 0, 63 kelas 1.
- Confusion matrix: TN=125, FP=12, FN=38, TP=25.
- Accuracy=75.00%; Precision=67.57%; Recall=39.68%; Specificity=91.24%; F1-score=0.5000.
- **ROC-AUC dihitung langsung dari probabilitas model aktif**. Nilainya dapat sedikit berbeda antarversi scikit-learn (contoh lingkungan uji: 0.7585). Jika hasil aplikasi yang sudah dideploy berbeda sedikit dengan gambar buku lama, gunakan satu versi environment dan perbarui gambar/nilai buku dari aplikasi final.

## Cara aplikasi menjaga konsistensi

1. Tab 3 membagi data dengan indeks training dan testing yang tidak saling tumpang tindih.
2. Model hanya di-`fit` pada training. Semua prediksi pada Bab 4–12 memakai objek model aktif yang sama pada satu eksekusi aplikasi.
3. Tab 5 menghitung hasil probabilitas/prediksi semua training. Tab 6 menghitung hasil probabilitas/prediksi semua testing.
4. Bab 7–11 menghitung metrik pada prediksi testing dengan threshold tetap 0.50.
5. ROC-AUC pada Bab 12 berasal dari **probabilitas testing**, bukan dari satu confusion matrix.
6. Mengubah seed, rasio split, Y, atau X akan menghasilkan model baru dan menyegarkan hasil semua tab berikutnya. Hasil default pada buku baru berlaku saat pengaturan default digunakan.

> Kinerja pada training bukan evaluasi generalisasi. Jangan memakai data testing untuk tuning, memilih prediktor, atau mengoptimalkan threshold lalu menyebut hasil tersebut sebagai estimasi final yang independen. Lakukan validasi atau cross-validation di data training jika ingin optimasi.

## Upload kode ke GitHub / Streamlit Community Cloud

Unggah **isi folder** V3 ke akar (root) repository. Untuk aplikasi lama `ml-reglog-1.streamlit.app`, file yang harus diperbarui terutama `app.py`, `ml_core.py`, `requirements.txt`, dan **`.streamlit/config.toml`**. Simpan CSV default di root. `app.py` tetap menjadi entry point.

Jika repo GitHub memiliki commit baru yang tidak ada di lokal, lakukan commit lokal lalu `git pull --rebase origin master` dan `git push origin master` (bila cabang utamanya `master`). Hindari `--force`.

## Pengujian

```bash
python -m pip install pytest
python -m pytest -q tests
```

File `tests/test_prediction_tables.py` mengecek apakah indeks, ID, label aktual, probabilitas, hasil prediksi, status benar/salah, dan seluruh metrik evaluasi saling sesuai. Pengujian komputasi tidak sama dengan pengujian antarmuka di browser.

## Isi folder

```text
ML_Regresi_Logistik_Biner_Streamlit_V3/
├── app.py
├── ml_core.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── Simulasi_1000_Nasabah_Regresi_Logistik_Biner.csv
├── Simulasi_1000_Nasabah_Regresi_Logistik_Biner.xlsx
└── tests/
    ├── test_ml_core.py
    └── test_prediction_tables.py
```
