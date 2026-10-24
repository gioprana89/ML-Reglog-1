# STATCAL — Praktikum Machine Learning: Regresi Logistik Biner

**Studi kasus:** Prediksi keputusan nasabah berlangganan deposito berjangka.  
**Data:** 1.000 nasabah **simulasi**, seluruh variabel dalam Bahasa Indonesia.

## Isi aplikasi dan pemetaan bab buku

| Tab | Bab buku | Fitur utama |
|---|---|---|
| 1 · Dataset | Bab 1 | Melihat semua 1.000 observasi, ringkasan kelas, kamus variabel, unggah data dan unduh dataset |
| 2 · Pemilihan Variabel | Bab 2 | Menentukan Y biner dan satu atau lebih variabel X, serta melihat distribusi target |
| 3 · Data Splitting | Bab 3 | Persentase training/testing (50–90%), random seed, stratifikasi, hasil dua tabel, ekspor **dua Excel terpisah** |
| 4 · Probabilitas & Klasifikasi | Bab 4 | Menampilkan koefisien dan persamaan lengkap, menghitung z dan probabilitas sigmoid, pengaturan threshold, klasifikasi, kontribusi setiap variabel, ekspor Excel |
| 5 · Evaluasi Model | Bab 5 | Mengukur Accuracy, Precision, Recall, Specificity, F1-score, ROC-AUC, confusion matrix, dan unduh hasil testing |

## Instalasi dan menjalankan (Windows, Mac, Linux)

1. Ekstrak ZIP ke folder kerja, **jangan pisahkan `app.py`, `ml_core.py`, dan file CSV**.
2. Pastikan Python 3.10 atau lebih baru terpasang.
3. Buka terminal (Command Prompt / PowerShell / VS Code Terminal) di folder tersebut.
4. Jalankan perintah:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

5. Akses alamat lokal yang muncul, umumnya `http://localhost:8501`.

Untuk menjalankan di Streamlit Community Cloud, unggah isi folder ke GitHub, kemudian pilih `app.py` sebagai entry point dan pastikan `requirements.txt` serta dataset CSV berada di repository yang sama.

## Konfigurasi baku sesuai contoh buku

- Variabel dependen: `Berlangganan_Deposito` (0 = Tidak, 1 = Ya).
- Variabel independen: keenam prediktor (`Usia`, `Penghasilan_Bulanan`, `Saldo_Tabungan`, `Pekerjaan`, `Memiliki_Pinjaman`, `Pernah_Dihubungi`).
- Training = **80% (800 nasabah)**, testing = **20% (200 nasabah)**.
- Random seed = **42**.
- Stratified splitting = **Ya**, dengan distribusi kelas pada training **549 kelas 0 + 251 kelas 1** dan testing **137 kelas 0 + 63 kelas 1**.
- Model = `sklearn.linear_model.LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000)` dengan regularisasi L2, threshold prediksi 0.50.
- Pengkodean tetap (tidak belajar dari testing): `Pekerjaan` menjadi 3 dummy (referensi **Karyawan**), `Ya` = 1, `Tidak` = 0. Penghasilan dan saldo dibagi 1.000.000 sehingga koefisien berlaku untuk satuan **juta rupiah**.
- Contoh nasabah: usia 45, penghasilan Rp10.000.000, saldo Rp50.000.000, PNS, pinjaman Tidak, pernah dihubungi Ya. Dengan versi paket yang digunakan untuk pengujian, nilai **z = 0.758880**, **P(Y=1) = 0.681111** (68.11%), prediksi **Y=1**.

**Penting:** beberapa versi `scikit-learn`/metode optimisasi dapat menghasilkan selisih kecil pada digit koefisien terakhir. Aplikasi selalu menampilkan hasil **model yang aktif**, menggunakan koefisien dengan presisi penuh. Karena itu, salin angka pada aplikasi ke naskah buku setelah menentukan dan mempertahankan satu versi perangkat lunak.

## Memahami pengaruh pengaturan

- Saat pilihan X, Y, training%, atau random seed diubah, model pada Tab 4 dan Tab 5 otomatis menggunakan data training baru.
- ID nasabah hanya disertakan pada tabel/Excel untuk penelusuran; ID **tidak** dimasukkan ke model.
- Jika memilih Y selain `Berlangganan_Deposito`, label klasifikasi merujuk pada Y yang baru, bukan lagi deposito.
- `Stratified splitting` menjaga proporsi kelas Y di kedua bagian sedekat mungkin.
- Aplikasi memisahkan testing sebelum melatih model. Encoding dan konversi satuan menggunakan aturan yang sudah ditetapkan, tidak menghitung parameter dari testing.
- Evaluasi Tab 5 memakai threshold 0.50; slider threshold di Tab 4 hanya untuk ilustrasi prediksi nasabah baru.

## Catatan untuk buku praktikum

Dataset **sepenuhnya sintetis**. Pola yang dipelajari model bukan fakta empiris bank. Evaluasi pada data testing mengukur kinerja pada data simulasi yang disisihkan, bukan kinerja di bank nyata.

Aplikasi **belum** melakukan tuning atau pemilihan hyperparameter, cross-validation, dan penanganan missing values. Karena dataset asli telah dibersihkan dan disimulasikan tanpa nilai kosong, validasi akan menolak unggahan yang tidak memenuhi format dataset buku. Untuk analisis data riil, tambahkan preprocessing dan validasi yang sesuai.

## Struktur file

```
STATCAL_Regresi_Logistik_Biner_Streamlit/
├── app.py
├── ml_core.py
├── requirements.txt
├── README.md
├── .streamlit/config.toml
├── Simulasi_1000_Nasabah_Regresi_Logistik_Biner.csv
├── Simulasi_1000_Nasabah_Regresi_Logistik_Biner.xlsx
└── tests/test_ml_core.py
```

**Menjalankan tes komputasi (opsional):**

```bash
python -m pip install pytest
python -m pytest -q tests
```
