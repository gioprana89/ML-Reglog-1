# Buku Praktikum Machine Learning — Bab 2

**Studi kasus:** Prediksi keputusan nasabah berlangganan deposito (data SIMULASI 1.000 nasabah).

Isi aplikasi:
- **Tab Kalkulator Probabilitas & Klasifikasi:** input seluruh X1–X6, tampilkan kontribusi koefisien, nilai z, probabilitas, prediksi klasifikasi, threshold interaktif, dan unduh hasil.
- **Tab Dataset & Koefisien:** informasi split, coding variabel kategorik, koefisien model, dan pratinjau dataset.

## Cara menjalankan

Buka terminal pada folder proyek, kemudian:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi membaca file CSV yang disertakan dalam folder ini. Tidak perlu mengubah path.

## Reproduksibilitas model

- Data: `Simulasi_1000_Nasabah_Regresi_Logistik_Biner.csv`
- Data training: 80%; data testing: 20%; stratifikasi berdasarkan Y; `random_state=42`.
- `LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000)` (regularisasi L2 standar).
- Data numerik: usia (tahun), penghasilan dan saldo (juta rupiah).
- Pekerjaan: referensi Karyawan; dummy PNS, Wiraswasta, Pensiunan.
- Memiliki Pinjaman dan Pernah Dihubungi: `Ya=1`, `Tidak=0`.
- ID nasabah tidak masuk model.
- Contoh awal: usia 45, penghasilan 10 juta, saldo 50 juta, PNS, tidak memiliki pinjaman, pernah dihubungi → probabilitas sekitar **68,11%**, klasifikasi Y=1 pada threshold 0,5.

**Catatan:** nilai koefisien dapat berbeda sedikit pada versi library/platform berbeda. Aplikasi selalu menghitung kontribusi dan probabilitas dengan koefisien penuh dari model yang benar-benar dilatih saat dijalankan.

Data merupakan simulasi untuk pembelajaran. Jangan mengartikan koefisien sebagai bukti kausal atau kinerja model di dunia nyata.
