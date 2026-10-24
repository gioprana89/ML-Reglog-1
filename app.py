"""Prototipe Bab 2: Tab kalkulator probabilitas regresi logistik biner.

Jalankan: streamlit run app.py
Dataset: simulasi 1.000 nasabah (bukan data nasabah riil).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

DATA_FILE = Path(__file__).with_name("Simulasi_1000_Nasabah_Regresi_Logistik_Biner.csv")
TARGET = "Berlangganan_Deposito"
FEATURES = [
    "Usia (tahun)",
    "Penghasilan (juta Rp)",
    "Saldo (juta Rp)",
    "Pekerjaan: PNS",
    "Pekerjaan: Wiraswasta",
    "Pekerjaan: Pensiunan",
    "Memiliki Pinjaman: Ya",
    "Pernah Dihubungi: Ya",
]
JOB_CHOICES = ["Karyawan", "PNS", "Wiraswasta", "Pensiunan"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Konversi deterministik; tidak mempelajari statistik dari data testing."""
    return pd.DataFrame(
        {
            "Usia (tahun)": pd.to_numeric(df["Usia"]),
            "Penghasilan (juta Rp)": pd.to_numeric(df["Penghasilan_Bulanan"]) / 1_000_000,
            "Saldo (juta Rp)": pd.to_numeric(df["Saldo_Tabungan"]) / 1_000_000,
            "Pekerjaan: PNS": (df["Pekerjaan"] == "PNS").astype(int),
            "Pekerjaan: Wiraswasta": (df["Pekerjaan"] == "Wiraswasta").astype(int),
            "Pekerjaan: Pensiunan": (df["Pekerjaan"] == "Pensiunan").astype(int),
            "Memiliki Pinjaman: Ya": (df["Memiliki_Pinjaman"] == "Ya").astype(int),
            "Pernah Dihubungi: Ya": (df["Pernah_Dihubungi"] == "Ya").astype(int),
        },
        index=df.index,
    )[FEATURES]


@st.cache_resource
def load_model_and_data():
    df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    X = build_features(df)
    y = df[TARGET].astype(int)
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=42, stratify=y
    )
    # Default sklearn: Logistic Regression + regularisasi L2, C=1.
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000)
    model.fit(X.iloc[idx_train], y.iloc[idx_train])
    return df, model, idx_train, idx_test


def build_nasabah_features(usia, penghasilan, saldo, pekerjaan, pinjaman, dihubungi):
    return pd.DataFrame(
        [[
            usia,
            penghasilan / 1_000_000,
            saldo / 1_000_000,
            int(pekerjaan == "PNS"),
            int(pekerjaan == "Wiraswasta"),
            int(pekerjaan == "Pensiunan"),
            int(pinjaman == "Ya"),
            int(dihubungi == "Ya"),
        ]],
        columns=FEATURES,
    )


def rupiah(n):
    return "Rp" + f"{n:,.0f}".replace(",", ".")


def main():
    st.set_page_config(
        page_title="Kalkulator Regresi Logistik Biner | Bab 2",
        page_icon="📘",
        layout="wide",
    )
    st.title("Regresi Logistik Biner — Praktikum Bab 2")
    st.write(
        "**Kasus:** Prediksi keputusan nasabah berlangganan deposito. "
        "Dataset: 1.000 nasabah **simulasi** untuk pembelajaran."
    )

    if not DATA_FILE.exists():
        st.error(f"Dataset tidak ditemukan di folder aplikasi: {DATA_FILE.name}")
        st.stop()

    df, model, train_idx, test_idx = load_model_and_data()
    koef = np.asarray(model.coef_[0], dtype=float)
    intersep = float(model.intercept_[0])

    tab_kalkulator, tab_koefisien = st.tabs(
        ["🧮 Tab Khusus: Probabilitas & Klasifikasi", "📊 Dataset & Koefisien Model"]
    )

    with tab_kalkulator:
        st.subheader("Masukkan enam karakteristik nasabah")
        st.caption(
            "Pilih nilai seluruh X1–X6. Pekerjaan memakai tiga dummy dengan kategori "
            "referensi Karyawan. Nominal rupiah otomatis diubah ke juta rupiah di dalam persamaan."
        )
        kol1, kol2, kol3 = st.columns(3)
        with kol1:
            usia = st.number_input("X1 — Usia (tahun)", min_value=18, max_value=100, value=45, step=1)
            pekerjaan = st.selectbox("X4 — Pekerjaan", JOB_CHOICES, index=1)
        with kol2:
            penghasilan = st.number_input(
                "X2 — Penghasilan Bulanan (Rp)", min_value=0, max_value=100_000_000,
                value=10_000_000, step=500_000, format="%d"
            )
            pinjaman = st.radio("X5 — Memiliki Pinjaman", ["Tidak", "Ya"], horizontal=True)
        with kol3:
            saldo = st.number_input(
                "X3 — Saldo Tabungan (Rp)", min_value=0, max_value=500_000_000,
                value=50_000_000, step=1_000_000, format="%d"
            )
            dihubungi = st.radio(
                "X6 — Pernah Dihubungi (kampanye sebelumnya)",
                ["Tidak", "Ya"], index=1, horizontal=True
            )
        st.caption(
            f"Penghasilan {rupiah(penghasilan)} = {penghasilan / 1_000_000:g} juta Rp; "
            f"Saldo {rupiah(saldo)} = {saldo / 1_000_000:g} juta Rp."
        )
        threshold = st.slider(
            "Threshold klasifikasi", min_value=0.05, max_value=0.95,
            value=0.50, step=0.05,
            help="Prediksi kelas 1 jika probabilitas ≥ threshold; selain itu kelas 0."
        )

        x_new = build_nasabah_features(usia, penghasilan, saldo, pekerjaan, pinjaman, dihubungi)
        kontribusi = koef * x_new.iloc[0].to_numpy(dtype=float)
        z = float(intersep + kontribusi.sum())
        probabilitas_manual = float(1 / (1 + np.exp(-z)))
        probabilitas_model = float(model.predict_proba(x_new)[0, 1])
        assert np.isclose(probabilitas_manual, probabilitas_model, atol=1e-10)
        pred = int(probabilitas_manual >= threshold)

        st.divider()
        colz, colp, colpred = st.columns(3)
        colz.metric("Nilai prediktor linear (z)", f"{z:.6f}")
        colp.metric("Probabilitas Y = 1", f"{100 * probabilitas_manual:.2f}%")
        colpred.metric("Hasil klasifikasi", f"Y = {pred}")
        st.progress(probabilitas_manual)
        if pred == 1:
            st.success("**Prediksi: BERLANGGANAN deposito** (Y = 1).")
        else:
            st.info("**Prediksi: TIDAK BERLANGGANAN deposito** (Y = 0).")
        st.write(
            f"Karena probabilitas {probabilitas_manual:.6f} "
            f"{'≥' if pred == 1 else '<'} threshold {threshold:.2f}, "
            f"maka kelas hasil prediksi adalah **{pred}**."
        )
        st.caption("Probabilitas bukan kepastian; hasil prediksi dapat berbeda dengan keputusan aktual nasabah.")

        st.subheader("Perhitungan rinci dari persamaan")
        st.latex(r"z = \beta_0 + \sum_{j=1}^{8} \beta_j x_j")
        detail = pd.DataFrame({
            "Variabel model": ["Konstanta"] + FEATURES,
            "Nilai X": [1.0] + x_new.iloc[0].astype(float).to_list(),
            "Koefisien β": [intersep] + koef.tolist(),
            "Kontribusi (β × X)": [intersep] + kontribusi.tolist(),
        })
        st.dataframe(
            detail.style.format({
                "Nilai X": "{:.2f}",
                "Koefisien β": "{:.6f}",
                "Kontribusi (β × X)": "{:.6f}",
            }), use_container_width=True, hide_index=True,
        )
        st.latex(rf"z = {z:.6f}")
        st.latex(rf"P(Y=1) = \frac{{1}}{{1 + e^{{-({z:.6f})}}}} = {probabilitas_manual:.6f}")
        st.caption(
            "Perhitungan menggunakan koefisien presisi penuh dari model; "
            "angka pada tabel ditampilkan setelah pembulatan enam desimal."
        )
        hasil = pd.DataFrame([{
            "Usia": usia,
            "Penghasilan_Bulanan_Rp": penghasilan,
            "Saldo_Tabungan_Rp": saldo,
            "Pekerjaan": pekerjaan,
            "Memiliki_Pinjaman": pinjaman,
            "Pernah_Dihubungi": dihubungi,
            "Nilai_z": z,
            "Probabilitas_Deposito": probabilitas_manual,
            "Threshold": threshold,
            "Prediksi_Klasifikasi": pred,
        }])
        st.download_button(
            "⬇️ Unduh hasil prediksi (CSV)",
            hasil.to_csv(index=False).encode("utf-8-sig"),
            file_name="Hasil_Prediksi_Nasabah.csv",
            mime="text/csv",
        )

    with tab_koefisien:
        st.subheader("Koefisien berasal dari dataset simulasi yang sama")
        st.write(
            f"Total **{len(df):,}** nasabah; training **{len(train_idx)}** (80%); "
            f"testing **{len(test_idx)}** (20%), dipisahkan secara *stratified* "
            "dengan `random_state=42`. Model: `LogisticRegression(C=1.0, "
            "solver='lbfgs', max_iter=5000)` (regularisasi L2 standar)."
        )
        st.dataframe(
            pd.DataFrame({
                "Variabel": ["Konstanta"] + FEATURES,
                "Koefisien": [intersep] + koef.tolist(),
            }).style.format({"Koefisien": "{:.6f}"}),
            hide_index=True, use_container_width=True,
        )
        st.markdown(
            "**Aturan coding pekerjaan:** Karyawan menjadi kategori referensi "
            "(seluruh dummy pekerjaan = 0). Untuk PNS, Wiraswasta, atau Pensiunan, "
            "hanya dummy pekerjaan terkait yang bernilai 1. "
            "**Pinjaman** dan **Pernah Dihubungi:** Ya = 1, Tidak = 0."
        )
        st.write("**20 baris pertama dataset simulasi:**")
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)
        st.caption(
            "Data ini sepenuhnya sintetis. Koefisien mencerminkan pola simulasi, "
            "bukan bukti hubungan kausal maupun pola yang berlaku di bank nyata. "
            "Fitur ID_Nasabah tidak digunakan untuk training."
        )


if __name__ == "__main__":
    main()
