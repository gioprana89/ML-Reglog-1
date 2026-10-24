"""Buku Praktikum Machine Learning: Regresi Logistik Biner.

Alur buku:
 Tab 1 Bab 1: Dataset
 Tab 2 Bab 2: Pemilihan variabel X dan Y
 Tab 3 Bab 3: Pembagian data training/testing dan unduhan Excel
 Tab 4 Bab 4: Persamaan, probabilitas, dan klasifikasi nasabah
 Tab 5 Bab 5: Evaluasi klasifikasi pada data testing

Jalankan: streamlit run app.py
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ml_core import (
    FEATURES,
    FEATURE_MAPPING,
    ID,
    JOB_LEVELS,
    TARGET,
    coefficients_table,
    encode_target,
    evaluate_model,
    fit_model,
    get_targets,
    latex_equation,
    predict_single,
    split_dataset,
    validate_dataset,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = BASE_DIR / "Simulasi_1000_Nasabah_Regresi_Logistik_Biner.csv"
DATA_XLSX = BASE_DIR / "Simulasi_1000_Nasabah_Regresi_Logistik_Biner.xlsx"

st.set_page_config(
    page_title="Machine Learning | Regresi Logistik Biner",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      /* Tema utama didefinisikan di .streamlit/config.toml.
         Warna khusus berikut selaras dengan dark mode. */
      .block-container {padding-top: 2rem; max-width: 1380px;}
      div[data-testid="stMetric"] {
        background: #172239;
        border: 1px solid #2A3B56;
        padding: 14px 16px;
        border-radius: 12px;
      }
      div[data-testid="stMetricLabel"] {color: #B6C5DA;}
      div[data-testid="stMetricValue"] {color: #F2F6FF;}
      div[data-testid="stTabs"] button {font-weight: 600;}
      .hero-caption {color: #ACBCD2; font-size: 0.98rem; margin-top: -8px; margin-bottom: 18px;}
      .step-label {color: #8FC5FF; letter-spacing: .1em; font-size: .78rem; font-weight: 800;}
      div[data-testid="stDownloadButton"] button {border-radius: 9px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def rupiah(v: int | float) -> str:
    return "Rp" + f"{v:,.0f}".replace(",", ".")


def pct(v: float) -> str:
    return f"{v * 100:.2f}%"


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return validate_dataset(pd.read_csv(DATA_CSV, encoding="utf-8-sig"))


def load_uploaded_data(upload) -> pd.DataFrame:
    if upload.name.lower().endswith(".csv"):
        raw = pd.read_csv(upload, encoding="utf-8-sig")
    else:
        raw = pd.read_excel(upload, sheet_name=0)
    return validate_dataset(raw)


@st.cache_data(show_spinner=False)
def excel_export(df: pd.DataFrame, info: tuple[tuple[str, str], ...]) -> bytes:
    """Laporan xlsx berformat rapi; data dan metadata pemisahan."""
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        df.to_excel(writer, sheet_name="Data", index=False)
        meta = pd.DataFrame(info, columns=["Parameter", "Nilai"])
        meta.to_excel(writer, sheet_name="Informasi", index=False)
        workbook = writer.book
        header = workbook.add_format({
            "bold": True, "font_color": "#FFFFFF", "bg_color": "#1B64B7",
            "border": 0, "align": "left", "valign": "vcenter",
        })
        money = workbook.add_format({"num_format": '#,##0', "align": "right"})
        for sheet_name, data in [("Data", df), ("Informasi", meta)]:
            sheet = writer.sheets[sheet_name]
            sheet.freeze_panes(1, 0)
            sheet.set_row(0, 27)
            for pos, name in enumerate(data.columns):
                sheet.write(0, pos, name, header)
                width = max(14, min(29, len(str(name)) + 3))
                if sheet_name == "Informasi":
                    width = 27 if pos == 0 else 75
                if sheet_name == "Data" and name in ("Penghasilan_Bulanan", "Saldo_Tabungan"):
                    sheet.set_column(pos, pos, width, money)
                else:
                    sheet.set_column(pos, pos, width)
            if not data.empty:
                sheet.autofilter(0, 0, len(data), len(data.columns) - 1)
    return bio.getvalue()


def render_dataset(data: pd.DataFrame, height: int = 400) -> None:
    cols = {
        "Penghasilan_Bulanan": st.column_config.NumberColumn("Penghasilan Bulanan (Rp)", format="%d"),
        "Saldo_Tabungan": st.column_config.NumberColumn("Saldo Tabungan (Rp)", format="%d"),
    }
    st.dataframe(data, hide_index=True, use_container_width=True, height=height, column_config=cols)


def count_comparison(df: pd.DataFrame, target: str, train_idx, test_idx) -> pd.DataFrame:
    y = encode_target(df, target)
    train_count = y.iloc[train_idx].value_counts().reindex([0, 1], fill_value=0)
    test_count = y.iloc[test_idx].value_counts().reindex([0, 1], fill_value=0)
    return pd.DataFrame({
        "Kategori Y": ["0 (Tidak)", "1 (Ya)"],
        "Training (n)": train_count.to_numpy(),
        "Training (%)": (100 * train_count.to_numpy() / len(train_idx)).round(2),
        "Testing (n)": test_count.to_numpy(),
        "Testing (%)": (100 * test_count.to_numpy() / len(test_idx)).round(2),
    })


def show_value_controls(selected_features: list[str]) -> dict:
    """Hanya menampilkan variabel X yang dipilih sebelumnya."""
    values = {
        "Usia": 45,
        "Penghasilan_Bulanan": 10_000_000,
        "Saldo_Tabungan": 50_000_000,
        "Pekerjaan": "PNS",
        "Memiliki_Pinjaman": "Tidak",
        "Pernah_Dihubungi": "Ya",
    }
    left, right = st.columns(2)
    with left:
        if "Usia" in selected_features:
            values["Usia"] = st.number_input("X1 — Usia (tahun)", min_value=0, max_value=110, value=45, step=1)
        if "Penghasilan_Bulanan" in selected_features:
            values["Penghasilan_Bulanan"] = st.number_input(
                "X2 — Penghasilan Bulanan (Rp)", min_value=0, max_value=1_000_000_000,
                value=10_000_000, step=500_000, format="%d",
            )
        if "Saldo_Tabungan" in selected_features:
            values["Saldo_Tabungan"] = st.number_input(
                "X3 — Saldo Tabungan (Rp)", min_value=0, max_value=5_000_000_000,
                value=50_000_000, step=1_000_000, format="%d",
            )
    with right:
        if "Pekerjaan" in selected_features:
            values["Pekerjaan"] = st.selectbox("X4 — Pekerjaan", JOB_LEVELS, index=1)
        if "Memiliki_Pinjaman" in selected_features:
            values["Memiliki_Pinjaman"] = st.radio(
                "X5 — Memiliki Pinjaman", ["Tidak", "Ya"], horizontal=True,
            )
        if "Pernah_Dihubungi" in selected_features:
            values["Pernah_Dihubungi"] = st.radio(
                "X6 — Pernah Dihubungi (kampanye sebelumnya)",
                ["Tidak", "Ya"], index=1, horizontal=True,
            )
    if "Penghasilan_Bulanan" in selected_features or "Saldo_Tabungan" in selected_features:
        st.caption(
            f"Nilai yang dimasukkan: penghasilan {rupiah(values['Penghasilan_Bulanan'])}, "
            f"saldo {rupiah(values['Saldo_Tabungan'])}. Dalam persamaan, nominal dibagi 1.000.000 "
            "agar dinyatakan dalam juta rupiah."
        )
    return values


def main():
    st.markdown(
        '<div class="step-label">BUKU PRAKTIKUM · MACHINE LEARNING</div>',
        unsafe_allow_html=True,
    )
    st.title("Machine Learning: Regresi Logistik Biner")
    st.markdown(
        '<div class="hero-caption">Praktik klasifikasi biner menggunakan Python Streamlit · '
        'Studi kasus: prediksi nasabah berlangganan deposito berjangka · '
        'Dataset simulasi 1.000 nasabah</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.header("📘 Panduan Praktikum")
    st.sidebar.markdown(
        "**Bab 1:** Mengenal Dataset  \n"
        "**Bab 2:** Pemilihan Variabel  \n"
        "**Bab 3:** Data Splitting  \n"
        "**Bab 4:** Probabilitas & Klasifikasi  \n"
        "**Bab 5:** Evaluasi Model"
    )
    st.sidebar.info(
        "Data bawaan adalah **simulasi**, bukan data nasabah bank yang sesungguhnya. "
        "Semua pengaturan berpengaruh pada model di tab berikutnya."
    )

    tab_data, tab_vars, tab_split, tab_pred, tab_eval = st.tabs([
        "📁 1 · Dataset",
        "🧩 2 · Pemilihan Variabel",
        "✂️ 3 · Data Splitting",
        "🧮 4 · Probabilitas & Klasifikasi",
        "📊 5 · Evaluasi Model",
    ])

    # TAB 1: BACA DAN TAMPILKAN DATA
    with tab_data:
        st.header("Bab 1 — Mengenal Dataset")
        st.write("Dataset bawaan berisi 1.000 nasabah dengan 6 prediktor, 1 target, dan 1 ID.")
        with st.expander("Unggah dataset sendiri (opsional)", expanded=False):
            uploaded = st.file_uploader(
                "Pilih Excel (.xlsx) atau CSV dengan kolom yang sama seperti data bawaan",
                type=["csv", "xlsx"], key="file_nasabah",
                help="Untuk buku, gunakan dataset bawaan agar hasil sama dengan contoh di naskah.",
            )
            st.caption("Unggahan harus memakai delapan nama kolom dan kategori yang sama seperti dataset buku.")
        try:
            if uploaded is None:
                if not DATA_CSV.exists():
                    st.error(f"File bawaan tidak ditemukan: {DATA_CSV.name}")
                    st.stop()
                df = load_default_data()
            else:
                df = load_uploaded_data(uploaded)
        except (ValueError, OSError, UnicodeError, ImportError) as exc:
            st.error(f"Dataset tidak dapat dibaca: {exc}")
            st.stop()

        st.caption("Sumber data: " + ("File yang diunggah" if uploaded else "Dataset simulasi bawaan aplikasi"))
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Jumlah nasabah", f"{len(df):,}".replace(",", "."))
        m2.metric("Jumlah kolom", str(len(df.columns)))
        m3.metric("Tidak berlangganan (0)", str((df[TARGET] == 0).sum()))
        m4.metric("Berlangganan (1)", str((df[TARGET] == 1).sum()))
        st.subheader("Dataset nasabah")
        st.caption("Tabel dapat digulir, dicari, dan diurutkan. Semua baris tersedia di bawah.")
        render_dataset(df, height=420)
        st.subheader("Kamus variabel")
        schema = pd.DataFrame([
            ["ID_Nasabah", "ID", "Teks", "Pengenal unik; tidak masuk model"],
            ["Usia", "X1", "Numerik", "Usia dalam tahun"],
            ["Penghasilan_Bulanan", "X2", "Numerik", "Penghasilan per bulan (Rp)"],
            ["Saldo_Tabungan", "X3", "Numerik", "Saldo tabungan (Rp)"],
            ["Pekerjaan", "X4", "Kategorik", "Karyawan / PNS / Wiraswasta / Pensiunan"],
            ["Memiliki_Pinjaman", "X5", "Kategorik biner", "Ya / Tidak"],
            ["Pernah_Dihubungi", "X6", "Kategorik biner", "Kontak pada kampanye sebelumnya"],
            ["Berlangganan_Deposito", "Y", "Biner (0/1)", "Target prediksi"],
        ], columns=["Nama kolom", "Peran", "Jenis data", "Keterangan"])
        st.dataframe(schema, use_container_width=True, hide_index=True)
        if uploaded is None and DATA_XLSX.exists():
            st.download_button(
                "⬇️ Unduh dataset asli (Excel)", DATA_XLSX.read_bytes(),
                file_name=DATA_XLSX.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_dataset_asli",
            )
        st.download_button(
            "⬇️ Unduh dataset aktif (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
            file_name="Dataset_Nasabah_Aktif.csv", mime="text/csv", key="download_dataset_csv",
        )

    # TAB 2: PILIH VARIABEL DEPENDEN DAN INDEPENDEN
    with tab_vars:
        st.header("Bab 2 — Pemilihan Variabel")
        st.write("Tentukan variabel dependen (Y) yang memiliki dua kelas dan prediktor independen (X).")
        targets = get_targets(df)
        target = st.selectbox(
            "Variabel dependen (Y)", targets,
            index=targets.index(TARGET) if TARGET in targets else 0,
            help="Buku menggunakan Berlangganan_Deposito (0 = Tidak; 1 = Ya).",
        )
        x_options = [x for x in FEATURES if x != target]
        selected = st.multiselect(
            "Variabel independen (X) — boleh memilih satu atau beberapa",
            options=x_options, default=x_options,
            help="ID_Nasabah tidak dapat dipilih karena merupakan identifier.",
        )
        if target != TARGET:
            st.warning(
                "Anda mengubah Y dari kasus utama buku. Model kini memprediksi variabel biner lain. "
                "Pilih Berlangganan_Deposito untuk mengikuti contoh Bab 1–5."
            )
        if not selected:
            st.error("Pilih minimal satu variabel X untuk melanjutkan ke Tab 3–5.")
            st.stop()
        st.success(f"Y: **{target}** · X: **{len(selected)} variabel asli**")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Variabel dependen**")
            y = encode_target(df, target)
            y_summary = pd.DataFrame({
                "Kategori": ["0 — Tidak", "1 — Ya"],
                "Jumlah": [(y == 0).sum(), (y == 1).sum()],
                "Persentase": [pct((y == 0).mean()), pct((y == 1).mean())],
            })
            st.dataframe(y_summary, hide_index=True, use_container_width=True)
        with col_b:
            st.markdown("**Variabel independen terpilih**")
            x_rows = []
            for name in selected:
                x_rows.append({"Variabel asli": name, "Jumlah kolom model": len(FEATURE_MAPPING[name])})
            st.dataframe(pd.DataFrame(x_rows), hide_index=True, use_container_width=True)
        st.info(
            "**Catatan encoding:** Pekerjaan menggunakan tiga dummy dengan Karyawan sebagai "
            "referensi. Ya = 1, Tidak = 0. Penghasilan dan saldo dinyatakan dalam juta rupiah "
            "pada persamaan. Transformasi ini ditentukan terlebih dahulu dan tidak mempelajari "
            "informasi apa pun dari data testing."
        )

    # TAB 3: DATA SPLITTING
    with tab_split:
        st.header("Bab 3 — Pembagian Data Training dan Testing")
        st.write("Tentukan proporsi data, random seed, dan apakah stratifikasi perlu digunakan.")
        setting1, setting2, setting3 = st.columns([2, 1, 1])
        with setting1:
            train_pct = st.slider("Persentase data training", min_value=50, max_value=90, value=80, step=5)
        with setting2:
            seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
        with setting3:
            stratified = st.checkbox(
                "Stratified splitting", value=True,
                help="Mempertahankan proporsi kelas Y sedekat mungkin pada kedua kelompok.",
            )
        try:
            train_idx, test_idx = split_dataset(df, target, train_pct, int(seed), stratified)
        except ValueError as exc:
            st.error(f"Pembagian data gagal: {exc}")
            st.stop()
        train_df = df.iloc[train_idx].reset_index(drop=True).copy()
        test_df = df.iloc[test_idx].reset_index(drop=True).copy()
        a, b, c = st.columns(3)
        a.metric("Total observasi", f"{len(df):,}".replace(",", "."))
        b.metric(f"Training ({train_pct}%)", str(len(train_df)))
        c.metric(f"Testing ({100 - train_pct}%)", str(len(test_df)))
        st.subheader("Distribusi kelas pada data hasil splitting")
        st.dataframe(count_comparison(df, target, train_idx, test_idx), use_container_width=True, hide_index=True)
        st.caption(
            "Seed yang sama menghasilkan pembagian baris yang sama, selama dataset, target, proporsi, "
            "dan pengaturan stratifikasi tetap sama."
        )
        meta_common = [
            ("Dataset", "Simulasi nasabah bank — untuk pembelajaran"),
            ("Jumlah observasi asal", str(len(df))),
            ("Variabel dependen (Y)", target),
            ("Variabel independen (X)", ", ".join(selected)),
            ("Data training (%)", str(train_pct)),
            ("Data testing (%)", str(100-train_pct)),
            ("Random seed", str(seed)),
            ("Stratified splitting", "Ya" if stratified else "Tidak"),
            ("Keterangan", "ID disertakan untuk penelusuran dan tidak menjadi prediktor"),
        ]
        dl1, dl2 = st.columns(2)
        with dl1:
            st.markdown("#### Data Training")
            st.write(f"Seluruh **{len(train_df)}** baris hasil pengacakan tersedia di tabel dan Excel.")
            with st.expander("Lihat tabel data training", expanded=False):
                render_dataset(train_df)
            st.download_button(
                "⬇️ Unduh Data Training.xlsx",
                excel_export(train_df, tuple(meta_common + [("Jenis data", "Training")])),
                file_name=f"Data_Training_{train_pct}persen_seed{seed}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="export_train",
            )
        with dl2:
            st.markdown("#### Data Testing")
            st.write(f"Seluruh **{len(test_df)}** baris hasil pengacakan tersedia di tabel dan Excel.")
            with st.expander("Lihat tabel data testing", expanded=False):
                render_dataset(test_df)
            st.download_button(
                "⬇️ Unduh Data Testing.xlsx",
                excel_export(test_df, tuple(meta_common + [("Jenis data", "Testing")])),
                file_name=f"Data_Testing_{100-train_pct}persen_seed{seed}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="export_test",
            )
        st.info("Data training dipakai untuk memperkirakan koefisien model. Data testing tidak digunakan untuk proses training.")

    # MODEL SELALU DIBANGUN DARI PILIHAN X/Y DAN PEMBAGIAN TERKINI
    try:
        trained = fit_model(df, selected, target, train_idx, test_idx)
    except ValueError as exc:
        with tab_pred:
            st.error(f"Model belum dapat dilatih: {exc}")
        st.stop()

    # TAB 4: PERSAMAAN DAN PREDIKSI NASABAH BARU
    with tab_pred:
        st.header("Bab 4 — Probabilitas dan Prediksi Klasifikasi")
        st.write(
            f"Koefisien di bawah berasal dari **{len(train_df)} data training** pada Tab 3 "
            f"(seed **{seed}**, stratifikasi **{'aktif' if stratified else 'nonaktif'}**). "
            "Mengubah pengaturan sebelumnya akan membentuk model baru."
        )
        st.subheader("A. Koefisien model hasil training")
        coeff = coefficients_table(trained)
        st.dataframe(
            coeff, hide_index=True, use_container_width=True,
            column_config={"Koefisien β": st.column_config.NumberColumn("Koefisien β", format="%.9f")},
        )
        st.caption(
            "Model: LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000), "
            "regularisasi L2. Nominal penghasilan dan saldo dikonversi ke juta rupiah, tanpa "
            "standardisasi tambahan. Karyawan adalah kategori referensi Pekerjaan."
        )
        st.subheader("B. Persamaan regresi logistik biner")
        st.latex(latex_equation(trained))
        st.latex(r"P(Y=1\mid X)=\frac{1}{1+e^{-z}}")
        st.caption(
            "Koefisien persamaan ditampilkan dengan 6 desimal; kalkulator memakai seluruh "
            "presisi koefisien model untuk menjaga keakuratan hasil."
        )
        st.subheader("C. Kalkulator probabilitas nasabah baru")
        st.write("Masukkan karakteristik nasabah sesuai variabel X terpilih pada Tab 2.")
        inputs = show_value_controls(selected)
        threshold = st.slider(
            "Classification threshold untuk nasabah baru",
            min_value=0.05, max_value=0.95, step=0.01, value=0.50,
            help="Prediksi kelas 1 apabila probabilitas ≥ threshold; kelas 0 bila lebih kecil.",
        )
        x_one, contribution, z, prob, yhat = predict_single(trained, inputs, selected, threshold)
        st.divider()
        p1, p2, p3 = st.columns(3)
        p1.metric("Prediktor linear (z)", f"{z:.6f}")
        p2.metric("Probabilitas kelas 1", pct(prob))
        p3.metric("Prediksi klasifikasi", f"Y = {yhat}")
        st.progress(prob)
        if yhat == 1:
            st.success(f"**Prediksi: YA** — {target} (Y = 1).")
        else:
            st.info(f"**Prediksi: TIDAK** — {target} (Y = 0).")
        st.write(
            f"Karena probabilitas **{prob:.6f}** {'≥' if yhat else '<'} threshold "
            f"**{threshold:.2f}**, keputusan klasifikasi adalah **{yhat}**."
        )
        st.caption("Probabilitas merupakan keluaran model, bukan kepastian keputusan aktual nasabah.")
        st.subheader("D. Perhitungan rinci kontribusi koefisien")
        detail = pd.DataFrame({
            "Variabel model": ["Konstanta"] + trained.names,
            "Nilai X": [1.0] + x_one.iloc[0].to_list(),
            "Koefisien β": [float(trained.model.intercept_[0])] + trained.model.coef_.ravel().tolist(),
            "Kontribusi β × X": [float(trained.model.intercept_[0])] + contribution.tolist(),
        })
        st.dataframe(
            detail, hide_index=True, use_container_width=True,
            column_config={
                "Nilai X": st.column_config.NumberColumn("Nilai X", format="%.2f"),
                "Koefisien β": st.column_config.NumberColumn("Koefisien β", format="%.9f"),
                "Kontribusi β × X": st.column_config.NumberColumn("Kontribusi β × X", format="%.9f"),
            },
        )
        st.latex(rf"z=\sum(\beta_j X_j)={z:.9f}")
        st.latex(rf"P(Y=1)=\frac{{1}}{{1+e^{{-({z:.9f})}}}}={prob:.9f}")
        export_one = pd.DataFrame([{
            **{k: inputs[k] for k in FEATURES if k in selected},
            "Target": target,
            "Persentase_Training": train_pct,
            "Random_Seed": int(seed),
            "Nilai_z": z,
            "Probabilitas_Y1": prob,
            "Threshold": threshold,
            "Prediksi_Y": yhat,
        }])
        ex1, ex2 = st.columns(2)
        with ex1:
            st.download_button(
                "⬇️ Unduh hasil perhitungan (Excel)",
                excel_export(export_one, tuple(meta_common + [("Jenis data", "Prediksi nasabah baru")])),
                file_name="Hasil_Probabilitas_Klasifikasi.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="export_prediction",
            )
        with ex2:
            st.download_button(
                "⬇️ Unduh tabel koefisien (CSV)",
                coeff.to_csv(index=False).encode("utf-8-sig"),
                file_name="Koefisien_Model_Logistik.csv", mime="text/csv",
                use_container_width=True,
                key="export_coef",
            )

    # TAB 5: EVALUASI SEBELUM MENGGUNAKAN MODEL SECARA OPERASIONAL
    with tab_eval:
        st.header("Bab 5 — Evaluasi Model pada Data Testing")
        st.write(
            f"Evaluasi dilakukan pada **{len(test_idx)} data testing**, yang tidak digunakan "
            "untuk membentuk koefisien. Untuk menjaga kesederhanaan praktikum, evaluasi memakai "
            "threshold tetap sebesar **0.50**."
        )
        result = evaluate_model(trained, threshold=0.50)
        m = st.columns(3)
        m[0].metric("Accuracy", pct(result["accuracy"]))
        m[1].metric("Precision", pct(result["precision"]))
        m[2].metric("Recall / Sensitivity", pct(result["recall"]))
        n = st.columns(3)
        n[0].metric("F1-score", f'{result["f1"]:.4f}')
        n[1].metric("Specificity", pct(result["specificity"]))
        n[2].metric("ROC-AUC", f'{result["roc_auc"]:.4f}')

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion Matrix")
            cm = result["confusion_matrix"]
            matrix_table = pd.DataFrame(
                cm,
                index=["Aktual 0", "Aktual 1"],
                columns=["Prediksi 0", "Prediksi 1"],
            )
            st.dataframe(matrix_table, use_container_width=True)
            st.caption(
                f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}. "
                "Baris menunjukkan kelas aktual dan kolom menunjukkan hasil prediksi."
            )
        with right:
            st.subheader("Kurva ROC")
            if "fpr" in result:
                fig, ax = plt.subplots(figsize=(5.8, 3.8))
                fig.patch.set_facecolor("#0B1220")
                ax.set_facecolor("#172239")
                ax.plot(
                    result["fpr"], result["tpr"], linewidth=2.2,
                    color="#63B3FF", label=f'AUC = {result["roc_auc"]:.4f}',
                )
                ax.plot(
                    [0, 1], [0, 1], linestyle="--", color="#AAB9CC",
                    label="Klasifikasi acak",
                )
                ax.set(
                    xlabel="False Positive Rate", ylabel="True Positive Rate",
                    xlim=(0, 1), ylim=(0, 1),
                )
                ax.tick_params(colors="#D8E6F6")
                ax.xaxis.label.set_color("#E6EDF7")
                ax.yaxis.label.set_color("#E6EDF7")
                for spine in ax.spines.values():
                    spine.set_color("#405471")
                ax.legend(
                    loc="lower right", facecolor="#172239",
                    edgecolor="#405471", labelcolor="#E6EDF7",
                )
                ax.grid(alpha=0.22, color="#899BB6")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.warning("ROC-AUC tidak dapat dihitung karena kelas pada data testing tidak lengkap.")
        st.subheader("Hasil prediksi pada seluruh data testing")
        eval_rows = test_df.copy()
        eval_rows["Probabilitas_Y1"] = result["probabilities"]
        eval_rows["Prediksi_Y"] = result["predictions"]
        st.dataframe(
            eval_rows, use_container_width=True, hide_index=True, height=340,
            column_config={"Probabilitas_Y1": st.column_config.NumberColumn("Probabilitas Y=1", format="%.6f")},
        )
        st.download_button(
            "⬇️ Unduh hasil evaluasi testing (Excel)",
            excel_export(eval_rows, tuple(meta_common + [("Jenis data", "Prediksi testing — threshold 0.50")])),
            file_name="Evaluasi_Data_Testing.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="export_eval",
        )
        st.info(
            "**Catatan metodologis:** model ini dilatih pada data training saja. "
            "Jangan memilih threshold, fitur, atau hyperparameter berdasarkan skor data testing "
            "berulang-ulang lalu melaporkan skor itu sebagai evaluasi final yang independen. "
            "Untuk optimasi model, gunakan validasi terpisah atau cross-validation dalam data training."
        )

    st.divider()
    st.caption(
        "Buku Praktikum Machine Learning — Regresi Logistik Biner · "
        "Data simulasi untuk pembelajaran, bukan data nasabah bank sebenarnya."
    )


if __name__ == "__main__":
    main()
