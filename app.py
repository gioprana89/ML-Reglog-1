"""Buku Praktikum Machine Learning: Regresi Logistik Biner.

Alur buku:
 Tab 1 Bab 1: Dataset
 Tab 2 Bab 2: Pemilihan variabel X dan Y
 Tab 3 Bab 3: Pembagian data training/testing dan unduhan Excel
 Tab 4 Bab 4: Persamaan, probabilitas, dan klasifikasi nasabah
 Tab 5 Bab 5: Prediksi klasifikasi data training
 Tab 6 Bab 6: Prediksi klasifikasi data testing
 Tab 7 Bab 7: Confusion Matrix testing
 Tab 8 Bab 8: Accuracy testing
 Tab 9 Bab 9: Recall / Sensitivity testing
 Tab 10 Bab 10: Specificity testing
 Tab 11 Bab 11: Precision dan F1-score testing
 Tab 12 Bab 12: ROC Curve dan ROC-AUC testing

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
    prediction_table,
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
      div[data-testid="stTabs"] {overflow-x:auto;}
      div[data-testid="stTabs"] button {white-space: nowrap;}
      div[data-testid="stTabs"] [data-baseweb="tab-list"] {gap: 6px;}
      .metric-formula {font-size: 1.04rem; color:#A6CBF5;}
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


def display_prediction_results(result_df: pd.DataFrame, *, training: bool, meta_common: list, seed: int) -> None:
    """Tabel hasil prediksi semua observasi, plus ekspor Excel."""
    label = "Training" if training else "Testing"
    correct = int((result_df["Status_Prediksi"] == "Benar").sum())
    wrong = len(result_df) - correct
    p1, p2, p3 = st.columns(3)
    p1.metric("Jumlah observasi", f"{len(result_df):,}".replace(",", "."))
    p2.metric("Prediksi benar", str(correct))
    p3.metric("Prediksi salah", str(wrong))
    st.caption(
        "Y aktual = kondisi dalam dataset; Y prediksi berasal dari probabilitas model. "
        "Prediksi kelas 1 jika probabilitas ≥ 0.50."
    )
    preview = result_df.copy()
    st.dataframe(
        preview, use_container_width=True, hide_index=True, height=510,
        column_config={
            "Nilai_z": st.column_config.NumberColumn("Nilai z", format="%.6f"),
            "Probabilitas_Y0": st.column_config.NumberColumn("Probabilitas Y=0", format="%.6f"),
            "Probabilitas_Y1": st.column_config.NumberColumn("Probabilitas Y=1", format="%.6f"),
        },
    )
    distribution = pd.DataFrame({
        "Kategori": ["Y = 0", "Y = 1"],
        "Aktual (n)": [(result_df["Y_Aktual"] == k).sum() for k in (0, 1)],
        "Prediksi (n)": [(result_df["Y_Prediksi"] == k).sum() for k in (0, 1)],
    })
    st.subheader("Distribusi kelas aktual dan prediksi")
    st.dataframe(distribution, use_container_width=True, hide_index=True)
    st.download_button(
        f"⬇️ Unduh Hasil Prediksi {label}.xlsx",
        excel_export(
            result_df,
            tuple(meta_common + [("Jenis laporan", f"Prediksi Data {label}"),
                                 ("Threshold klasifikasi", "0.50"),
                                 ("Random seed", str(seed))]),
        ),
        file_name=f"Hasil_Prediksi_{label}_Seed_{seed}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        key=f"pred_{label.lower()}_download",
    )


def show_confusion(result: dict) -> tuple[int, int, int, int]:
    cm = result["confusion_matrix"]
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    matrix_table = pd.DataFrame(cm,
        index=["Aktual 0 (Tidak)", "Aktual 1 (Ya)"],
        columns=["Prediksi 0 (Tidak)", "Prediksi 1 (Ya)"],
    )
    st.dataframe(matrix_table, use_container_width=True)
    a, b, c, d = st.columns(4)
    a.metric("True Negative (TN)", tn)
    b.metric("False Positive (FP)", fp)
    c.metric("False Negative (FN)", fn)
    d.metric("True Positive (TP)", tp)
    return tn, fp, fn, tp


def show_metric_calc(title: str, latex: str, expression: str, value: str, interpretation: str) -> None:
    st.subheader(title)
    st.latex(latex)
    st.latex(expression)
    st.metric(title, value)
    st.info(interpretation)


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
        "**Bab 5:** Prediksi Training  \n"
        "**Bab 6:** Prediksi Testing  \n"
        "**Bab 7:** Confusion Matrix  \n"
        "**Bab 8:** Accuracy  \n"
        "**Bab 9:** Recall / Sensitivity  \n"
        "**Bab 10:** Specificity  \n"
        "**Bab 11:** Precision & F1-score  \n"
        "**Bab 12:** ROC & ROC-AUC"
    )
    st.sidebar.info(
        "Data bawaan adalah **simulasi**, bukan data nasabah bank yang sesungguhnya. "
        "Semua pengaturan berpengaruh pada model di tab berikutnya."
    )

    (tab_data, tab_vars, tab_split, tab_pred,
     tab_train_pred, tab_test_pred, tab_cm,
     tab_accuracy, tab_recall, tab_specificity, tab_precision_f1, tab_roc) = st.tabs([
        "📁 1 · Dataset",
        "🧩 2 · Variabel",
        "✂️ 3 · Splitting",
        "🧮 4 · Probabilitas",
        "📝 5 · Prediksi Training",
        "📝 6 · Prediksi Testing",
        "🔢 7 · Confusion Matrix",
        "🎯 8 · Accuracy",
        "🔎 9 · Recall",
        "🛡️ 10 · Specificity",
        "⚖️ 11 · Precision & F1",
        "📈 12 · ROC & AUC",
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
                "Pilih Berlangganan_Deposito untuk mengikuti contoh Bab 1–12."
            )
        if not selected:
            st.error("Pilih minimal satu variabel X untuk melanjutkan ke Tab 3–12.")
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

    # PREDIKSI DATA TRAINING DAN TESTING BERASAL DARI MODEL AKTIF YANG SAMA.
    # Threshold evaluasi tetap = 0.50 untuk seluruh contoh dalam buku.
    threshold_eval = 0.50
    train_predictions = prediction_table(trained, df, train_idx, subset="train", threshold=threshold_eval)
    test_predictions = prediction_table(trained, df, test_idx, subset="test", threshold=threshold_eval)
    result = evaluate_model(trained, threshold=threshold_eval)
    tn, fp, fn, tp = [int(v) for v in result["confusion_matrix"].ravel()]
    total_test = tn + fp + fn + tp

    with tab_train_pred:
        st.header("Bab 5 — Hasil Prediksi Klasifikasi Data Training")
        st.write(
            f"Model yang dibentuk menggunakan **{len(train_df)} data training** sekarang diterapkan "
            "pada data training tersebut untuk menghasilkan probabilitas dan kelas prediksi. "
            "Tabel di bawah memuat **seluruh observasi** termasuk nilai aktual, z, "
            "probabilitas Y=0/Y=1, dan prediksi berdasarkan threshold **0.50**."
        )
        st.warning(
            "Hasil pada data training dapat terlihat lebih baik karena data ini digunakan "
            "untuk membentuk model. Gunakan data testing untuk evaluasi akhir generalisasi."
        )
        display_prediction_results(train_predictions, training=True, meta_common=meta_common, seed=int(seed))

    with tab_test_pred:
        st.header("Bab 6 — Hasil Prediksi Klasifikasi Data Testing")
        st.write(
            f"Model **tidak dilatih ulang** pada {len(test_df)} data testing. "
            "Model yang sama dari Bab 4 diterapkan pada seluruh data testing, "
            "dengan threshold klasifikasi tetap sebesar **0.50**."
        )
        display_prediction_results(test_predictions, training=False, meta_common=meta_common, seed=int(seed))
        st.info("Kolom Status_Prediksi membandingkan Y_Aktual dengan Y_Prediksi. "
                "Hasil pada bab ini akan digunakan untuk evaluasi Bab 7–12.")

    with tab_cm:
        st.header("Bab 7 — Evaluasi Model: Confusion Matrix (Data Testing)")
        st.write(
            f"Confusion Matrix disusun dari **{len(test_df)} data testing** pada Bab 6, "
            "dengan threshold 0.50. Baris menunjukkan kelas aktual, sedangkan kolom "
            "menunjukkan kelas hasil prediksi."
        )
        show_confusion(result)
        st.markdown(
            f"**Interpretasi:** Sebanyak **{tn}** nasabah kategori aktual 0 diprediksi 0 (TN); "
            f"**{fp}** nasabah aktual 0 diprediksi 1 (FP); "
            f"**{fn}** nasabah aktual 1 diprediksi 0 (FN); dan "
            f"**{tp}** nasabah aktual 1 diprediksi 1 (TP). "
            f"Total = **{total_test}** observasi."
        )
        conf_long = pd.DataFrame([
            ["True Negative", "TN", tn], ["False Positive", "FP", fp],
            ["False Negative", "FN", fn], ["True Positive", "TP", tp],
        ], columns=["Komponen", "Kode", "Jumlah"])
        st.download_button(
            "⬇️ Unduh Confusion Matrix.xlsx",
            excel_export(conf_long, tuple(meta_common + [("Threshold", "0.50"), ("Bab", "7")])),
            file_name="Bab_7_Confusion_Matrix.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_confusion", use_container_width=True,
        )

    with tab_accuracy:
        st.header("Bab 8 — Evaluasi Model: Accuracy (Data Testing)")
        st.write("Accuracy mengukur proporsi seluruh prediksi yang sesuai dengan kelas aktual.")
        show_metric_calc(
            "Accuracy", r"\mathrm{Accuracy}=\frac{TP+TN}{TP+TN+FP+FN}",
            rf"=\frac{{{tp}+{tn}}}{{{tp}+{tn}+{fp}+{fn}}}=\frac{{{tp+tn}}}{{{total_test}}}",
            pct(result["accuracy"]),
            f"Dari {total_test} data testing, sebanyak {tp+tn} diprediksi dengan benar "
            f"dan {fp+fn} diprediksi salah. Accuracy perlu dilihat bersama recall dan metrik lainnya "
            "karena kedua kelas tidak memiliki jumlah yang sama."
        )

    with tab_recall:
        st.header("Bab 9 — Evaluasi Model: Recall / Sensitivity (Data Testing)")
        st.write("Recall mengukur proporsi nasabah aktual berlangganan deposito (Y=1) yang terdeteksi oleh model.")
        show_metric_calc(
            "Recall / Sensitivity", r"\mathrm{Recall}=\frac{TP}{TP+FN}",
            rf"=\frac{{{tp}}}{{{tp}+{fn}}}=\frac{{{tp}}}{{{tp+fn}}}",
            pct(result["recall"]),
            f"Dari {tp+fn} nasabah yang benar-benar berlangganan deposito, "
            f"hanya {tp} diidentifikasi sebagai kelas positif dan {fn} terlewat (false negative)."
        )

    with tab_specificity:
        st.header("Bab 10 — Evaluasi Model: Specificity (Data Testing)")
        st.write("Specificity mengukur proporsi nasabah aktual tidak berlangganan (Y=0) yang dikenali model.")
        show_metric_calc(
            "Specificity", r"\mathrm{Specificity}=\frac{TN}{TN+FP}",
            rf"=\frac{{{tn}}}{{{tn}+{fp}}}=\frac{{{tn}}}{{{tn+fp}}}",
            pct(result["specificity"]),
            f"Dari {tn+fp} nasabah yang aktual tidak berlangganan, "
            f"sebanyak {tn} diprediksi dengan benar dan {fp} keliru diprediksi berlangganan."
        )

    with tab_precision_f1:
        st.header("Bab 11 — Evaluasi Model: Precision dan F1-score (Data Testing)")
        st.write("Precision menilai ketepatan prediksi positif. F1-score menggabungkan precision dan recall.")
        show_metric_calc(
            "Precision", r"\mathrm{Precision}=\frac{TP}{TP+FP}",
            rf"=\frac{{{tp}}}{{{tp}+{fp}}}=\frac{{{tp}}}{{{tp+fp}}}",
            pct(result["precision"]),
            f"Dari {tp+fp} nasabah yang diprediksi akan berlangganan, "
            f"sebanyak {tp} benar-benar berlangganan dan {fp} tidak."
        )
        st.divider()
        show_metric_calc(
            "F1-score", r"F_1=\frac{2\,\mathrm{Precision}\,\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}=\frac{2TP}{2TP+FP+FN}",
            rf"=\frac{{2({tp})}}{{2({tp})+{fp}+{fn}}}=\frac{{{2*tp}}}{{{2*tp+fp+fn}}}",
            f'{result["f1"]:.4f}',
            "F1-score adalah rata-rata harmonik precision dan recall; "
            "nilainya rendah jika salah satu dari keduanya rendah."
        )

    with tab_roc:
        st.header("Bab 12 — Evaluasi Model: ROC Curve dan ROC-AUC (Data Testing)")
        st.write(
            "Kurva ROC menggambarkan hubungan True Positive Rate (Recall) dan False Positive Rate "
            "pada berbagai threshold. ROC-AUC dihitung dari **probabilitas kelas 1**, "
            "bukan hanya hasil klasifikasi pada threshold 0.50."
        )
        if "fpr" not in result:
            st.error("ROC tidak tersedia karena data testing tidak memiliki kedua kelas.")
        else:
            st.metric("ROC-AUC pada data testing", f'{result["roc_auc"]:.4f}')
            fig, ax = plt.subplots(figsize=(8.2, 4.8))
            fig.patch.set_facecolor("#0B1220")
            ax.set_facecolor("#172239")
            ax.plot(result["fpr"], result["tpr"], linewidth=2.4,
                    color="#63B3FF", label=f'ROC (AUC = {result["roc_auc"]:.4f})')
            ax.plot([0, 1], [0, 1], linestyle="--", color="#AAB9CC", label="Acuan acak (AUC = 0.5)")
            false_positive_rate = fp / (fp + tn) if (fp + tn) else float("nan")
            true_positive_rate = tp / (tp + fn) if (tp + fn) else float("nan")
            ax.scatter([false_positive_rate], [true_positive_rate],
                       s=90, marker="o", color="#F9BF6B", zorder=3,
                       label="Titik operasi threshold 0.50")
            ax.set(xlabel="False Positive Rate (1 - Specificity)",
                   ylabel="True Positive Rate (Recall)", xlim=(0, 1), ylim=(0, 1))
            ax.tick_params(colors="#D8E6F6")
            ax.xaxis.label.set_color("#E6EDF7")
            ax.yaxis.label.set_color("#E6EDF7")
            for spine in ax.spines.values():
                spine.set_color("#405471")
            ax.legend(loc="lower right", facecolor="#172239",
                      edgecolor="#405471", labelcolor="#E6EDF7")
            ax.grid(alpha=0.22, color="#899BB6")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
            st.write(
                f"**Titik threshold 0.50:** FPR = {false_positive_rate:.4f} "
                f"(1 − specificity), TPR = {true_positive_rate:.4f} (recall). "
                "AUC mendekati 1 menunjukkan kemampuan pemeringkatan kelas yang semakin baik; "
                "0.5 setara dengan pemeringkatan acak."
            )
            roc_data = pd.DataFrame({
                "False_Positive_Rate": result["fpr"],
                "True_Positive_Rate": result["tpr"],
                "Threshold_ROC": result["roc_thresholds"],
            })
            with st.expander("Lihat koordinat kurva ROC", expanded=False):
                st.dataframe(roc_data, hide_index=True, use_container_width=True)
            st.download_button(
                "⬇️ Unduh Titik ROC.xlsx",
                excel_export(roc_data.replace([np.inf, -np.inf], np.nan), tuple(meta_common + [
                    ("Bab", "12"), ("ROC-AUC", f'{result["roc_auc"]:.9f}')
                ])),
                file_name="Bab_12_ROC_Curve_AUC.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="dl_roc",
            )
        st.warning(
            "Jangan memilih threshold, prediktor, atau hyperparameter dengan melihat hasil testing "
            "berulang kali, lalu mengklaim skor tersebut sebagai evaluasi final independen. "
            "Untuk optimasi gunakan validation split atau cross-validation hanya di data training."
        )

    st.divider()
    st.caption(
        "Buku Praktikum Machine Learning — Regresi Logistik Biner · "
        "Data simulasi untuk pembelajaran, bukan data nasabah bank sebenarnya."
    )


if __name__ == "__main__":
    main()
