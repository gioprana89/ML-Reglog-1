"""Inti analisis untuk aplikasi praktikum regresi logistik biner.

Modul ini sengaja terpisah dari antarmuka Streamlit agar perhitungan,
pembagian data, dan interpretasi koefisien dapat diuji secara mandiri.
Semua dataset dalam buku adalah simulasi untuk tujuan pembelajaran.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

ID = "ID_Nasabah"
TARGET = "Berlangganan_Deposito"
FEATURES = [
    "Usia",
    "Penghasilan_Bulanan",
    "Saldo_Tabungan",
    "Pekerjaan",
    "Memiliki_Pinjaman",
    "Pernah_Dihubungi",
]
REQUIRED_COLUMNS = [ID, *FEATURES, TARGET]
JOB_LEVELS = ["Karyawan", "PNS", "Wiraswasta", "Pensiunan"]
BINARY_LEVELS = ["Tidak", "Ya"]
POSSIBLE_TARGETS = [TARGET, "Memiliki_Pinjaman", "Pernah_Dihubungi"]

# Fixed-category encoding avoids fitting transformations on the test set.
# It also makes the coefficients interpretable in the book's units.
FEATURE_MAPPING = {
    "Usia": ["Usia (tahun)"],
    "Penghasilan_Bulanan": ["Penghasilan (juta Rp)"],
    "Saldo_Tabungan": ["Saldo (juta Rp)"],
    "Pekerjaan": [
        "Pekerjaan: PNS",
        "Pekerjaan: Wiraswasta",
        "Pekerjaan: Pensiunan",
    ],
    "Memiliki_Pinjaman": ["Memiliki Pinjaman: Ya"],
    "Pernah_Dihubungi": ["Pernah Dihubungi: Ya"],
}

LATEX_SYMBOL = {
    "Usia (tahun)": r"X_{\mathrm{usia}}",
    "Penghasilan (juta Rp)": r"X_{\mathrm{penghasilan}}",
    "Saldo (juta Rp)": r"X_{\mathrm{saldo}}",
    "Pekerjaan: PNS": r"D_{\mathrm{PNS}}",
    "Pekerjaan: Wiraswasta": r"D_{\mathrm{wiraswasta}}",
    "Pekerjaan: Pensiunan": r"D_{\mathrm{pensiunan}}",
    "Memiliki Pinjaman: Ya": r"D_{\mathrm{pinjaman}}",
    "Pernah Dihubungi: Ya": r"D_{\mathrm{dihubungi}}",
}


@dataclass
class TrainedModel:
    model: LogisticRegression
    names: list[str]
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def validate_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Periksa struktur dan nilai, tanpa mengubah baris asli dataset."""
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing_cols:
        raise ValueError("Kolom dataset tidak ditemukan: " + ", ".join(missing_cols))
    df = raw.copy()
    if df.empty:
        raise ValueError("Dataset tidak boleh kosong.")
    for col in [ID, "Pekerjaan", "Memiliki_Pinjaman", "Pernah_Dihubungi"]:
        if df[col].isna().any():
            raise ValueError(f"Ada nilai kosong pada kolom {col}.")
        df[col] = df[col].astype(str).str.strip()
        if df[col].eq("").any():
            raise ValueError(f"Ada teks kosong pada kolom {col}.")
    if df[ID].duplicated().any():
        raise ValueError("Kolom ID_Nasabah mengandung identitas ganda.")
    for col in ["Usia", "Penghasilan_Bulanan", "Saldo_Tabungan", TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Kolom {col} mengandung nilai kosong/bukan angka.")
    if not np.isfinite(df[["Usia", "Penghasilan_Bulanan", "Saldo_Tabungan", TARGET]].to_numpy(dtype=float)).all():
        raise ValueError("Dataset memiliki angka tidak terhingga pada kolom numerik.")
    if (df[["Usia", "Penghasilan_Bulanan", "Saldo_Tabungan"]] < 0).any().any():
        raise ValueError("Usia, penghasilan, dan saldo tidak boleh bernilai negatif.")
    if not (df[TARGET].isin([0, 1])).all() or df[TARGET].nunique() != 2:
        raise ValueError("Berlangganan_Deposito harus memuat dua kelas: 0 dan 1.")
    df[TARGET] = df[TARGET].astype(int)
    if not df["Pekerjaan"].isin(JOB_LEVELS).all():
        raise ValueError("Pekerjaan hanya boleh: " + ", ".join(JOB_LEVELS))
    for col in ["Memiliki_Pinjaman", "Pernah_Dihubungi"]:
        if not df[col].isin(BINARY_LEVELS).all():
            raise ValueError(f"{col} hanya boleh berisi 'Ya' atau 'Tidak'.")
    return df


def get_targets(df: pd.DataFrame) -> list[str]:
    return [t for t in POSSIBLE_TARGETS if t in df.columns and df[t].nunique() == 2]


def encode_target(df: pd.DataFrame, target: str) -> pd.Series:
    if target == TARGET:
        return df[target].astype(int)
    if target in ("Memiliki_Pinjaman", "Pernah_Dihubungi"):
        return (df[target] == "Ya").astype(int)
    raise ValueError("Variabel dependen tidak didukung. Pilih target biner yang tersedia.")


def encode_features(df: pd.DataFrame, selected: Sequence[str]) -> pd.DataFrame:
    """Ubah enam prediktor buku secara deterministik menjadi kolom numerik."""
    if not selected:
        raise ValueError("Pilih paling sedikit satu variabel independen.")
    if any(f not in FEATURES for f in selected):
        raise ValueError("Variabel X hanya dapat dipilih dari keenam prediktor buku.")
    result: dict[str, pd.Series] = {}
    for feat in FEATURES:
        if feat not in selected:
            continue
        if feat == "Usia":
            result["Usia (tahun)"] = df[feat].astype(float)
        elif feat == "Penghasilan_Bulanan":
            result["Penghasilan (juta Rp)"] = df[feat].astype(float) / 1_000_000
        elif feat == "Saldo_Tabungan":
            result["Saldo (juta Rp)"] = df[feat].astype(float) / 1_000_000
        elif feat == "Pekerjaan":
            for cat in JOB_LEVELS[1:]:
                result[f"Pekerjaan: {cat}"] = (df[feat] == cat).astype(float)
        elif feat == "Memiliki_Pinjaman":
            result["Memiliki Pinjaman: Ya"] = (df[feat] == "Ya").astype(float)
        elif feat == "Pernah_Dihubungi":
            result["Pernah Dihubungi: Ya"] = (df[feat] == "Ya").astype(float)
    return pd.DataFrame(result, index=df.index)


def split_dataset(
    df: pd.DataFrame, target: str, train_pct: int = 80, seed: int = 42, stratify: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    if not 50 <= train_pct <= 90:
        raise ValueError("Persentase training harus antara 50% dan 90%.")
    y = encode_target(df, target)
    if stratify and y.value_counts().min() < 2:
        raise ValueError("Stratified splitting membutuhkan sedikitnya dua observasi per kelas.")
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=1 - train_pct / 100,
        random_state=int(seed),
        stratify=y if stratify else None,
    )
    if y.iloc[train_idx].nunique() != 2:
        raise ValueError("Data training hanya berisi satu kelas. Aktifkan stratifikasi atau ubah seed.")
    return np.asarray(train_idx), np.asarray(test_idx)


def fit_model(
    df: pd.DataFrame,
    selected: Sequence[str],
    target: str,
    train_idx: Sequence[int],
    test_idx: Sequence[int],
) -> TrainedModel:
    X = encode_features(df, selected)
    y = encode_target(df, target)
    # Data testing sama sekali tidak dipakai dalam proses fit.
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=5000)
    x_train, x_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(x_train, y_train)
    return TrainedModel(model, X.columns.to_list(), x_train, x_test, y_train, y_test)


def predict_single(trained: TrainedModel, values: dict, selected: Sequence[str], threshold: float):
    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold harus lebih besar dari 0 dan lebih kecil dari 1.")
    row = pd.DataFrame([values])
    x = encode_features(row, selected).reindex(columns=trained.names)
    contribution = x.iloc[0].to_numpy(dtype=float) * trained.model.coef_.ravel()
    z = float(trained.model.intercept_[0] + contribution.sum())
    prob = float(expit(z))
    reference_prob = float(trained.model.predict_proba(x)[0, 1])
    if not np.isclose(prob, reference_prob, rtol=1e-10, atol=1e-10):
        raise AssertionError("Probabilitas perhitungan manual dan sklearn tidak sama.")
    return x, contribution, z, prob, int(prob >= threshold)


def evaluate_model(trained: TrainedModel, threshold: float = 0.5) -> dict:
    p = trained.model.predict_proba(trained.x_test)[:, 1]
    pred = (p >= threshold).astype(int)
    y = trained.y_test.to_numpy(dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    results = {
        "probabilities": p,
        "predictions": pred,
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
    }
    if len(np.unique(y)) == 2:
        results["fpr"], results["tpr"], results["roc_thresholds"] = roc_curve(y, p)
    return results


def coefficients_table(trained: TrainedModel) -> pd.DataFrame:
    return pd.DataFrame({
        "Variabel model": ["Konstanta"] + trained.names,
        "Koefisien β": [float(trained.model.intercept_[0])] + trained.model.coef_.ravel().tolist(),
    })


def latex_equation(trained: TrainedModel) -> str:
    """Persamaan multiline LaTeX dengan koefisien dari model aktif."""
    b0 = float(trained.model.intercept_[0])
    lines = [rf"z &= {b0:+.6f}"]
    for name, coef in zip(trained.names, trained.model.coef_.ravel()):
        symbol = LATEX_SYMBOL.get(name, r"X")
        lines.append(rf"& {float(coef):+.6f}\,{symbol}")
    return r"\begin{aligned}" + r" \\ ".join(lines) + r"\end{aligned}"
