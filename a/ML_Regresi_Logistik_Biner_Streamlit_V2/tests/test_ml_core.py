from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml_core import (
    FEATURES, TARGET, encode_features, encode_target, validate_dataset,
    split_dataset, fit_model, predict_single, evaluate_model,
    coefficients_table, latex_equation,
)

DATA = Path(__file__).resolve().parents[1] / "Simulasi_1000_Nasabah_Regresi_Logistik_Biner.csv"


@pytest.fixture(scope="module")
def dataset():
    return validate_dataset(pd.read_csv(DATA, encoding="utf-8-sig"))


def test_dataset_schema_and_distribution(dataset):
    assert dataset.shape == (1000, 8)
    assert dataset["ID_Nasabah"].is_unique
    assert int((dataset[TARGET] == 1).sum()) == 314
    assert int((dataset[TARGET] == 0).sum()) == 686


def test_split_size_stratification_and_seed(dataset):
    train, test = split_dataset(dataset, TARGET, 80, 42, True)
    train2, test2 = split_dataset(dataset, TARGET, 80, 42, True)
    assert len(train) == 800 and len(test) == 200
    assert set(train).isdisjoint(set(test))
    assert set(train).union(set(test)) == set(range(1000))
    assert np.array_equal(train, train2)
    assert np.array_equal(test, test2)
    assert int((dataset[TARGET].iloc[train] == 1).sum()) == 251
    assert int((dataset[TARGET].iloc[test] == 1).sum()) == 63
    train3, _ = split_dataset(dataset, TARGET, 80, 100, True)
    assert not np.array_equal(train, train3)


def test_encoding_and_excluding_id(dataset):
    X = encode_features(dataset, FEATURES)
    assert X.shape == (1000, 8)
    assert "ID_Nasabah" not in X.columns
    assert (X.select_dtypes(include=["number"]).notna()).all().all()
    assert set(X["Pekerjaan: PNS"].unique()) <= {0.0, 1.0}


def test_probability_equal_predict_proba(dataset):
    train, test = split_dataset(dataset, TARGET, 80, 42, True)
    fitted = fit_model(dataset, FEATURES, TARGET, train, test)
    demo = dict(
        Usia=45, Penghasilan_Bulanan=10_000_000, Saldo_Tabungan=50_000_000,
        Pekerjaan="PNS", Memiliki_Pinjaman="Tidak", Pernah_Dihubungi="Ya",
    )
    X, contr, z, p, cls = predict_single(fitted, demo, FEATURES, 0.5)
    assert cls == 1
    assert p == pytest.approx(0.6811105373214634, abs=1e-10)
    assert z == pytest.approx(0.7588800784057126, abs=1e-10)
    assert float(fitted.model.predict_proba(X)[0, 1]) == pytest.approx(p)
    assert len(contr) == 8
    assert len(coefficients_table(fitted)) == 9
    assert "\\begin{aligned}" in latex_equation(fitted)
    _, _, _, _, cls_high = predict_single(fitted, demo, FEATURES, 0.7)
    assert cls_high == 0


def test_select_features_and_y_other_than_deposit(dataset):
    train, test = split_dataset(dataset, "Memiliki_Pinjaman", 75, 7, True)
    fitted = fit_model(dataset, ["Usia", "Pekerjaan", "Saldo_Tabungan"], "Memiliki_Pinjaman", train, test)
    assert len(fitted.names) == 5
    assert fitted.y_train.isin([0, 1]).all()
    assert (encode_target(dataset, "Memiliki_Pinjaman") == (dataset["Memiliki_Pinjaman"] == "Ya")).all()
    evaluated = evaluate_model(fitted)
    assert evaluated["confusion_matrix"].sum() == 250


def test_evaluation_metrics_and_predictions(dataset):
    train, test = split_dataset(dataset, TARGET, 80, 42, True)
    fitted = fit_model(dataset, FEATURES, TARGET, train, test)
    evaluated = evaluate_model(fitted)
    assert evaluated["confusion_matrix"].sum() == 200
    assert len(evaluated["probabilities"]) == 200
    assert len(evaluated["predictions"]) == 200
    assert evaluated["accuracy"] == pytest.approx(0.75)
    assert 0 <= evaluated["roc_auc"] <= 1


def test_reject_invalid_data(dataset):
    bad = dataset.copy()
    bad.loc[0, "Pekerjaan"] = "Tidak dikenal"
    with pytest.raises(ValueError, match="Pekerjaan"):
        validate_dataset(bad)
    bad2 = dataset.copy()
    bad2.loc[1, TARGET] = 2
    with pytest.raises(ValueError, match="0 dan 1"):
        validate_dataset(bad2)


def test_excel_dataset_matches_csv(dataset):
    workbook = DATA.with_suffix(".xlsx")
    excel = validate_dataset(pd.read_excel(workbook, sheet_name="Data_Simulasi"))
    pd.testing.assert_frame_equal(dataset, excel)
