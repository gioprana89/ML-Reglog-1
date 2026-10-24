"""Uji integrasi tabel prediksi dan evaluasi semua bab (5--12)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_core import (
    FEATURES, TARGET, encode_target, evaluate_model, fit_model,
    prediction_table, split_dataset, validate_dataset,
)

DATA = Path(__file__).resolve().parents[1] / "Simulasi_1000_Nasabah_Regresi_Logistik_Biner.csv"


@pytest.fixture(scope="module")
def setup_model():
    df = validate_dataset(pd.read_csv(DATA, encoding="utf-8-sig"))
    idx_train, idx_test = split_dataset(df, TARGET, 80, 42, True)
    model = fit_model(df, FEATURES, TARGET, idx_train, idx_test)
    return df, idx_train, idx_test, model


def test_all_predictions_have_same_order_and_unique_ids(setup_model):
    df, itrain, itest, trained = setup_model
    train = prediction_table(trained, df, itrain, "train")
    test = prediction_table(trained, df, itest, "test")
    assert train.shape[0] == 800 and test.shape[0] == 200
    assert list(train["ID_Nasabah"]) == df.iloc[itrain]["ID_Nasabah"].tolist()
    assert list(test["ID_Nasabah"]) == df.iloc[itest]["ID_Nasabah"].tolist()
    assert set(train.ID_Nasabah).isdisjoint(set(test.ID_Nasabah))
    for subset in (train, test):
        assert subset["Status_Prediksi"].isin(["Benar", "Salah"]).all()
        assert ((subset["Y_Aktual"] == subset["Y_Prediksi"]) ==
                (subset["Status_Prediksi"] == "Benar")).all()
        np.testing.assert_allclose(subset["Probabilitas_Y0"] + subset["Probabilitas_Y1"], 1)
        assert (subset["Probabilitas_Y1"] >= 0).all()
        assert (subset["Probabilitas_Y1"] <= 1).all()
        np.testing.assert_array_equal(subset["Y_Prediksi"], (subset["Probabilitas_Y1"] >= .5).astype(int))


def test_testing_predictions_match_metrics_and_confusion(setup_model):
    df, itrain, itest, trained = setup_model
    predictions = prediction_table(trained, df, itest, "test")
    result = evaluate_model(trained, .5)
    assert result["confusion_matrix"].tolist() == [[125, 12], [38, 25]]
    np.testing.assert_allclose(predictions["Probabilitas_Y1"], result["probabilities"])
    np.testing.assert_array_equal(predictions["Y_Prediksi"], result["predictions"])
    assert (predictions["Status_Prediksi"] == "Benar").sum() == 150
    assert (predictions["Status_Prediksi"] == "Salah").sum() == 50
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["recall"] == pytest.approx(25 / 63)
    assert result["specificity"] == pytest.approx(125 / 137)
    assert result["precision"] == pytest.approx(25 / 37)
    assert result["f1"] == pytest.approx(0.5)


def test_changing_seed_or_split_does_not_mingle_indices(setup_model):
    df, _, _, _ = setup_model
    for split, seed in [(70, 42), (80, 100), (90, 2026)]:
        tr, te = split_dataset(df, TARGET, split, seed, True)
        fitted = fit_model(df, FEATURES, TARGET, tr, te)
        a = prediction_table(fitted, df, tr, "train")
        b = prediction_table(fitted, df, te, "test")
        assert len(a) + len(b) == len(df)
        assert set(a["ID_Nasabah"]).isdisjoint(set(b["ID_Nasabah"]))
        assert a["Y_Aktual"].eq(encode_target(df, TARGET).iloc[tr].to_numpy()).all()
        assert b["Y_Aktual"].eq(encode_target(df, TARGET).iloc[te].to_numpy()).all()
        assert len(evaluate_model(fitted)["probabilities"]) == len(b)


def test_invalid_subset_is_rejected(setup_model):
    df, train, _, fitted = setup_model
    with pytest.raises(ValueError, match="subset"):
        prediction_table(fitted, df, train, "testing")
    with pytest.raises(ValueError, match="Threshold"):
        prediction_table(fitted, df, train, "train", threshold=1.0)
