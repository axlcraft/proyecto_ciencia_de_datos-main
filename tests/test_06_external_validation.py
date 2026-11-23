import importlib

import numpy as np
import pandas as pd
import pytest

ev = importlib.import_module("flows.06_external_validation")


@pytest.fixture
def bronze_like():
    """Cinco filas con el contrato de Bronze, una sin edad y otra sin etiqueta."""
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [np.nan] * 5,
        "Pclass": [1, 2, 3, 3, 1],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Palsson, Master. Gosta Leonard",
            "Moran, Dr. James",
        ],
        "Sex": ["male", "female", "female", "male", "male"],
        "Age": [22.0, 38.0, np.nan, 2.0, np.nan],
        "SibSp": [1, 1, 0, 3, 0],
        "Parch": [0, 0, 0, 1, 0],
        "Ticket": list("ABCDE"),
        "Fare": [7.25, 71.28, 7.92, 21.07, 8.46],
        "Cabin": [None, "C85", None, None, "B77"],
        "Embarked": ["S", "C", None, "S", "Q"],
    })


def test_prepare_features_reuses_silver_rules_and_keeps_nan_age(bronze_like):
    X = ev.prepare_features(bronze_like)
    assert list(X.columns) == list(ev.ml.CANDIDATE_COLUMNS)
    assert X["Familia"].tolist() == [1, 1, 0, 4, 0]
    assert X["Alone"].tolist() == ["with Family", "with Family", "Alone",
                                   "with Family", "Alone"]
    assert X["TTitulo"].tolist() == ["Mr", "Mrs", "Miss", "Mr", "Otro"]
    assert X["Age"].isna().sum() == 2                       # el pipeline imputará
    assert X["Embarked"].isna().sum() == 1                   # no se imputa fuera del modelo


def test_psi_is_zero_for_identical_and_high_for_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(30, 10, 1000)
    assert ev.psi(ref, ref) == pytest.approx(0, abs=0.01)
    assert ev.psi(ref, rng.normal(60, 10, 1000)) > 0.2
    assert ev.classify(0.05) == "estable"
    assert ev.classify(0.15) == "moderada"
    assert ev.classify(0.35) == "alta"


def test_categorical_drift_reports_new_categories():
    ref = pd.DataFrame({"Sex": ["male"] * 80 + ["female"] * 20})
    ext = pd.DataFrame({"Sex": ["male"] * 40 + ["female"] * 40 + ["other"] * 20})
    # Se usa una sola columna válida del conjunto del modelo.
    original = ev.CATEGORICAL_FEATURES
    ev.CATEGORICAL_FEATURES = ["Sex"]
    try:
        result = ev.categorical_drift(ref, ext)
        assert result["Sex"]["categorias_nuevas"] == ["other"]
        assert result["Sex"]["tvd"] > 0.2
        assert result["Sex"]["senal"] == "alta"
    finally:
        ev.CATEGORICAL_FEATURES = original
