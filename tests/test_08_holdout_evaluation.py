import importlib
import json

import numpy as np
import pandas as pd
import pytest

from flows.common.hashing import sha256_of
from flows.common.paths import SILVER_NAME

ho = importlib.import_module("flows.08_holdout_evaluation")
ml = importlib.import_module("flows.05_ml_models")
ev = importlib.import_module("flows.06_external_validation")


@pytest.fixture
def silver_like(tmp_path):
    """Silver sintético con las columnas que consume el modelo."""
    rng = np.random.default_rng(7)
    n = 160
    frame = {
        "Age": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(1, 75, n)),
        "SibSp": rng.integers(0, 4, n),
        "Parch": rng.integers(0, 3, n),
        "Fare": rng.uniform(5, 100, n),
        "Familia": rng.integers(0, 6, n),
        "Pclass": rng.choice([1, 2, 3], n),
        "Sex": rng.choice(["male", "female"], n),
        "Embarked": rng.choice(["S", "C", "Q"], n),
        "TTitulo": rng.choice(["Mr", "Mrs", "Miss", "Otro"], n),
        "Alone": rng.choice(["Alone", "with Family"], n),
        "HasCabin": rng.integers(0, 2, n),
        "TCabin": rng.choice(list("XXXXCB"), n),
    }
    df = pd.DataFrame(frame)
    df["AgeOriginal"] = df["Age"]
    df["Survived"] = ((df["Sex"] == "female") & (rng.random(n) < 0.9)).astype(int)
    path = tmp_path / "silver" / SILVER_NAME
    path.parent.mkdir(parents=True)
    df.to_parquet(path, index=False)
    return path, df


def test_run_freezes_once_then_only_verifies(silver_like, tmp_path):
    data, _ = silver_like
    out = tmp_path / "evaluacion_prueba.json"

    ho.run(data_path=data, out_path=out)
    first = out.read_bytes()
    payload = json.loads(first)
    assert payload["metricas"]["n_test"] == len(payload["probas_prueba"])
    assert {"roc_auc", "accuracy", "roc_auc_ci95", "accuracy_ci95"} <= set(payload["metricas"])

    ho.run(data_path=data, out_path=out)  # segunda corrida: no evalúa de nuevo
    assert out.read_bytes() == first
    loaded = ho.load_holdout(path=out, data_sha256=sha256_of(data))
    assert loaded["metricas"] == payload["metricas"]


def test_load_rejects_missing_tampered_and_mismatched_data(silver_like, tmp_path):
    data, df = silver_like
    out = tmp_path / "evaluacion_prueba.json"
    with pytest.raises(FileNotFoundError, match="make holdout"):
        ho.load_holdout(path=out)

    ho.run(data_path=data, out_path=out)
    tampered = json.loads(out.read_text(encoding="utf-8"))
    tampered["metricas"]["roc_auc"] = 0.01
    out.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="integridad"):
        ho.load_holdout(path=out)

    ho.run(data_path=data, out_path=out, force=True)
    changed = df.copy()
    changed.loc[len(changed)] = changed.iloc[0]
    changed.to_parquet(data, index=False)
    with pytest.raises(RuntimeError, match="datos cambiaron"):
        ho.run(data_path=data, out_path=out)
    with pytest.raises(RuntimeError, match="datos cambiaron"):
        ho.load_holdout(path=out, data_sha256=sha256_of(data))


def test_load_rejects_changed_decisions(silver_like, tmp_path, monkeypatch):
    data, _ = silver_like
    out = tmp_path / "evaluacion_prueba.json"
    ho.run(data_path=data, out_path=out)

    original = ho.decisions
    monkeypatch.setattr(ho, "decisions", lambda: {**original(), "threshold": 0.4})
    with pytest.raises(RuntimeError, match="decisiones del modelo cambiaron"):
        ho.load_holdout(path=out)


def test_train_uses_frozen_metrics_without_reopening_test(silver_like, tmp_path, monkeypatch):
    data, _ = silver_like
    out = tmp_path / "evaluacion_prueba.json"
    ho.run(data_path=data, out_path=out)
    esperado = json.loads(out.read_text(encoding="utf-8"))

    def prohibido(*args, **kwargs):
        raise AssertionError("El entrenamiento reabrió el conjunto de prueba")

    monkeypatch.setattr(ml, "evaluate_test_with_ci", prohibido)
    monkeypatch.setattr(ml, "evaluate_on_test", prohibido)
    model = ml.run(
        in_dir=data.parent, models_dir=tmp_path / "models", holdout_path=out
    )
    meta = json.loads(model.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    assert meta["metrics"] == esperado["metricas"]
    assert meta["holdout_sha256"] == esperado["integridad_sha256"]


def test_prediction_drift_uses_frozen_probabilities(silver_like, monkeypatch):
    data, _ = silver_like
    X, y = ml.load_dataset(data)
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X.iloc[:40], y.iloc[:40])

    def prohibido(*args, **kwargs):
        raise AssertionError("El informe reentrenó para reabrir la prueba")

    monkeypatch.setattr(ml, "train_final", prohibido)
    preds = ev.prediction_drift(
        pipe, X.iloc[:3], X.iloc[:40], y.iloc[:40], 0.5,
        probas_prueba=[0.1, 0.8, 0.6],
    )
    assert preds["n_prueba"] == 3
    assert preds["probabilidad_media_prueba"] == pytest.approx(0.5)
    assert preds["tasa_positiva_prueba"] == pytest.approx(2 / 3)
