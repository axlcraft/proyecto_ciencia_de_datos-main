import importlib

import numpy as np
import pandas as pd

rf = importlib.import_module("flows.07_report_figures")


def test_plot_eda_evidence_writes_pngs(tmp_path):
    silver = pd.DataFrame({
        "Survived": [0, 1, 1, 0, 1, 0] * 10,
        "Sex": ["male", "female"] * 30,
        "Pclass": [1, 2, 3] * 20,
        "Alone": ["Alone", "with Family"] * 30,
        "AgeOriginal": [22.0, 38.0, np.nan, 2.0, 30.0, 28.0] * 10,
        "Age": [22.0, 38.0, 30.0, 2.0, 30.0, 28.0] * 10,
    })
    paths = rf.plot_eda_evidence(silver, tmp_path / "figuras")
    assert [p.name for p in paths] == ["eda_supervivencia.png", "eda_imputacion_edad.png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_plot_model_diagnostics_writes_pngs_and_stats(tmp_path):
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 80))
    proba = np.clip(rng.uniform(0.1, 0.9, 80), 0, 1)
    paths, stats = rf.plot_model_diagnostics(y, proba, 0.5, tmp_path / "figuras")
    assert [p.name for p in paths] == [
        "modelo_roc_pr.png", "modelo_calibracion.png", "modelo_confusion.png",
    ]
    assert all(p.exists() for p in paths)
    assert set(stats) == {"auc", "ap", "brier", "n_oof", "threshold"}
    assert stats["n_oof"] == 80 and 0 <= stats["auc"] <= 1
