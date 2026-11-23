"""Validación gráfica reproducible: genera las figuras que se incrustan en los informes.

No son capturas pegadas a mano: cada imagen sale de los artefactos del proyecto
(Silver, Gold y el pipeline congelado) y se regenera con `make figuras`.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin pantalla (CI, servidor)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from flows.common.config import settings
from flows.common.logging import get_logger
from flows.common.paths import REPORTS, SILVER, SILVER_NAME

ml = importlib.import_module("flows.05_ml_models")

log = get_logger(__name__)

FIG_DIR_NAME = "figuras"
REPORT_NAME = "validacion_grafica.md"


def frozen_pipeline():
    """Pipeline con el estimador y los hiperparámetros congelados en la ficha."""
    estimator = ml.get_candidates(settings.random_state)[ml.FINAL_ESTIMATOR]
    return ml.build_pipeline(estimator).set_params(**ml.FINAL_PARAMS)


def oof_probabilities(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Probabilidades fuera de pliegue: ninguna fila entrena a su propio puntaje."""
    cv = StratifiedKFold(settings.cv_folds, shuffle=True, random_state=settings.random_state)
    return cross_val_predict(frozen_pipeline(), X, y, cv=cv, method="predict_proba")[:, 1]


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_eda_evidence(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Dos figuras que contrastan los hallazgos del EDA con los datos de Silver."""
    out_dir = Path(out_dir)
    paths = []

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ["Sex", "Pclass", "Alone"]):
        sns.pointplot(data=df, x=col, y="Survived", errorbar=("ci", 95), ax=ax, color="steelblue")
        ax.set_title(f"Supervivencia por {col}")
        ax.set_ylabel("Tasa de supervivencia")
        ax.set_xlabel("")
    fig.suptitle("Tasas con intervalo de confianza al 95 % (Silver, n = 891)")
    fig.tight_layout()
    path = out_dir / "eda_supervivencia.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df["AgeOriginal"].dropna(), bins=30, density=True, alpha=0.6,
            label="Edad original (con NaN)", color="steelblue")
    ax.hist(df["Age"].dropna(), bins=30, density=True, alpha=0.6,
            label="Edad imputada por género", color="darkorange")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Densidad")
    ax.set_title("Efecto visible de la imputación: pico central artificial")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "eda_imputacion_edad.png"
    _save(fig, path)
    paths.append(path)
    return paths


def plot_model_diagnostics(y_true: pd.Series, proba: np.ndarray, threshold: float,
                           out_dir: Path) -> tuple[list[Path], dict]:
    """ROC/PR, calibración y matriz de confusión sobre predicciones OOF."""
    out_dir = Path(out_dir)
    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)
    brier = brier_score_loss(y_true, proba)
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    RocCurveDisplay.from_predictions(y_true, proba, name="logreg (OOF)", ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="azar")
    axes[0].set_title(f"ROC — AUC = {auc:.3f}")
    axes[0].legend()
    PrecisionRecallDisplay.from_predictions(y_true, proba, name="logreg (OOF)", ax=axes[1])
    axes[1].set_title(f"Precisión–recall — AP = {ap:.3f}")
    fig.tight_layout()
    path = out_dir / "modelo_roc_pr.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 6))
    frac_true, frac_pred = calibration_curve(y_true, proba, n_bins=8)
    ax.plot([0, 1], [0, 1], "k--", label="perfectamente calibrado")
    ax.plot(frac_pred, frac_true, "o-", label=f"logreg (OOF), Brier = {brier:.3f}")
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    ax.set_title("Calibración (8 intervalos, validación cruzada)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "modelo_calibracion.png"
    _save(fig, path)
    paths.append(path)

    pred = (proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, display_labels=["No sobrevivió", "Sobrevivió"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(f"Matriz de confusión OOF (umbral {threshold:.1f})")
    path = out_dir / "modelo_confusion.png"
    _save(fig, path)
    paths.append(path)

    return paths, {"auc": float(auc), "ap": float(ap), "brier": float(brier),
                   "n_oof": len(y_true), "threshold": float(threshold)}


def write_report(stats: dict, out_dir: Path) -> Path:
    """Informe puente entre las cifras de los informes y las imágenes generadas."""
    lines = [
        "# Validación gráfica de EDA y modelo",
        "",
        ("**Generado:** `make figuras` · Figuras en `figuras/`. Las imágenes no son capturas "
        "manuales: se regeneran desde Silver y desde el pipeline congelado."),
        "",
        "## 1. Los hallazgos del EDA, vistos con incertidumbre",
        "",
        "![Supervivencia por grupo](figuras/eda_supervivencia.png)",
        "",
        ("*Figura 1.* Puntos = tasa de supervivencia; barras = intervalo de confianza al 95 %. "
        "Confirma visualmente la gradiente por clase, la ventaja femenina y que viajar acompañado "
        "se asocia con más supervivencia. Las barras más anchas (pocos casos) son la advertencia: "
        "no hay que leer diferencias pequeñas como reglas."),
        "",
        "![Efecto de la imputación de edad](figuras/eda_imputacion_edad.png)",
        "",
        ("*Figura 2.* Comparar la edad original con la imputada revela un pico central que no estaba "
        "en los datos: 177 edades se rellenaron con promedios por género. Es el resultado esperado "
        "de la decisión de limpieza, y por eso el modelo usa `AgeOriginal` y vuelve a imputar "
        "dentro del `Pipeline`."),
        "",
        "## 2. Validación del modelo sin reabrir la prueba",
        "",
        (f"Las tres figuras siguientes usan **{stats['n_oof']} predicciones fuera de pliegue** "
        "(5 pliegues sobre los 712 de entrenamiento). El conjunto de prueba de 179 sigue "
        "intocable: sus cifras están en `model_card.md`."),
        "",
        "![ROC y precisión-recall](figuras/modelo_roc_pr.png)",
        "",
        (f"*Figura 3.* ROC (AUC OOF = **{stats['auc']:.3f}**) y precisión–recall "
        f"(AP = **{stats['ap']:.3f}**). La curva PR es la más informativa con un 38 % de "
        "positivos: muestra el compromiso entre encontrar sobrevivientes y no desperdiciar "
        "predicciones positivas."),
        "",
        "![Calibración](figuras/modelo_calibracion.png)",
        "",
        (f"*Figura 4.* Calibración por intervalos (Brier = **{stats['brier']:.3f}**). "
        "Si los puntos se apartan de la diagonal, la probabilidad no debe leerse como una "
        "frecuencia empírica. Es una comprobación gráfica del Brier reportado en la ficha."),
        "",
        "![Matriz de confusión](figuras/modelo_confusion.png)",
        "",
        (f"*Figura 5.* Errores a umbral {stats['threshold']:.1f}. Permite ver los falsos "
        "negativos y positivos que la exactitud global esconde, y coincide con la lectura por "
        "subgrupos de `problema_ml.md`: el modelo reproduce la desigualdad histórica."),
        "",
        "## 3. Deriva",
        "",
        ("La validación sobre los 418 pasajeros sin etiqueta y sus figuras están en "
        "`validacion_externa.md`. Recuérdese que allí **no hay métrica de desempeño**: solo "
        "cuando hay etiquetas se puede medir aciertos; esa cohorte solo permite vigilar deriva."),
        "",
    ]
    path = Path(out_dir) / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(out_dir: Path = REPORTS) -> Path:
    """Genera todas las figuras y el informe; devuelve la ruta del Markdown."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / FIG_DIR_NAME
    X, y = ml.load_dataset()
    X_tr, _, y_tr, _ = ml.split(X, y)
    proba = oof_probabilities(X_tr, y_tr)

    eda_paths = plot_eda_evidence(pd.read_parquet(SILVER / SILVER_NAME), fig_dir)
    model_paths, stats = plot_model_diagnostics(y_tr, proba, ml.THRESHOLD, fig_dir)
    report = write_report(stats, out_dir)

    log.info("Validación gráfica: %d figuras, AUC OOF %.3f, Brier %.3f",
             len(eda_paths) + len(model_paths), stats["auc"], stats["brier"])
    log.info("Informe: %s", report)
    return report


if __name__ == "__main__":
    run()
