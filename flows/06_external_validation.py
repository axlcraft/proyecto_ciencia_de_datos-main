"""Validación externa no supervisada: cohorte sin etiqueta -> informe de deriva.

Los 418 pasajeros de Bronze nunca se usaron para entrenar, pero **no tienen
etiqueta**: no se puede calcular accuracy ni ROC-AUC. Este flujo compara sus
covariables y las predicciones del modelo guardado contra la cohorte etiquetada
de Silver. Es monitorización de deriva, no una medición de desempeño.
"""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # seaborn/matplotlib sin pantalla (servidor, CI)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE,
    BRONZE_UNLABELED_NAME,
    MODELS,
    REPORTS,
    ROOT,
    SILVER,
    SILVER_NAME,
)

ml = importlib.import_module("flows.05_ml_models")
de = importlib.import_module("flows.01_data_engineering")
ho = importlib.import_module("flows.08_holdout_evaluation")

log = get_logger(__name__)

FIG_DIR_NAME = "figuras"
REPORT_NAME = "validacion_externa.md"
JSON_NAME = "validacion_externa.json"
NUMERIC_FEATURES = list(ml.NUM_FEATURES)
CATEGORICAL_FEATURES = list(ml.CAT_FEATURES)

# Heurística habitual para PSI / distancia de variación total. Es una regla de
# cribado, no una prueba estadística: depende del contexto y del n.
STABLE_MAX = 0.10
MODERATE_MAX = 0.20


def _f(value) -> float:
    """Convierte a float nativo (JSON no serializa numpy.float64)."""
    return float(value)


def _display_path(path: Path) -> str:
    """Ruta relativa al repositorio; evita rutas absolutas en los informes."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def classify(value: float) -> str:
    """Clasifica PSI/TVD en estable, moderada o alta (umbrales documentados)."""
    if not np.isfinite(value):
        return "indeterminada"
    if abs(value) < STABLE_MAX:
        return "estable"
    if abs(value) < MODERATE_MAX:
        return "moderada"
    return "alta"


def load_external(path: Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Lee los pasajeros sin etiqueta y valida el contrato de Bronze."""
    source = Path(path) if path else BRONZE / BRONZE_UNLABELED_NAME
    raw = pd.read_csv(source)
    de.validate_bronze(raw)
    return raw, source


def prepare_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Aplica a Bronze las mismas transformaciones deterministas que en el cuaderno 04.

    No se imputa `Embarked` ni `Age` aquí: el pipeline guardado hace la imputación
    con estadísticas aprendidas en entrenamiento. No se usa la etiqueta (no existe).
    """
    out = (
        raw.pipe(de.cast_types)
        .pipe(de.add_cabin_features)
        .pipe(de.add_family_features)
        .pipe(de.add_title)
        .pipe(de.consolidate_titles)
    )
    X = out[list(ml.CANDIDATE_COLUMNS)].copy()
    X["Age"] = out["Age"].astype(float)  # edad ORIGINAL, con NaN
    return X


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index con cortes cuantílicos de la referencia.

    Valores orientativos: <0,10 estable; 0,10–0,20 moderado; >0,20 alto.
    Los NaN se excluyen por variable y se reportan aparte.
    """
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(pd.Series(current), errors="coerce").dropna().to_numpy(dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # referencia casi constante: el PSI no es informativo
        return 0.0 if np.isclose(ref.mean(), cur.mean()) else 1.0
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_p = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_p = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)
    ref_p /= ref_p.sum()
    cur_p /= cur_p.sum()
    return _f(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def numeric_drift(X_ref: pd.DataFrame, X_ext: pd.DataFrame) -> dict[str, dict]:
    """PSI, test de Kolmogorov–Smirnov y resumen numérico por variable."""
    result = {}
    for col in NUMERIC_FEATURES:
        ref = pd.to_numeric(X_ref[col], errors="coerce").dropna()
        ext = pd.to_numeric(X_ext[col], errors="coerce").dropna()
        value = psi(ref, ext)
        ks = stats.ks_2samp(ref, ext) if len(ref) and len(ext) else None
        result[col] = {
            "n_ref": len(ref),
            "n_externo": len(ext),
            "media_ref": _f(ref.mean()),
            "media_externo": _f(ext.mean()),
            "mediana_ref": _f(ref.median()),
            "mediana_externo": _f(ext.median()),
            "missing_pct_ref": _f(100 * X_ref[col].isna().mean()),
            "missing_pct_externo": _f(100 * X_ext[col].isna().mean()),
            "psi": _f(value),
            "ks_stat": _f(ks.statistic) if ks else float("nan"),
            "ks_p": _f(ks.pvalue) if ks else float("nan"),
            "senal": classify(value),
        }
    return result


def categorical_drift(X_ref: pd.DataFrame, X_ext: pd.DataFrame) -> dict[str, dict]:
    """Distancia de variación total y categorías nuevas por variable."""
    result = {}
    for col in CATEGORICAL_FEATURES:
        p_ref = X_ref[col].value_counts(normalize=True, dropna=True)
        p_ext = X_ext[col].value_counts(normalize=True, dropna=True)
        categories = sorted(set(p_ref.index) | set(p_ext.index), key=str)
        tvd = 0.5 * sum(abs(float(p_ref.get(c, 0)) - float(p_ext.get(c, 0))) for c in categories)
        new_categories = sorted(set(p_ext.index) - set(p_ref.index), key=str)
        result[col] = {
            "tvd": _f(tvd),
            "n_categorias_ref": len(p_ref),
            "n_categorias_externo": len(p_ext),
            "categorias_nuevas": [str(c) for c in new_categories],
            "missing_pct_ref": _f(100 * X_ref[col].isna().mean()),
            "missing_pct_externo": _f(100 * X_ext[col].isna().mean()),
            "distribuciones": {
                "ref": {str(k): _f(v) for k, v in p_ref.items()},
                "externo": {str(k): _f(v) for k, v in p_ext.items()},
            },
            "senal": classify(tvd),
        }
    return result


def prediction_drift(
    pipeline, X_ext: pd.DataFrame, X_ref: pd.DataFrame, y_ref: pd.Series,
    threshold: float, probas_prueba: list[float] | None = None,
) -> dict:
    """Distribución de predicciones: cohorte externa vs. prueba honesta (179).

    La referencia son las probabilidades **congeladas** por `make holdout`
    (modelo entrenado solo con los 712 y evaluado una vez en los 179). Este
    informe no reentrena ni reabre la prueba: si no se pasan las probabilidades,
    se leen del artefacto versionado. Las probabilidades del modelo definitivo
    sobre los 891 se incluyen solo como contexto in-sample (optimista).
    """
    if probas_prueba is None:
        probas_prueba = ho.load_holdout(
            data_sha256=sha256_of(SILVER / SILVER_NAME)
        )["probas_prueba"]
    scores = np.asarray(probas_prueba, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("probas_prueba debe ser una secuencia no vacía de probabilidades")
    honest = pd.DataFrame({
        "proba_survive": scores,
        "pred": (scores >= threshold).astype(int),
    })
    external = ml.predict(pipeline, X_ext, threshold)
    in_sample = ml.predict(pipeline, X_ref, threshold)
    return {
        "n_externo": len(external),
        "n_prueba": len(honest),
        "threshold": _f(threshold),
        "tasa_real_entrenamiento": _f(y_ref.mean()),
        "probabilidad_media_externo": _f(external["proba_survive"].mean()),
        "probabilidad_media_prueba": _f(honest["proba_survive"].mean()),
        "probabilidad_media_insample": _f(in_sample["proba_survive"].mean()),
        "tasa_positiva_externo": _f(external["pred"].mean()),
        "tasa_positiva_prueba": _f(honest["pred"].mean()),
        "tasa_positiva_insample": _f(in_sample["pred"].mean()),
        "n_nan": int(external.isna().sum().sum()),
        "probas": external["proba_survive"].tolist(),
        "probas_prueba": honest["proba_survive"].tolist(),
    }


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def make_figures(numeric: dict, categorical: dict, preds: dict, out_dir: Path) -> list[Path]:
    """Guarda las tres figuras de deriva que se incrustan en el informe."""
    out_dir = Path(out_dir)
    paths = []

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, NUMERIC_FEATURES):
        ax.hist(numeric[col]["valores_ref"], bins=30, density=True, alpha=0.6,
                label="Silver (891)", color="steelblue")
        ax.hist(numeric[col]["valores_ext"], bins=30, density=True, alpha=0.6,
                label="Sin etiqueta (418)", color="darkorange")
        ax.set_title(f"{col}\nPSI={numeric[col]['psi']:.3f} ({numeric[col]['senal']})")
        ax.legend(fontsize=7)
    fig.suptitle("Deriva de variables numéricas (distribuciones relativas)")
    fig.tight_layout()
    path = out_dir / "deriva_numerica.png"
    _save(fig, path)
    paths.append(path)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flat, CATEGORICAL_FEATURES):
        dist = categorical[col]["distribuciones"]
        keys = sorted(set(dist["ref"]) | set(dist["externo"]), key=str)
        x = np.arange(len(keys))
        ax.bar(x - 0.2, [dist["ref"].get(k, 0) for k in keys], 0.4,
               label="Silver (891)", color="steelblue")
        ax.bar(x + 0.2, [dist["externo"].get(k, 0) for k in keys], 0.4,
               label="Sin etiqueta (418)", color="darkorange")
        ax.set_xticks(x, keys, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{col}\nTVD={categorical[col]['tvd']:.3f} ({categorical[col]['senal']})")
        ax.legend(fontsize=7)
    fig.suptitle("Deriva de variables categóricas (proporciones)")
    fig.tight_layout()
    path = out_dir / "deriva_categorica.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(preds["probas_prueba"], bins=25, density=True, alpha=0.6,
            label=f"Prueba honesta (n={preds['n_prueba']})", color="steelblue")
    ax.hist(preds["probas"], bins=25, density=True, alpha=0.6,
            label=f"Sin etiqueta (n={preds['n_externo']})", color="darkorange")
    ax.axvline(preds["threshold"], color="black", linestyle="--", label="Umbral 0,5")
    ax.set_xlabel("Probabilidad predicha de sobrevivir")
    ax.set_ylabel("Densidad")
    ax.set_title("Deriva de predicciones (sin etiquetas = sin métrica de desempeño)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "deriva_predicciones.png"
    _save(fig, path)
    paths.append(path)
    return paths


def write_report(
    numeric: dict, categorical: dict, preds: dict, meta: dict, out_dir: Path,
) -> Path:
    """Escribe el informe Markdown con tablas, figuras y límites explícitos."""
    flagged = [f"`{c}` ({d['senal']}, PSI {d['psi']:.3f})"
               for c, d in numeric.items() if d["senal"] != "estable"]
    flagged += [f"`{c}` ({d['senal']}, TVD {d['tvd']:.3f})"
                for c, d in categorical.items() if d["senal"] != "estable"]
    resumen = ", ".join(flagged) if flagged else "ninguna supera el umbral de 0,10"

    lines = [
        "# Validación externa y deriva — cohorte sin etiqueta",
        "",
        "**Generado:** " + meta["generado_at"] + " · **Comando:** `make validar-externa`",
        "",
        "## Alcance (leer antes que las cifras)",
        "",
        (f"Se cargan los **{preds['n_externo']} pasajeros sin etiqueta** de "
        f"`{meta['fuente']}` (SHA-256 `{meta['sha256_fuente'][:12]}…`), que nunca "
        "entraron en el entrenamiento, y se puntúan con el modelo guardado "
        f"`{meta['modelo']}` (SHA-256 `{meta['sha256_modelo'][:12]}…`)."),
        "",
        ("> **No hay validación de desempeño.** Al no existir `Survived` en esta cohorte "
        "no se puede calcular exactitud, recall ni ROC-AUC, y **no** se reabre la "
        "métrica de prueba. Lo que sigue es *monitorización de deriva*: ¿se parece la "
        "cohorte nueva a la de entrenamiento en sus entradas y en las predicciones?"),
        "",
        ("Referencia: 891 pasajeros etiquetados de Silver. Las probabilidades de "
        "«prueba honesta» están **congeladas** en `evaluacion_prueba.json` (evaluación "
        f"única sobre 712 entrenamiento y {preds['n_prueba']} prueba, hecha por `make "
        "holdout`); este informe no las recalcula ni reabre la métrica final. La columna "
        "«in-sample» usa el modelo definitivo sobre los 891 y es optimista: solo contexto."),
        "",
        "## Variables numéricas",
        "",
        "| Variable | Media ref. | Media ext. | PSI | KS p-valor | Faltantes ext. | Señal |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for col, d in numeric.items():
        lines.append(
            f"| `{col}` | {d['media_ref']:.2f} | {d['media_externo']:.2f} | "
            f"{d['psi']:.3f} | {d['ks_p']:.3g} | {d['missing_pct_externo']:.1f} % | "
            f"{d['senal']} |"
        )
    lines += [
        "",
        "![Distribuciones numéricas](figuras/deriva_numerica.png)",
        "",
        ("*Figura 1.* Las barras azules son la cohorte etiquetada; las naranjas, la cohorte sin "
        "etiqueta. Un desplazamiento visible con PSI alto indica que el modelo podría estar "
        "puntuando un perfil poco frecuente en entrenamiento. El p-valor de KS es sensible al "
        "tamaño muestral: se lee junto al PSI y a la figura, no aislado."),
        "",
        "## Variables categóricas",
        "",
        "| Variable | TVD | Categorías nuevas | Faltantes ext. | Señal |",
        "|---|---:|---|---:|---|",
    ]
    for col, d in categorical.items():
        new = ", ".join(d["categorias_nuevas"]) or "—"
        lines.append(
            f"| `{col}` | {d['tvd']:.3f} | {new} | {d['missing_pct_externo']:.1f} % | "
            f"{d['senal']} |"
        )
    lines += [
        "",
        "![Distribuciones categóricas](figuras/deriva_categorica.png)",
        "",
        ("*Figura 2.* Proporciones por categoría. La distancia de variación total (TVD) es la "
        "mitad de la suma de diferencias absolutas: 0 sería idéntica, 1 no solapada. Las "
        "categorías nuevas se absorben con `handle_unknown=\"infrequent_if_exist\"`, pero conviene "
        "vigilarlas: el modelo no aprendió su efecto."),
        "",
        "## Predicciones",
        "",
        "| Métrica | Prueba honesta (179) | Sin etiqueta (418) | In-sample (891) |",
        "|---|---:|---:|---:|",
        (f"| Probabilidad media | {preds['probabilidad_media_prueba']:.3f} | "
        f"{preds['probabilidad_media_externo']:.3f} | "
        f"{preds['probabilidad_media_insample']:.3f} |"),
        (f"| Tasa predicha positiva (umbral {preds['threshold']:.1f}) | "
        f"{preds['tasa_positiva_prueba']:.3f} | {preds['tasa_positiva_externo']:.3f} | "
        f"{preds['tasa_positiva_insample']:.3f} |"),
        "",
        (f"Tasa real de supervivencia en entrenamiento: {preds['tasa_real_entrenamiento']:.3f}. "
        f"Predicciones con NaN: **{preds['n_nan']}**. La cohorte sin etiqueta no permite saber "
        "si esas predicciones son correctas: solo permite comparar la distribución con la de "
        "un conjunto donde sí conocemos la respuesta."),
        "",
        "![Distribución de predicciones](figuras/deriva_predicciones.png)",
        "",
        ("*Figura 3.* Si la curva naranja se desplaza de la azul, cambió la mezcla de perfiles "
        "(deriva de covariables) o el modelo está extrapolando. No dice nada sobre aciertos."),
        "",
        "## Conclusión y límites",
        "",
        f"Señales fuera del rango «estable» (PSI/TVD ≥ {STABLE_MAX:.2f}): **{resumen}**.",
        "",
        ("- Los umbrales 0,10 / 0,20 son una **heurística de cribado**, no un contrato: con "
        "418 filas, variaciones pequeñas pueden ser ruido y variaciones grandes pueden ser "
        "cambios legítimos de composición (por ejemplo, más pasajeros de 3.ª clase)."),
        ("- Bronze sin etiqueta pertenece al **mismo viaje** y a la misma fuente que el "
        "entrenamiento: es una cohorte de inferencia, no una validación temporal ni un "
        "dataset de otro naufragio. No demuestra transferibilidad."),
        ("- Las transformaciones se reutilizan de `flows/01_data_engineering.py` y la "
        "imputación ocurre dentro del pipeline guardado: no se filtraron estadísticas de la "
        "cohorte nueva hacia el entrenamiento."),
        ("- Si la deriva es «alta», el siguiente paso es investigar la fuente del cambio y "
        "reentrenar **con etiquetas nuevas**; ajustar el modelo contra esta cohorte sin "
        "etiquetas no es posible y sería metodológicamente incorrecto."),
        "",
        ("Detalles serializables (todas las cifras, hashes y umbrales): "
        f"`{JSON_NAME}`. Figuras: `figuras/`."),
        "",
    ]
    path = Path(out_dir) / REPORT_NAME
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run(
    model_path: Path | None = None, external_path: Path | None = None,
    out_dir: Path = REPORTS,
) -> Path:
    """Ejecuta la validación externa y devuelve la ruta del informe Markdown."""
    out_dir = Path(out_dir)
    fig_dir = out_dir / FIG_DIR_NAME
    model_file = Path(model_path) if model_path else MODELS / f"{ml.MODEL_NAME}.joblib"
    pipeline, meta = ml.load_model(model_file)
    raw, source = load_external(external_path)
    X_ext = prepare_features(raw)
    X_ref, y_ref = ml.load_dataset()

    holdout = ho.load_holdout(data_sha256=sha256_of(SILVER / SILVER_NAME))
    numeric = numeric_drift(X_ref, X_ext)
    for col in NUMERIC_FEATURES:
        numeric[col]["valores_ref"] = X_ref[col].dropna().tolist()
        numeric[col]["valores_ext"] = X_ext[col].dropna().tolist()
    categorical = categorical_drift(X_ref, X_ext)
    preds = prediction_drift(
        pipeline, X_ext, X_ref, y_ref,
        float(meta.get("threshold", ml.THRESHOLD)),
        probas_prueba=holdout["probas_prueba"],
    )
    if preds["n_prueba"] != holdout["metricas"]["n_test"]:
        raise RuntimeError("El artefacto de prueba y la evaluación congelada no coinciden")

    figures = make_figures(numeric, categorical, preds, fig_dir)

    serializable = {
        col: {k: v for k, v in d.items() if not k.startswith("valores_")}
        for col, d in numeric.items()
    }
    payload = {
        "generado_at": datetime.now(UTC).isoformat(),
        "tipo": "validacion_externa_no_supervisada",
        "advertencia": "Sin etiquetas en la cohorte externa: no se calculan metricas de desempeno.",
        "fuente": _display_path(source),
        "filas_externas": len(X_ext),
        "sha256_fuente": sha256_of(source),
        "modelo": _display_path(model_file),
        "sha256_modelo": meta["model_sha256"],
        "holdout": {
            "archivo": holdout["archivo"],
            "sha256": holdout["integridad_sha256"],
            "generado_at": holdout["generado_at"],
            "n_prueba": preds["n_prueba"],
        },
        "umbrales_senal": {"estable_max": STABLE_MAX, "moderada_max": MODERATE_MAX},
        "numericas": serializable,
        "categoricas": categorical,
        "predicciones": {k: v for k, v in preds.items() if not k.startswith("probas")},
        "figuras": [_display_path(p) for p in figures],
    }
    (out_dir / JSON_NAME).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    report = write_report(numeric, categorical, preds, payload, out_dir)
    flagged = [c for c, d in {**numeric, **categorical}.items() if d["senal"] != "estable"]
    log.info("Validación externa: %d pasajeros, %d figuras, deriva señalada en %s",
             len(X_ext), len(figures), flagged or "ninguna variable")
    log.info("Informe: %s", report)
    return report


if __name__ == "__main__":
    run()
