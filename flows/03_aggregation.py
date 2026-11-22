"""Agregaciones: Silver -> Gold. Cada función devuelve una tabla lista para un informe."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


def _as_table(df: pd.DataFrame) -> pd.DataFrame:
    """Índice como columna normal y nombres de columna texto (requisito Parquet)."""
    out = df.reset_index().copy()
    out.columns = [str(col) for col in out.columns]
    return out


def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Estadísticas descriptivas por grupos (Paso 28)."""
    return {
        "age_by_survival": _as_table(df.groupby("Survived")["Age"].describe()),
        "survival_by_sex": _as_table(
            df.groupby("Sex")["Survived"].agg(["mean", "sum", "count"])
        ),
        "survival_by_pclass": _as_table(
            df.groupby("Pclass")["Survived"].agg(["mean", "sum", "count"])
        ),
        "survival_by_alone": _as_table(
            df.groupby("Alone")["Survived"].agg(["mean", "sum", "count"])
        ),
    }


def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Prueba de independencia chi-cuadrado entre Sex y Survived (Paso 29)."""
    tabla = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p_value, dof, _ = stats.chi2_contingency(tabla)
    return {"chi2": float(chi2), "p_value": float(p_value), "dof": int(dof)}


def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia con Pclass en filas y Sex en columnas (Paso 34)."""
    return _as_table(df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean"))


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa y número de pasajeros por puerto de embarque (Paso 35)."""
    return _as_table(df.groupby("Embarked")["Survived"].agg(["mean", "count"]))


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa por puerto estratificada por clase (Paso 35) -- separa la confusión."""
    return _as_table(
        df.pivot_table(values="Survived", index="Embarked", columns="Pclass", aggfunc="mean")
    )


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Cuartiles de Fare calculados DENTRO de cada clase (Paso 36). No muta la entrada."""
    out = df.copy()
    out["FareQ"] = out.groupby("Pclass")["Fare"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop")
    )
    return out


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa de supervivencia por cuartil de tarifa y clase (Paso 36)."""
    return _as_table(
        add_fare_quartile(df).pivot_table(
            values="Survived", index="FareQ", columns="Pclass", aggfunc="mean"
        )
    )


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Tasa y número de pasajeros según tamaño de la familia (Paso 37)."""
    return _as_table(df.groupby("Familia")["Survived"].agg(["mean", "count"]))


def outlier_mask_iqr(s: pd.Series) -> pd.Series:
    """Máscara de atípicos según la regla IQR (Paso 38)."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Conteo de atípicos en Fare, Age y Familia + supervivencia de atípicos de Fare."""
    age_col = "AgeOriginal" if "AgeOriginal" in df.columns else "Age"
    rows = []
    for label, col in [("Fare", "Fare"), ("Age", age_col), ("Familia", "Familia")]:
        mask = outlier_mask_iqr(df[col])
        rows.append({
            "variable": label,
            "n_outliers": int(mask.sum()),
            "pct_outliers": float(mask.mean() * 100),
        })
    summary = pd.DataFrame(rows)
    fare_mask = outlier_mask_iqr(df["Fare"])
    summary["survival_rate_outliers"] = float("nan")
    summary["survival_rate_rest"] = float("nan")
    summary.loc[summary["variable"] == "Fare", "survival_rate_outliers"] = float(
        df.loc[fare_mask, "Survived"].mean()
    ) if fare_mask.any() else float("nan")
    summary.loc[summary["variable"] == "Fare", "survival_rate_rest"] = float(
        df.loc[~fare_mask, "Survived"].mean()
    )
    return summary


def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet((in_dir or SILVER) / SILVER_NAME)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        **group_stats(df),
        "survival_by_class_sex": survival_by_class_sex(df),
        "survival_by_embarked": survival_by_embarked(df),
        "survival_by_embarked_class": survival_by_embarked_class(df),
        "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),
        "survival_by_family": survival_by_family(df),
        "outlier_summary": outlier_summary(df),
    }
    for name, table in tables.items():
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "chi2_sex_survival.json").write_text(
        json.dumps(chi2_sex_survival(df), indent=2), encoding="utf-8"
    )
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir
