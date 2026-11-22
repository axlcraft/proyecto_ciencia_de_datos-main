"""Ingeniería de datos: Bronze -> Silver.

Cada transformación es una función pura: recibe un DataFrame y devuelve uno
nuevo. El orden importa y está fijado en build_silver().
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE,
    BRONZE_LABELED_NAME,
    SILVER,
    SILVER_FINAL_NAME,
    SILVER_NAME,
)

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]
DECKS = list("ABCDEFGTX")            # orden FIJO: X = desconocida (reproducible)
TITLE_MAP = {                        # consolidación de títulos (Paso 26)
    "Master": "Mr", "the Countess": "Mrs", "Dona": "Mrs", "Mme": "Mrs",
    "Lady": "Miss", "Mlle": "Miss", "Ms": "Miss",
}
MAIN_TITLES = ["Mr", "Mrs", "Miss"]

SILVER_REQUIRED = [
    "Survived", "Pclass", "Age", "Fare", "Embarked", "Person", "Alone", "TCabin",
    "TTitulo", "NSex", "NEmbarked", "NCabin", "NTitulo",
]


def validate_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """Contrato de entrada: columnas esperadas y PassengerId único."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Bronze incompleto; faltan columnas: {missing}")
    if df["PassengerId"].duplicated().any():
        raise ValueError("Bronze inválido: PassengerId duplicados")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Tipos explícitos: conteos como int, medidas como float."""
    out = df.copy()
    for col in ["Survived", "Pclass", "SibSp", "Parch"]:
        # Los pasajeros sin etiqueta tienen Survived vacío: no se puede convertir a int.
        if out[col].notna().all():
            out[col] = out[col].astype(int)
    for col in ["Age", "Fare"]:
        out[col] = out[col].astype(float)
    return out


def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Person': 'child' si Age < 16; si no, el valor de Sex.

    Regla para edad desconocida (NaN): se usa Sex. Esto subestima a los niños
    sin edad registrada; el sesgo se cuantifica en el Reto 8.1.
    """
    out = df.copy()
    out["Person"] = np.where(out["Age"] < 16, "child", out["Sex"])
    return out


def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'TCabin' (primera letra de Cabin, 'X' si es nula) y 'HasCabin' (0/1)."""
    out = df.copy()
    out["TCabin"] = out["Cabin"].str[0].fillna("X")
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    return out


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Familia' = SibSp + Parch y 'Alone' ('with Family' si Familia > 0)."""
    out = df.copy()
    out["Familia"] = out["SibSp"] + out["Parch"]
    out["Alone"] = np.where(out["Familia"] > 0, "with Family", "Alone")
    return out


def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Survivor' mapeando 0 -> 'No' y 1 -> 'Yes' (solo para etiquetar)."""
    out = df.copy()
    out["Survivor"] = out["Survived"].map({0: "No", 1: "Yes"})
    return out


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el título del nombre: 'Braund, Mr. Owen Harris' -> 'Mr'."""
    out = df.copy()
    out["TTitulo"] = out["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    return out


def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Consolida títulos raros usando TITLE_MAP y MAIN_TITLES ('Otro')."""
    out = df.copy()
    out["TTitulo"] = out["TTitulo"].replace(TITLE_MAP)
    out["TTitulo"] = out["TTitulo"].where(out["TTitulo"].isin(MAIN_TITLES), "Otro")
    return out


def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena Embarked con la moda observada."""
    out = df.copy()
    out["Embarked"] = out["Embarked"].fillna(out["Embarked"].mode()[0])
    return out


def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa 'Age' con el promedio por 'Sex'.

    Conserva la edad original en 'AgeOriginal' (con NaN) y marca las filas
    imputadas en 'AgeImputada' (bool). Usada para EXPLORAR; el modelo (Parte 11)
    vuelve a imputar solo con datos de entrenamiento.
    """
    out = df.copy()
    out["AgeOriginal"] = out["Age"]
    out["AgeImputada"] = out["Age"].isna()
    promedios = out.groupby("Sex")["Age"].mean()
    out["Age"] = out["Age"].fillna(out["Sex"].map(promedios))
    return out


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Codificaciones numéricas ordinales para el análisis (no para el modelo)."""
    out = df.copy()
    out["NEmbarked"] = out["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    out["NSex"] = out["Sex"].map({"female": 0, "male": 1})
    deck_map = {letter: i for i, letter in enumerate(DECKS)}
    out["NCabin"] = out["TCabin"].map(deck_map)
    out["NTitulo"] = out["TTitulo"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Otro": 3})
    return out


def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una COPIA sin Name, Sex, Embarked y Cabin."""
    return df.drop(columns=["Name", "Sex", "Embarked", "Cabin"]).copy()


def validate_silver(df: pd.DataFrame) -> pd.DataFrame:
    """Contrato de salida: sin nulos en las columnas críticas (AgeOriginal sí puede)."""
    nulls = {col: int(n) for col, n in df[SILVER_REQUIRED].isnull().sum().items() if n}
    if nulls:
        raise ValueError(f"Silver inválido; hay nulos en columnas críticas: {nulls}")
    return df


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Orden deliberado: Person usa la edad ORIGINAL, así que va antes de imputar."""
    return (
        bronze
        .pipe(validate_bronze)
        .pipe(cast_types)
        .pipe(add_person)
        .pipe(add_cabin_features)
        .pipe(add_family_features)
        .pipe(add_survivor_label)
        .pipe(add_title)
        .pipe(consolidate_titles)
        .pipe(impute_embarked)
        .pipe(impute_age)
        .pipe(encode_categoricals)
    )


def run(in_dir: Path = BRONZE, out_dir: Path = SILVER) -> Path:
    bronze = pd.read_csv(Path(in_dir) / BRONZE_LABELED_NAME)
    silver = validate_silver(build_silver(bronze))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_dir / SILVER_NAME, index=False)
    drop_original_columns(silver).to_parquet(out_dir / SILVER_FINAL_NAME, index=False)
    log.info("Silver listo: %d filas, %d columnas", *silver.shape)
    return out_dir / SILVER_NAME
