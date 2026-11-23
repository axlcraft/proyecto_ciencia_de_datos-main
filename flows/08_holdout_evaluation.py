"""Evaluación final del holdout: única puerta que abre los 179 pasajeros de prueba.

El resultado se guarda en ``reports/evaluacion_prueba.json`` y se versiona. Las
demás etapas (entrenamiento, informes, cuadernos) solo consumen ese artefacto:
no vuelven a entrenar con los 712 ni a calcular métricas sobre los 179. Si los
datos o las decisiones cambian, la guardia por hash obliga a declarar un nuevo
experimento en lugar de reescribir silenciosamente la evaluación.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import sklearn

from flows.common.config import settings
from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import REPORTS, ROOT, SILVER, SILVER_NAME

ml = importlib.import_module("flows.05_ml_models")

log = get_logger(__name__)

HOLDOUT_NAME = "evaluacion_prueba.json"
HOLDOUT_PATH = REPORTS / HOLDOUT_NAME


def _display(path: Path) -> str:
    """Ruta relativa al repositorio; evita rutas absolutas en el artefacto."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def decisions() -> dict:
    """Decisiones congeladas que identifican a esta evaluación."""
    num, cat = ml._final_features()
    return {
        "estimator": ml.FINAL_ESTIMATOR,
        "params": dict(ml.FINAL_PARAMS),
        "features": {"numeric": num, "categorical": cat},
        "target": ml.TARGET,
        "threshold": ml.THRESHOLD,
        "use_cabin_features": ml.USE_CABIN_FEATURES,
        "random_state": settings.random_state,
        "test_size": settings.test_size,
        "split": "train_test_split estratificado",
    }


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _integrity_hash(payload: dict) -> str:
    sin_hash = {k: v for k, v in payload.items() if k != "integridad_sha256"}
    return hashlib.sha256(_canonical(sin_hash).encode("utf-8")).hexdigest()


def load_holdout(path: Path | None = None, data_sha256: str | None = None) -> dict:
    """Lee el artefacto congelado y valida integridad, datos y decisiones."""
    source = Path(path) if path else HOLDOUT_PATH
    if not source.exists():
        raise FileNotFoundError(
            f"No existe {source}: ejecuta `make holdout` una vez sobre Silver "
            "para generar la evaluación final antes de entrenar o reportar."
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("integridad_sha256") != _integrity_hash(payload):
        raise ValueError(f"{source} fue modificado: la huella de integridad no coincide.")

    if data_sha256 is not None and payload["datos"]["sha256"] != data_sha256:
        raise RuntimeError(
            f"Los datos cambiaron desde la evaluación congelada ({source}). "
            "Ese conjunto de prueba ya no es el mismo: no se reabre ni se sobreescribe; "
            "define un nuevo experimento/versión de modelo."
        )
    if payload.get("decisiones") != decisions():
        raise RuntimeError(
            f"Las decisiones del modelo cambiaron desde la evaluación congelada ({source}). "
            "No se puede reutilizar la métrica de prueba: define un nuevo experimento."
        )
    return payload


def run(
    data_path: Path | None = None, out_path: Path | None = None, force: bool = False,
) -> Path:
    """Evalúa una sola vez; las corridas posteriores solo verifican el artefacto.

    ``force`` permite regenerar explícitamente el artefacto cuando se acepta que
    es una evaluación nueva (por ejemplo, tras cambiar la versión del modelo).
    """
    data = Path(data_path) if data_path else SILVER / SILVER_NAME
    out = Path(out_path) if out_path else HOLDOUT_PATH
    data_sha = sha256_of(data)

    if out.exists():
        try:
            load_holdout(path=out, data_sha256=data_sha)
        except (ValueError, RuntimeError) as exc:
            if not force:
                raise RuntimeError(
                    f"{out} ya existe pero no es compatible: {exc} "
                    "Usa `--force` solo si aceptas iniciar una evaluación nueva."
                ) from exc
        else:
            log.info("Holdout ya congelado en %s (integridad verificada): no se reevalúa", out)
            return out

    X, y = ml.load_dataset(data)
    X_tr, X_te, y_tr, y_te = ml.split(X, y)
    pipe_eval = ml.train_final(X_tr, y_tr)
    metrics, proba = ml.evaluate_test_with_ci(pipe_eval, X_te, y_te, ml.THRESHOLD)

    payload = {
        "tipo": "evaluacion_final_holdout",
        "generado_at": datetime.now(UTC).isoformat(),
        "archivo": _display(out),
        "datos": {
            "archivo": _display(data),
            "sha256": data_sha,
            "n_total": len(X),
            "n_train": len(X_tr),
            "n_test": len(y_te),
        },
        "decisiones": decisions(),
        "metricas": metrics,
        "probas_prueba": [float(v) for v in proba],
        "entorno": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    payload["integridad_sha256"] = _integrity_hash(payload)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log.info(
        "Evaluación final congelada en %s: ROC-AUC %.3f, exactitud %.3f (n=%d)",
        out, metrics["roc_auc"], metrics["accuracy"], metrics["n_test"],
    )
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruta", type=Path, default=None, help="ruta alternativa del artefacto")
    parser.add_argument("--force", action="store_true", help="forzar una evaluación nueva")
    args = parser.parse_args(argv)
    run(out_path=args.ruta, force=args.force)


if __name__ == "__main__":
    main()
