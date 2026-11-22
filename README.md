# Proyecto Ciencia de Datos — Titanic: EDA y ML con buenas prácticas

Proyecto de taller de **Buenas Prácticas en Ciencia de Datos e IA**. Analiza el dataset del
Titanic (1912) para responder dos preguntas:

- **EDA:** ¿qué factores se asocian con la supervivencia de los pasajeros?
- **ML:** dadas las características de un pasajero al embarcar, ¿cuál es su probabilidad de
  sobrevivir y qué tan confiable es esa estimación?

El código sigue la arquitectura **medallion** (Bronze → Silver → Gold), separa responsabilidades
en módulos, mantiene la configuración fuera de la lógica, usa pruebas automáticas y evita la fuga
de datos con `Pipeline` y validación cruzada. La especificación completa está en
[`docs/GUIA.md`](docs/GUIA.md) y la instalación en [`docs/INSTALACION.md`](docs/INSTALACION.md).

## Inicio rápido

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env
make test
make pipeline
```

`make pipeline` ejecuta en orden: ingesta a Bronze, ingeniería de datos a Silver, agregaciones
a Gold y entrenamiento/guardado del modelo. Los artefactos generados quedan en `data/`, `models/`
y `reports/` (ignorados por Git).

## Comandos

| Comando | Acción |
|---|---|
| `make install` | Instala el paquete en modo editable con dependencias de desarrollo |
| `make ingest` | Descarga el CSV y lo guarda tal cual en `data/bronze/` |
| `make silver` | Valida, limpia y enriquece los datos en `data/silver/` |
| `make gold` | Genera tablas agregadas en `data/gold/` |
| `make train` | Evalúa y guarda `models/titanic_model_v1.joblib` con sus metadatos |
| `make pipeline` | Ejecuta todas las etapas anteriores |
| `make test` | Ejecuta pytest |
| `make lint` | Ejecuta ruff |
| `make clean` | Borra datos generados y cachés |

## Estructura

```text
flows/common/        configuración, rutas, logging y hashing
flows/00_ingestion.py          fuente -> Bronze (+ metadatos y reporte de calidad)
flows/01_data_engineering.py   Bronze -> Silver (funciones puras encadenadas)
flows/03_aggregation.py        Silver -> Gold (tablas para el informe)
flows/05_ml_models.py          Silver -> modelo guardado con metadatos
flows/job.py                   orquestador de etapas
notebooks/                     exploración (sin lógica productiva)
tests/                         pruebas unitarias de cada módulo
reports/                       informes y ficha del modelo (generados)
data/, models/                 artefactos derivados (ignorados por Git)
tareas/                        talleres anteriores, conservados como referencia
```

## Configuración

Copia `.env.example` a `.env`. Parámetros: `TITANIC_URL` (fuente remota), `TITANIC_LOCAL_PATH`
(ruta local de respaldo si la URL caduca), `TITANIC_N_LABELED` (891), `RANDOM_STATE`, `TEST_SIZE`
y `CV_FOLDS`. El `.env` no se versiona.

> Los archivos `.joblib` pueden ejecutar código al cargarse: carga únicamente modelos generados
> por este proyecto o de fuentes que controles.
