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
| `make holdout` | Ejecuta **una sola vez** la evaluación de los 179 pasajeros y congela `reports/evaluacion_prueba.json`; corridas posteriores solo verifican su integridad |
| `make train` | Guarda `models/titanic_model_v1.joblib` con metadatos; consume la métrica congelada, sin reabrir la prueba |
| `make pipeline` | Ejecuta todas las etapas anteriores |
| `make notebooks` | Ejecuta los cuadernos y guarda copias con salidas en `reports/ejecutados/` (los originales quedan limpios) |
| `make figuras` | Genera `reports/figuras/*.png` y `reports/validacion_grafica.md` desde Silver y el pipeline congelado |
| `make validar-externa` | Puntúa los 418 pasajeros sin etiqueta y documenta la deriva en `reports/validacion_externa.md` (+ `.json`) |
| `make informes` | Ejecuta `figuras` y `validar-externa` |
| `make test` | Ejecuta pytest |
| `make lint` | Ejecuta ruff |
| `make clean` | Borra datos generados y cachés |

> **Sobre el conjunto de prueba:** `make holdout` es la única puerta que lo abre. Guarda métricas,
> intervalos y probabilidades en `reports/evaluacion_prueba.json` (versionado); `make pipeline`,
> `make informes` y `make notebooks` solo leen ese artefacto y validan sus huellas. Si cambian los
> datos o las decisiones del modelo, fallan en lugar de reevaluar en silencio.

> **Sobre la validación externa:** los 418 pasajeros de Bronze no están etiquetados, así que solo
> permite vigilar deriva de covariables y de predicciones. **No** se calculan accuracy ni ROC-AUC y
> **no** se reabre el conjunto de prueba. Las figuras, los informes y el artefacto de prueba sí se
> versionan (excepciones en `.gitignore`); los datos, modelos y el resto de derivados siguen ignorados.

## Estructura

```text
flows/common/        configuración, rutas, logging y hashing
flows/00_ingestion.py          fuente -> Bronze (+ metadatos y reporte de calidad)
flows/01_data_engineering.py   Bronze -> Silver (funciones puras encadenadas)
flows/03_aggregation.py        Silver -> Gold (tablas para el informe)
flows/05_ml_models.py          Silver -> modelo guardado con metadatos
flows/06_external_validation.py  418 sin etiqueta -> informe de deriva
flows/07_report_figures.py     Silver + pipeline -> figuras de validación gráfica
flows/08_holdout_evaluation.py evaluación única -> evaluacion_prueba.json (congelada)
flows/job.py                   orquestador de etapas
notebooks/                     exploración (sin lógica productiva, sin salidas)
tests/                         pruebas unitarias de cada módulo
reports/                       informes .md, .png y cuadernos ejecutados (versionados)
data/, models/                 artefactos derivados (ignorados por Git)
tareas/                        talleres anteriores, conservados como referencia
```

## Configuración

Copia `.env.example` a `.env`. Parámetros: `TITANIC_URL` (fuente remota), `TITANIC_LOCAL_PATH`
(ruta local de respaldo si la URL caduca), `TITANIC_N_LABELED` (891), `RANDOM_STATE`, `TEST_SIZE`
y `CV_FOLDS`. El `.env` no se versiona.

> Los archivos `.joblib` pueden ejecutar código al cargarse: carga únicamente modelos generados
> por este proyecto o de fuentes que controles.
