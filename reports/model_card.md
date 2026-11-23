# Ficha del modelo — `titanic_model_v1`

## Propósito y uso previsto

Estimar, para un pasajero del Titanic con información disponible **al embarcar**, la probabilidad de que
sobreviva y una etiqueta 0/1 con umbral 0,5. Uso previsto: **didáctico y de análisis histórico** —
ilustrar un flujo de ML con validación honesta y documentar el sesgo del dataset.

**No apto** para decidir servicios, indemnizaciones o prioridades hoy: usa `Sex` y `Pclass`, que
reproducen las reglas de evacuación de 1912. Predecir bien *lo que ocurrió* no implica que *deba* ocurrir
así. Tampoco es un modelo clínico, financiero ni de seguridad.

## Datos de entrenamiento

| Campo | Valor |
|---|---|
| Fuente | `https://paste.c-net.org/MarvelJinxed` (CSV 1.309 filas; 891 etiquetadas + 418 sin etiqueta); respaldo local: `TITANIC_LOCAL_PATH` |
| Capa usada | `data/silver/titanic_silver.parquet` |
| Tamaño | 891 pasajeros etiquetados; 179 (20 %) reservados para prueba, 712 para entrenamiento/validación |
| Huella SHA-256 de los datos | `40a0e8400ec1c07bf634e9df1c17bd26087e71c8f30ef1f1c1d2a6fce26bf0e4` |
| Periodo | Naufragio del RMS Titanic, abril de 1912 (un único evento histórico) |
| Semilla / división | `RANDOM_STATE=42`, estratificada, `TEST_SIZE=0.2` |
| Huella del artefacto | `f8aa74fd2afcd154989d5fd62d6c2b83481ad1f5f525d7342f0a782a612c2fa9` |

## Variables y pipeline

- **Numéricas:** `Age` (usa `AgeOriginal`, con NaN), `SibSp`, `Parch`, `Fare`, `Familia`.
- **Categóricas:** `Pclass`, `Sex`, `Embarked`, `TTitulo`, `Alone`.
- **Excluidas:** `Survived`/`Survivor` (objetivo), `PassengerId`/`Ticket`/`Name` (identificadores),
  `NSex`/`NEmbarked`/`NCabin`/`NTitulo` (orden arbitrario), `Person` (redundante), `Cabin`/`TCabin`/
  `HasCabin` (registro dudoso; la ablación no mostró ganancia, Paso 61).
- **Flujo:** `GroupAgeImputer(Sex, Pclass)` → mediana + escalado (numéricas) / moda + *one-hot* con
  `handle_unknown="infrequent_if_exist", min_frequency=10` (categóricas) → `LogisticRegression(C=0.1)`.
  Todo dentro de un `Pipeline`: imputación y categorías se aprenden solo en entrenamiento.
- **Modelo final:** reentrenado con los 891 pasajeros etiquetados una vez congeladas las decisiones.

## Métricas

Validación = media de 5 pliegues estratificados sobre los 712 de entrenamiento. Prueba = **una sola
evaluación** sobre 179 pasajeros, con intervalos bootstrap percentil (1.000 réplicas), congelada en
`reports/evaluacion_prueba.json` (huella de integridad verificada por las demás etapas).

| Métrica | Validación (CV) | Prueba (una vez) | IC 95 % prueba |
|---|---|---|---|
| ROC-AUC (principal) | 0,858 ± 0,024 | **0,848** | [0,784 – 0,904] |
| Exactitud | 0,792 | **0,816** | [0,760 – 0,872] |
| Precisión | 0,748 | 0,800 | — |
| Recall | 0,696 | 0,696 | — |
| F1 | 0,720 | 0,744 | — |

Referencias: dummy "siempre no" 0,617 de exactitud; regla de género 0,789 de exactitud y 0,715 de F1.
El modelo final supera la regla de género en exactitud y ofrece probabilidades ordenadas (AUC 0,848).
Brier score OOF: 0,141. Umbral elegido en validación: 0,5 (el óptimo F1 sería 0,32, pero aumentaría
falsos positivos sin un análisis de costos: decisión conservadora documentada).

## Desempeño por subgrupos (OOF, 712 pasajeros)

| Subgrupo | n | Tasa real | Exactitud | Precisión | Recall |
|---|---|---|---|---|---|
| Mujeres | 253 | 0,743 | 0,791 | 0,797 | 0,963 |
| Hombres | 459 | 0,185 | 0,808 | 0,429 | 0,106 |
| 1.ª clase | 171 | 0,649 | 0,766 | 0,859 | 0,766 |
| 2.ª clase | 150 | 0,447 | 0,873 | 0,914 | 0,791 |
| 3.ª clase | 391 | 0,243 | 0,790 | 0,571 | 0,547 |
| Niños | 66 | 0,591 | 0,652 | 0,735 | 0,641 |

El recall es mucho menor en hombres y en 3.ª clase: el modelo aprende la desigualdad histórica.
Los resultados en niños se basan en 66 personas (intervalos anchos).

## Limitaciones

- Muestra pequeña y un solo naufragio; sin validación temporal ni externa. Intervalos amplios.
- 177 edades (19,9 %) y 687 cabinas (77,1 %) faltantes; imputación y categoría `X` son supuestos.
- Variables sensibles (`Sex`, `Pclass`): su uso describe 1912, no debe trasladarse a decisiones actuales.
- Probabilidades calibradas solo en el rango medio (Brier 0,141); extremos conviene leerlos con cuidado.
- `.joblib` puede ejecutar código al cargarse: **cargar solo artefactos generados por este proyecto**.
  El metadato registra la versión de scikit-learn y avisa si cambia al cargar.

## Reproducibilidad y versiones

```bash
make install      # dependencias (requirements.lock.txt)
make test && make lint
make holdout      # única evaluación de los 179 -> reports/evaluacion_prueba.json (versionado)
make pipeline     # Bronze -> Silver -> Gold -> models/titanic_model_v1.joblib (lee el holdout)
make train        # solo reentrenamiento (lee el holdout)
```

Python 3.14.4 · scikit-learn 1.9.1 · pandas 3.1.1 · semilla 42.
Metadatos completos (fecha, versiones, hashes, parámetros y métricas):
`models/titanic_model_v1.metadata.json`. Cuadernos: `notebooks/02_exploracion.ipynb` y
`notebooks/04_modelado.ipynb`. Informes: `reports/conclusiones_eda.md` y `reports/problema_ml.md`.
Prueba de humo (Paso 69): 418 pasajeros sin etiqueta predicen sin NaN ni errores (ver cuaderno 04).

## Monitoreo posterior a la entrega

La cohorte de 418 pasajeros sin etiqueta se puntúa de nuevo con el artefacto guardado en
[`validacion_externa.md`](validacion_externa.md), que documenta la deriva de covariables y de
probabilidades (PSI, KS, TVD) frente a los datos de entrenamiento. La referencia de «prueba honesta»
es la distribución congelada en `evaluacion_prueba.json`, no un reentrenamiento. **No reporta
accuracy ni AUC**: sin etiquetas no existen aciertos que medir, y la métrica de prueba no se recalcula.
Las figuras de ROC/PR, calibración y confusión sobre predicciones OOF están en
[`validacion_grafica.md`](validacion_grafica.md). Todo se regenera con `make informes`.

