# Formulación del problema de ML y selección de modelos

Cuaderno: `notebooks/04_modelado.ipynb`. Código productivo: `flows/05_ml_models.py`.
**La evaluación final sobre prueba se ejecutó una sola vez**, después de congelar todas las decisiones.

## 1. Decisión del problema (Paso 39)

| Decisión | Respuesta y justificación |
|---|---|
| Tipo de problema | Clasificación binaria supervisada: la etiqueta `Survived` es 0/1. |
| Variable objetivo | `Survived` (1 = sobrevivió). `Survivor` es su copia textual: se excluye. |
| Unidad y momento | Un pasajero, **al embarcar**: solo se usan variables conocidas entonces. |
| Métrica principal | **ROC-AUC**: no depende del umbral y es adecuada con ~38 % de positivos; mide qué tan bien se ordenan los pasajeros por riesgo. |
| Métricas secundarias | Precisión, recall, F1 y exactitud; completan la visión del umbral operativo elegido. |
| Costo de errores | En una evacuación, un **falso negativo** (no identificar a quien sobrevivió) pierde una oportunidad de ayuda; un **falso positivo** desvía recursos escasos. Para una aseguradora, el primero subestima la exposición y el segundo sobreestima reservas. Sin una matriz de costos acordada, se reportan ambas tasas y se mantiene el umbral estándar 0,5. |
| Criterio de éxito | Superar la **regla de género** (exactitud ≈ 0,789, F1 ≈ 0,715 en entrenamiento) con un ROC-AUC de validación ≥ 0,80 y brecha entrenamiento–validación pequeña (< 0,02). |

> La exactitud sola engaña: `DummyClassifier(strategy="most_frequent")` ya acierta **0,617** en CV
> (siempre "no sobrevive"). Por eso la vara es la regla de género, no el dummy.

## 2. Auditoría de variables y fuga de datos (Pasos 40, 43–46)

| Columna(s) de Silver | ¿Existe antes del evento? | Riesgo | Decisión |
|---|---|---|---|
| `Survived`, `Survivor` | No (es el objetivo) | Fuga directa | Excluir |
| `PassengerId`, `Ticket`, `Name` | Sí | Identificadores: memorizan, no generalizan | Excluir; el título entra como `TTitulo` |
| `NSex`, `NEmbarked`, `NCabin`, `NTitulo` | Derivadas | Duplican categorías con orden arbitrario | Excluir; *one-hot* |
| `Age` imputada en Silver | — | Estadística calculada con **todos** los datos | `load_dataset()` usa `AgeOriginal` (NaN incluidos) |
| `Cabin`, `TCabin`, `HasCabin` | Dudoso | Pudo registrarse mejor para sobrevivientes | Evaluar con/sin ellas (Paso 61); **excluidas al final** |
| `Person` | Derivada de `Age`/`Sex` | Redundante | Excluir del modelo base |
| `Familia`, `Alone` | Sí | Derivadas de SibSp/Parch, sin fuga | Incluir |

**Demostración de fuga (Paso 43).** La media de edad por sexo calculada con todos los datos y solo con
entrenamiento difiere poco aquí (~20 % de faltantes en grupos grandes), pero el primer cálculo filtra
estadísticas de `X_test` hacia el entrenamiento: es un error de método aunque la cifra apenas cambie.

**Mecanismos anti-fuga implementados:**

1. `load_dataset()` reemplaza `Age` por `AgeOriginal`.
2. `GroupAgeImputer` aprende las medias por (Sex, Pclass) **solo en `fit()`**; grouping no visto cae a la
   media global de entrenamiento (probado en `tests/test_05_ml_models.py`).
3. Mediana + `StandardScaler` (numéricas) y moda + `OneHotEncoder(handle_unknown="infrequent_if_exist",
   min_frequency=10)` (categóricas), todo dentro del `Pipeline` de `build_pipeline()`.
4. División estratificada: 712 entrenamiento / 179 prueba, tasas 0,3834 / 0,3855, semilla 42.

## 3. Baselines (Paso 42)

| Regla | Exactitud | F1 |
|---|---|---|
| Dummy "siempre no" (CV sobre train) | 0,617 | 0,000 |
| Regla de género (`Sex == female`) | 0,789 | 0,715 |

## 4. Selección de modelos (Pasos 47–58)

Validación cruzada estratificada de 5 pliegues sobre los 712 pasajeros de entrenamiento (semilla 42).
ROC-AUC medio (desviación entre pliegues), brecha train−valid:

| Modelo | ROC-AUC valid | Desv. | Exactitud | Brecha train−valid |
|---|---|---|---|---|
| `hgb` (HistGradientBoosting) | **0,887** | 0,019 | 0,820 | **0,107** ⚠ |
| `forest` | 0,866 | 0,016 | 0,803 | 0,133 ⚠ |
| `tree` | 0,847 | 0,026 | 0,805 | 0,046 |
| `logreg` | 0,858 | 0,024 | 0,792 | **0,006** ✓ |
| `knn` | 0,841 | 0,017 | 0,787 | 0,088 |
| `dummy` | 0,500 | 0,000 | 0,617 | 0,000 |

**GridSearchCV** (mismos pliegues, métrica ROC-AUC):

| Modelo | Mejores parámetros | Mejora pareada vs. default |
|---|---|---|
| `hgb` | lr 0,1, depth 4, iter 200, l2 1,0 → 0,892 | +0,0057 ± 0,0103 (menor que la desviación: no concluyente) |
| `forest` | 400 árboles, leaf 3, depth ∞ → 0,882 | +0,0165 ± 0,0141 |
| `logreg` | C = 1 → 0,8584517 | C = 0,1 → 0,8584350: **empate** (Δ ≈ 2 × 10⁻⁵) |

**Decisión congelada.** Aunque `hgb` gana ~0,03 de AUC, memoriza el entrenamiento (brecha 0,107)
y su mejora por ajuste no supera el ruido entre pliegues. El uso previsto es un informe interpretable para
una aseguradora: se elige **`logreg` con C = 0,1** (la opción más regularizada entre las empatadas),
sin variables de cabina, umbral 0,5. Las curvas de aprendizaje confirman que la brecha es pequeña y que
más datos aportarían poco: el límite está en las variables disponibles (sesgo), no en la varianza.

Todas estas decisiones se tomaron mirando **solo** entrenamiento/validación; `X_test` permaneció cerrado.

## 5. Umbral, errores e interpretación (Pasos 51–61)

- **Matriz de confusión (OOF, umbral 0,5):** TN 381, FP 58, FN 83, TP 190 sobre 712 predicciones.
  Es decir, 83 pasajeros que sobrevivieron no fueron identificados y 58 que no sobrevivieron fueron
  marcados como supervivientes. La exactitud global (~0,80) esconde errores caros en ambos lados.
- **ROC y PR:** la curva precisión–recall es más informativa con 38 % de positivos; `logreg` y `hgb`
  se separan claramente del dummy.
- **Calibración:** Brier 0,141; el umbral que maximiza F1 en validación es **0,320** (F1 0,746 frente a
  0,729 con 0,5). Se **conserva 0,5**: bajar el umbral aumenta los falsos positivos y no hay análisis
  de costos que lo justifique. Decisión documentada, no optimizada contra prueba.
- **Errores por subgrupo (OOF):** mujeres recall 0,963 (n = 253) frente a hombres 0,106 (n = 459);
  niños exactitud 0,652 (n = 66) frente a mujeres adultas 0,803 (n = 218). El modelo reproduce la
  desigualdad histórica: predecir lo ocurrido **no** implica que deba ocurrir así.
- **Coeficientes (estandarizados)/permutación:** `Sex` (ΔAUC ≈ 0,088) y `Pclass` (0,067) dominan,
  seguidos de `Age` (0,037) y `TTitulo` (0,021). `Pclass=3` tiene el coeficiente más negativo (−0,760;
  OR 0,47) y `Pclass=1` el más positivo (+0,687; OR 1,99). Es **asociación predictiva**, no causalidad.
- **Ablación (ROC-AUC):** base 0,8585; + cabina 0,8629 (Δ +0,0044, menor que la desv. 0,024; y la
  cabina es un artefacto de registro) → **se excluye**; sin `TTitulo` 0,8568; sin `Sex` 0,8573
  (no mejoran, pero `Sex` queda marcado como variable sensible).

## 6. Evaluación final sobre prueba (Pasos 62–63) — una sola vez

Con las decisiones ya congeladas: `train_final` entrena con los 712 de entrenamiento y se evalúa una vez
en los 179 de prueba. **No se volvió atrás para ajustar nada tras ver estas cifras.**

| Modelo | ROC-AUC | Exactitud |
|---|---|---|
| Dummy (CV train) | 0,500 | 0,617 |
| Regla de género (train) | — | 0,789 |
| logreg (CV train) | 0,858 | 0,792 |
| **logreg (PRUEBA)** | **0,848** [IC 95 % 0,784–0,904] | **0,816** [IC 95 % 0,760–0,872] |

Precisión 0,800, recall 0,696, F1 0,744 (n = 179). Intervalos por bootstrap percentil (1.000 muestras).

## 7. Limitaciones (Paso 63) y ética

- 891 etiquetas, 179 en prueba: los intervalos son anchos (~±0,06 en AUC).
- Un solo naufragio; sin validación temporal ni externa.
- 177 edades y 687 cabinas faltantes: imputación y categoría `X` son supuestos documentados.
- `Sex` y `Pclass` reflejan las reglas de evacuación de 1912: **no usar** este modelo para decidir quién
  recibe un servicio, una indemnización o una prioridad hoy. Es una herramienta didáctica y un
  diagnóstico descriptivo del pasado, no una política.
- La métrica reportada es la de un único conjunto de prueba: repetir el experimento con otra semilla
  puede mover las cifras dentro del intervalo indicado.

## 8. Reproducibilidad

`make pipeline` regenera Bronze, Silver, Gold y el modelo con semilla 42. La ficha del artefacto está en
`reports/model_card.md`; los metadatos exactos, en `models/titanic_model_v1.metadata.json`.
