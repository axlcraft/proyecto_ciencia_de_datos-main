# Conclusiones del EDA — Titanic (informe para la aseguradora)

**Pregunta:** ¿qué factores se *asocian* con la supervivencia de los pasajeros?
**Datos:** 891 pasajeros etiquetados (Bronze, 12 columnas) + 418 sin etiqueta, de un total de 1.309.
Tasa global de supervivencia: **38,4 %** (342 de 891).
Cuaderno reproducible: `notebooks/02_exploracion.ipynb`; tablas: `data/gold/`.

## 1. Hallazgos (cada cifra con su n)

1. **El género es la asociación más fuerte.** Mujeres: **74,2 %** de supervivencia (233/314); hombres:
   **18,9 %** (109/577). χ² = 260,7, gl = 1, p ≈ 1,2 × 10⁻⁵⁸ (`data/gold/chi2_sex_survival.json`).
   *Asociación observada en este naufragio; no una regla universal ni una causalidad.*
2. **La clase social también marca una fuerte gradiente.** 1.ª: **63,0 %** (136/216); 2.ª: **47,3 %**
   (87/184); 3.ª: **24,2 %** (119/491). Al cruzar clase y género, mujeres de 1.ª llegan al **96,8 %**
   (n = 144) y hombres de 3.ª solo al **13,5 %** (n = 347): el efecto del género **no es igual en todas
   las clases** (interacción).
3. **"Mujeres y niños primero" se cumple en general, con matices.** Niños (Edad < 16): **59,0 %**
   (49/83); mujeres adultas: **75,7 %** (205/271); hombres: **16,4 %** (88/537). Pero en 3.ª clase la
   supervivencia femenina cae a **50,0 %** (n = 144) y los intervalos de los niños de 1.ª son muy
   anchos (pocos casos).
4. **Viajar acompañado se asocia con más supervivencia — hasta cierto tamaño.** Con familia: **50,6 %**
   (179/354); solo: **30,4 %** (163/537). Por tamaño: 1–3 familiares → 55,3 % (n = 161), 57,8 %
   (n = 102) y 72,4 % (n = 29); 5–10 → 13,6 % (n = 22), 33,3 % (n = 12), 0 % (n = 6) y 0 % (n = 7).
   Los grupos grandes son pequeños: el descenso es sugerente, no preciso.
5. **La tarifa no es un efecto aislado de la clase.** Dentro de 1.ª, de cuartil 1 a 4 la tasa va
   46,3 % → 57,4 % → 74,5 % → 73,6 %; pero `Fare` y `Pclass` están muy correlacionados y la tarifa
   es asimétrica (media 32,2, mediana 14,5, skew 4,8). 116 tarifas (13,0 %) son atípicas por IQR y
   sobrevivió el **68,1 %** de ellas frente al 33,9 % del resto: son sobre todo pasajeros de 1.ª.
   **Atípico no significa erróneo ni protector.**

## 2. Decisiones de limpieza y su sesgo

- **Edad imputada por género (177 filas, 19,9 %).** La desviación estándar baja de **14,53 a 13,01**:
  177 valores se sustituyeron por promedios de grupo, reduciendo la variabilidad artificialmente y
  creando un pico central falso en el histograma. Se conservó `AgeOriginal` (con NaN) y `AgeImputada`
  para auditar; el modelo **vuelve a imputar** dentro del `Pipeline` con estadísticas de entrenamiento.
- **`Master` → `Mr` (títulos).** Se consolidaron títulos raros en 4 categorías: Mr 557, Miss 186,
  Mrs 127, Otro 21. `Master` designa niños varones: al fusionarlo con `Mr` se mezclan niños y adultos.
  Es defendible **solo** como simplificación documentada, parcialmente compensada por `Person`, que sí
  usa la edad original.
- **Cabin desconocida → `X` (687 filas, 77,1 %).** No se borraron: `X` es una categoría explícita y
  reproducible (`DECKS = ABCDEFGTX` fijo). Los registros de cubierta podrían estar ligados a la
  supervivencia (quien sobrevivió pudo declarar mejor su cabina): tiene **77,1 % de missing**, no es
  información fiable.
- **`Person` sin edad.** Regla: `Age < 16` → `child`; si no, `Sex` (NaN < 16 es `False`). Ninguno de los
  177 pasajeros sin edad se clasifica como niño. Con un 11,6 % de niños entre quienes sí tienen edad
  (83/714), faltarían ~**21 `child`**: la tasa infantil observada puede estar diluida en `male`/`female`.
- **`Embarked` (2 nulos)** se rellenó con la moda `S`; `Fare` (1 nulo en los 1.309 registros) queda fuera
  del subconjunto etiquetado y se imputa en el pipeline para inferencia.

## 3. Confusiones detectadas y cómo se separaron

- **Puerto vs. clase.** En bruto, Cherbourg parece mejor (55,4 %, n = 168) que Southampton (33,9 %,
  n = 646); pero Cherbourg traía 50,6 % de pasajeros de 1.ª y Queenstown 93,5 % de 3.ª. Al estratificar
  por clase, la ventaja se reduce dentro de cada clase (3.ª: C 37,8 % vs S 19,0 %). El puerto mezclaba
  composición social con destino: la estratificación separa ambos efectos.
- **Tarifa vs. clase.** Se calcularon cuartiles de `Fare` **dentro de cada `Pclass`** para que la tarifa
  no actuara como proxy de clase.
- **Tamaño de familia vs. `Alone`.** Se mantuvieron las dos vistas: `Alone` agrupa (sí/no) y `Familia`
  detalla el número; las tasas por tamaño muestran que el efecto no es monotonía.
- **Correlaciones `N…`.** Las codificaciones ordinales imponen un orden arbitrario: el signo de
  `NCabin` en la matriz de correlación no tiene interpretación sustantiva. Se usan solo para explorar,
  nunca en el modelo (*one-hot*).

## 4. Qué NO podemos afirmar

- **Asociación ≠ causalidad.** Que las mujeres sobrevivieran más no significa que el género *causara* la
  supervivencia ni que un pasajero con las mismas características sobreviviría en otro contexto.
- **Un solo naufragio.** Es un evento histórico único, con 891 etiquetas; no hay validación temporal
  ni externa. Los intervalos de confianza de los subgrupos pequeños son anchos.
- **Datos incompletos y posible sesgo de registro.** 177 edades, 687 cabinas y 2 puertos faltantes;
  la imputación, la categoría `X` y la regla de `Person` son supuestos documentados. El p-valor del χ²
  indica que la asociación difícilmente es azar; **no** indica que sea grande, causal o transferible.
- **Uso indebido.** Este EDA describe lo que ocurrió en 1912; no debe emplearse para decidir quién
  recibe un servicio hoy. `Sex` y `Pclass` reproducen las reglas de evacuación de la época.

La formulación del problema predictivo está en `reports/problema_ml.md`.

## 5. Evidencia gráfica

Las cifras de este informe se pueden contrastar visualmente en
[`validacion_grafica.md`](validacion_grafica.md): tasas de supervivencia por género, clase y
`Alone` con intervalos de confianza al 95 %, y el efecto de la imputación de edad sobre la
distribución original. Las imágenes se regeneran con `make figuras` desde `data/silver/`; no son
capturas manuales. El cuaderno ejecutado correspondiente está en
`reports/ejecutados/02_exploracion.ejecutado.ipynb` (los originales de `notebooks/` se conservan
limpios).
