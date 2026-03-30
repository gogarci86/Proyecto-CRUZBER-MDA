# Valoración Experta del Proyecto CRUZBER — Predicción de Demanda Semanal

**Fecha de revisión:** 30 de marzo de 2026
**Evaluador:** Head of Data & Machine Learning — CRUZBER
**Alcance:** Evaluación completa del proyecto, datasets, iteraciones y recomendaciones de continuación

---

## 1. Resumen Ejecutivo

El proyecto tiene un **objetivo de negocio bien definido y relevante**: predecir la demanda semanal por SKU con horizonte de 4 semanas para optimizar la producción en España. La iniciativa está técnicamente bien concebida, metodológicamente sistemática y ha demostrado superar ampliamente la previsión interna de CRUZBER (MAPE 67% vs 211% en el benchmark interno).

Sin embargo, el proyecto presenta una **desconexión crítica** entre la arquitectura de iteraciones 1-14 (nivel SKU×Municipio) y la nueva estrategia de notebooks 15-16 (nivel Nacional×Canal), que ha supuesto una regresión significativa en los indicadores de error. Los mejores resultados del proyecto (MAPE 18.3%) siguen estando en la iteración 8, publicada hace varios meses.

**Valoración global: 6.5 / 10**

---

## 2. Análisis del Problema y los Datasets

### 2.1 Formulación del Problema

| Aspecto | Evaluación | Comentario |
|---|---|---|
| Objetivo de negocio | ✅ Correcto | "¿Cuánto fabricar de qué SKU en las próximas 4 semanas?" es una pregunta de producción nacional |
| Granularidad temporal | ✅ Correcto | Semana es la unidad natural para producción y distribución |
| Granularidad de SKU | ✅ Correcto | Nivel correcto para planning |
| Foco en España | ✅ Justificado | Exportación tiene dinámicas distintas y series más irregulares |
| Filtrado de obsoletos | ✅ Implementado | 47.7% del catálogo eliminado correctamente (NB15-16) |
| Horizonte 4 semanas | ⚠️ Parcialmente | El horizonte se menciona en el objetivo pero la validación no siempre modela 4 semanas ahead de forma explícita |
| Métrica objetivo | ⚠️ Revisable | MAPE penaliza mucho productos de bajo volumen; MAE ponderado puede ser más alineado al negocio |

### 2.2 Datasets Disponibles — Evaluación de Idoneidad

**Datos internos (Datos Internos/):**

- **LineasAlbaranCliente.xlsx** (106 MB, 938K filas, 2021-presente): Es el corazón del proyecto. Contiene el histórico transaccional completo. **Muy bueno** — cubre 4+ años con granularidad suficiente.
- **MaestroArticulos.xlsx**: Contiene clasificación ABC, tarifa nacional y otros atributos. **Clave** para segmentar modelos.
- **MaestroClientes.xlsx**: Geografía y tipo de canal. **Esencial** para mapeado regional.
- **Agrupacion Canales venta.xlsx**: Define TipoCruz → Canal (Tradicional/FLEET/ND). **Bien utilizado** en NB15-16.
- **MaestroFamilias.xlsx + Familias Articulos.xlsx**: Jerarquía de producto. **Infrautilizado** hasta ahora — podría mejorar la generalización en SKUs con poco historial.

**Datos externos:**

- **clima_semanal_openmeteo.csv** (424 KB): Temperatura, precipitación, viento por semana. Importancia ~4% en modelos actuales. Bien incorporado pero con impacto modesto.
- **Calendario Ciclismo 22_24.xlsx**: Calendario de carreras 2022-2024. **Solo cubre 3 años y solo hasta 2024** — habría que extender a 2025 si hay datos de test más recientes.

**Datasets procesados:**

- **Dataset_España_base.csv** (121 MB): Dataset base España limpio. Intermediario bien construido.
- **df_final_modelado.csv** / **df_final_modelado_it6.csv** (27-28 MB): Dataset para modelado con features adicionales.
- **df_españa_limpio.csv** (127 MB): Duplicado aparente de Dataset_España_base — revisar si es necesario mantener ambos (consume 121+127 = 248 MB innecesariamente).
- **Dataset_para_Modelos.csv** (127 MB): Idem — hay solapamiento de archivos intermedios que complica la trazabilidad.

**Alerta crítica sobre los datasets procesados**: Existe una inconsistencia entre los datasets de iteraciones 1-14 (basados en `Dataset_España_base.csv`, ~252K filas) y los de NB15-16 (basados en `LineasAlbaranCliente.xlsx`, empezando desde 938K filas brutas). Los primeros agregan por semana×municipio×artículo; los segundos son un rebuild completo. Esta bifurcación dificulta la comparación de métricas entre iteraciones.

---

## 3. Evolución por Iteraciones — Análisis Crítico

### 3.1 Iteraciones 1-5 (Notebooks 03-08): Construcción de la Baseline

**Partida:** MAE 0.793 (It1) → **MAE 0.641 | MAPE 26.0%** (It5)

| It | Mejora clave | MAE | MAPE | R² | Valoración |
|---|---|---|---|---|---|
| 1 | CatBoost baseline | 0.793 | — | 0.295 | Buena elección de algoritmo |
| 2 | Rolling mean 4 semanas | 0.773 | — | 0.330 | Correcto — capture de tendencia reciente |
| 3 | Lag año anterior | 0.769 | — | 0.330 | Impacto mínimo, pero correcto conceptualmente |
| 4 | **Transformación log1p** | 0.649 | 26.3% | 0.287 | **Mayor ganancia individual** (+15.6% MAE) |
| 5 | Target encoding + Optuna | 0.641 | 26.0% | 0.288 | Mejora marginal |

**Análisis crítico It1-5:**
- La transformación log1p fue el hallazgo más valioso — correcta para demanda con cola larga y muchos ceros.
- R² de 0.288 es bajo para un modelo de producción, pero es esperable con datos SKU×municipio muy granulares.
- La variable `por_descuento2` (correlación 0.206, efecto ×4.7x sobre demanda) se identifica como crítica pero **aún no está en el modelo en It5**. Error estratégico que retrasa 2 iteraciones el mayor lever disponible.
- El análisis de importancia de variables es riguroso: precio (16%), target encoding (15.9%), rolling mean (12.8%) explican la lógica del modelo.

### 3.2 Iteraciones 6-8 (Notebooks 09-13): Descuentos y Regionalización

**Evolución:** MAE 0.641 → MAPE **18.3%** (regional tipo A)

| It | Mejora clave | MAPE | Valoración |
|---|---|---|---|
| 6 | Añade `por_descuento2` + regiones | ~24% | Correcto — finalmente el mayor lever |
| 7 | **Modelos separados A vs B&C** | 21.5% (municipal) | Excelente decisión arquitectural |
| 8 | **Agregación regional** | **18.3%** | Mejor resultado del proyecto |

**Análisis crítico It6-8:**
- La separación en modelos A (alta rotación) vs B/C (esporádico) es **la mejor decisión arquitectural del proyecto**. Son problemas estadísticamente distintos que no deben mezclarse.
- La arquitectura de dos capas (Municipal + Regional) tiene coherencia: producción usa regional, distribución usa municipal. **Bien diseñado.**
- MAPE 18.3% a nivel regional tipo A es un resultado **sólido para forecasting de demanda** de productos físicos.
- **Canarias emerge como caso especial**: MAPE 38.1% en It8, peor que en It7 (33.5%). La agregación regional no ayuda en islas por la inversión estacional turística. Correcto detectarlo.
- **Norte: MAPE 28.1%** — alta variabilidad, también necesita tratamiento especial.
- El modelo Hurdle para B/C (propuesto pero **no implementado**) es una idea técnicamente sólida que se abandona en los notebooks siguientes.

### 3.3 Iteración 9 (Notebook 14): Mejoras de Features

**Objetivo:** +16 nuevas variables temporales, lags extendidos, tendencias. Métricas no publicadas en el notebook de conclusiones.

**Análisis crítico It9:**
- Los 13 lags adicionales (semanas 2, 3, 6, 8, 12) y las ventanas rolling de 2, 6, 12 semanas son técnicamente correctos.
- El `ratio_yoy` (semana actual vs mismo periodo año anterior) es especialmente valioso para un negocio con estacionalidad ciclista.
- **Problema**: No hay un notebook de conclusiones It9 equivalente a NB08 o NB13 que documente los resultados finales obtenidos. Se pierde la trazabilidad.

### 3.4 Notebook 15: Nueva Estrategia Nacional×Canal — EVALUACIÓN CRÍTICA

**Esta es la iteración más problemática del proyecto.**

#### Contexto del cambio estratégico
El notebook 15 supone una **ruptura completa con la arquitectura anterior**: abandona el enfoque SKU×Municipio/Región (construido durante 9 iteraciones) para adoptar un modelo SKU×Nacional×Canal. El razonamiento de negocio es correcto — producción es una decisión nacional, no regional. Pero la ejecución presenta graves problemas.

#### Resultados obtenidos

| Modelo | Objetivo | MAE | MAPE | R² | Overfitting |
|---|---|---|---|---|---|
| **A - Nacional** | Todos los canales | 2.552 | **59.2%** | 0.642 | 37.7% |
| **B - Tradicional×Región** | Canal tradicional | 0.878 | **42.8%** | 0.520 | Moderado |
| **C - FLEET Nacional** | Canal flota | 4.392 | **88.6%** | 0.413 | Alto |

**Cross-validation Modelo A:** Folds 83.7% / 56.6% / 51.9% → Media **64.0% ± 14.0%** — Alta varianza entre folds: señal de instabilidad.

#### Problemas identificados en NB15

1. **Regresión de MAPE severa**: De 18.3% (It8, regional) a 42.8-88.6% (NB15). Aunque los niveles de agregación no son idénticos, el deterioro es demasiado pronunciado para explicarse solo por diferencia de granularidad. El modelo nacional debería ser **más fácil** de predecir (demanda más suave), no más difícil.

2. **FLEET es un problema diferente**: El canal FLEET (corporativo/contratos) tiene una demanda episódica basada en contratos, no en patrones de consumo. Aplicarle el mismo modelo que a Tradicional no tiene sentido estadístico. MAPE 88.6% lo confirma.

3. **Pérdida de features acumuladas**: El rebuild desde cero en NB15 no incorpora todos los avances de features de iteraciones 1-9 (target encoding, rolling features extendidas, variables de tendencia). Se regresa a un modelo básico.

4. **Overfitting alto (37.7%)**: RMSE train=6.133 vs test=8.443. El modelo memoriza 2022-2023 y falla en 2024. Los hiperparámetros de NB15 no están suficientemente regularizados.

5. **Métricas por región incongruentes**: Noreste tiene MAPE 52.3% en NB15 vs 13.5% en It8. Esta divergencia extrema sugiere diferencias en el pipeline de datos, no solo en el nivel de agregación.

6. **Benchmark interno mal interpretado**: El MAPE 165.4% de la previsión interna de CRUZBER probablemente usa el mismo cálculo ingenuo que infla el MAPE con productos de bajo volumen. La comparación es válida como referencia pero no como métrica absoluta de calidad.

#### Lo positivo de NB15

- Correcto al separar FLEET del resto (aunque el modelo FLEET necesita un enfoque diferente).
- El filtrado de artículos obsoletos (47.7%) es un paso adelante.
- Partir desde `LineasAlbaranCliente.xlsx` asegura reproducibilidad.
- La arquitectura Nacional + Tradicional×Región comienza a alinearse mejor con la toma de decisiones de producción.

### 3.5 Notebook 16: v2 Mejoras — EVALUACIÓN CRÍTICA

#### Resultados obtenidos vs esperados

| Modelo | MAPE v1 | MAPE v2 | Delta real | Delta esperado |
|---|---|---|---|---|
| A - Nacional | 59.2% | 61.2% | **+2.0 pp** ❌ | −20 pp |
| B - Tradicional | 42.8% | 43.1% | **+0.3 pp** ❌ | −15 pp |
| C - FLEET | 88.6% | 87.2% | −1.4 pp ✅ | −20 pp |

**Las cuatro mejoras propuestas (M1-M4) no produjeron el impacto esperado:**

**M1 - Split ABC:** Implementado. NAC-A: 67.0% MAPE | NAC-BC: 52.4% MAPE | NAC-Naive: 68.9% MAPE. La separación es correcta pero los MAPEs individuales siguen siendo altos. El split reduce la mezcla de señales pero no soluciona el overfitting subyacente.

**M2 - Historial 5 años (2020-2024):** Dataset ampliado de 509K a 741K filas (+46%). Sin embargo, 2020 es un año atípico (COVID) que puede introducir ruido. No se documenta si se controla este sesgo.

**M3 - Regularización conservadora (depth=4, lr=0.05, l2=10):** El overfitting en NAC-A cae de 37.7% a 33.6% — mejora marginal. Los parámetros son demasiado conservadores: depth=4 puede estar infraajustando para la complejidad del problema.

**M4 - Filtro modelables (≥8 semanas):** 580 SKUs a forecast naïve. Buen concepto pero el forecast naïve tiene MAPE 68.9% y R²=-0.24, lo que significa que predice peor que la media. Sugiere que la media histórica no es el mejor baseline para estos SKUs.

**Problema estructural de NB16:** El R² del Modelo B cae de 0.520 (v1) a **0.452** (v2). Regularizar excesivamente puede llevar a underfitting. El modelo con más datos y más regularización explica MENOS varianza, lo que indica que algo en el pipeline v2 está introduciendo ruido (posiblemente el año 2020 COVID).

---

## 4. Análisis Transversal — Problemas Sistémicos

### 4.1 Discontinuidad arquitectural NB14 → NB15

El proyecto tiene **dos líneas de desarrollo paralelas e inconexas**:

```
Línea A (NB01-14): SKU × Municipio/Región × Tiempo
   └── Mejor resultado: MAPE 18.3% (It8, regional, tipo A)

Línea B (NB15-16): SKU × Nacional × Canal × Tiempo (rebuild from scratch)
   └── Mejor resultado: MAPE 42.8% (Modelo B, Tradicional)
```

Esta discontinuidad **no está justificada en los notebooks**. No hay un documento que explique por qué se abandona la línea A con sus resultados superiores. La hipótesis más probable es que la línea A no estaba alineada con el problema de producción (que es nacional), pero la línea B no ha incorporado aún las mejoras de features de la línea A.

### 4.2 La métrica MAPE está distorsionada

El MAPE tiene una debilidad conocida para demanda intermitente o de bajo volumen: un error de 1 unidad en un SKU que vende 2 unidades/semana representa un MAPE del 50%. Con muchos SKUs de bajo volumen (tipo B/C), el MAPE agregado se infla artificialmente.

**Evidencia**: NAC-Naive tiene MAPE 68.9% pero MAE 1.521. Para un SKU que históricamente vende 2 ud/semana, 1.521 de error es razonable, pero el MAPE asociado es brutal.

**Recomendación**: Reportar también **WMAPE** (weighted MAPE, ponderado por volumen) y **SMAPE** como métricas complementarias. Evaluar el negocio en euros de inventario mal planificado, no solo en MAPE.

### 4.3 El horizonte de 4 semanas no está modelado explícitamente

El objetivo de negocio es predecir 4 semanas hacia el futuro. Sin embargo, los modelos actuales entrenan con lag_1w, lag_4w, etc., pero no fuerzan explícitamente el horizonte de predicción en la validación. Esto puede producir modelos que son buenos a 1 semana pero degradan rápidamente a semana 3-4.

**Recomendación**: Implementar **walk-forward validation** con ventanas de 4 semanas: train hasta semana T, predecir semanas T+1, T+2, T+3, T+4, avanzar una semana, repetir. Reportar MAPE por horizonte (1w, 2w, 3w, 4w).

### 4.4 Artículos obsoletos — verificación

Se confirma el filtrado de artículos obsoletos (47.7% del catálogo en NB15-16). Sin embargo, en las iteraciones 1-14 **no queda documentado explícitamente** que este filtro estuviera activo. Si los modelos de It1-8 incluían artículos obsoletos, sus métricas están infladas con ruido, lo que podría explicar parte del aparente deterioro en NB15 (que sí los filtra).

---

## 5. Recomendaciones para Reducir MAPE y MAE

### 5.1 Correcciones inmediatas (alto impacto, bajo esfuerzo)

**R1 — Unificar las dos líneas de desarrollo**

No continuar el desarrollo en paralelo. La arquitectura objetivo debe ser:
- **Modelo de producción**: SKU × Nacional × Canal (línea B de NB15-16)
- **Modelo de distribución**: SKU × Región (línea A It8)

Pero la línea B debe incorporar **todas las features de la línea A** (rolling means extendidos, lags año anterior, target encoding, variables de tendencia, variables temporales cíclicas). Esto solo debería recuperar 10-15 pp de MAPE.

**R2 — Separar FLEET completamente**

FLEET no es un problema de forecasting de demanda, es un problema de gestión de contratos. Recomendaciones:
- Excluir FLEET del modelo principal
- Para FLEET: usar un modelo de reglas + CRM (contratos activos × volumen histórico por contrato)
- Si se mantiene ML: modelo dedicado con features de contratos (nº de contratos activos, renovaciones, vencimientos)

**R3 — Cambiar el baseline para SKUs naïve**

El forecast naïve actual usa la media histórica (R²=-0.24). Mejorar con:
- **Suavizado exponencial simple** (SES) para SKUs esporádicos
- **Croston's method** para demanda intermitente (vende algunos meses, no todos)
- Esto puede reducir el MAPE del segmento naïve de 68.9% a ~40-50%

**R4 — Controlar el año 2020 en el historial**

Si se usa historial 2020-2024, **crear una feature binaria `es_covid` para 2020** (o directamente excluir 2020 y empezar en 2021). La demanda de 2020 tiene patrones anómalos que perjudican el aprendizaje de estacionalidad normal.

### 5.2 Mejoras técnicas (alto impacto, esfuerzo medio)

**R5 — Implementar walk-forward validation por horizonte**

```python
# Estructura recomendada para evaluar el modelo a 4 semanas
for semana_corte in range(semana_inicio_test, semana_fin_test):
    modelo.fit(data[data.semana < semana_corte])
    for h in [1, 2, 3, 4]:
        pred = modelo.predict(data[data.semana == semana_corte + h])
        metricas[h].append(calcular_mape(pred))
```
Esto revelará si el modelo se degrada en horizontes 3-4 (lo más probable) y guiará las mejoras.

**R6 — Implementar el modelo Hurdle para B/C (propuesto en It8, nunca ejecutado)**

Dos pasos:
1. **Clasificador**: ¿Habrá venta esta semana? (XGBoost binario, threshold calibrado)
2. **Regresor**: Si sí, ¿cuántas unidades? (CatBoost, entrenado solo en semanas con venta > 0)

Estimación de mejora: −5 a −8 pp MAPE en segmento B/C, que hoy está en 52.4%.

**R7 — Feature: Previsión de ventas interna de CRUZBER**

El maestro de artículos contiene `PrevisionVentasAA` (previsión anual interna). Esta feature contiene conocimiento experto del equipo comercial. Incorporarla como feature externa puede capturar información no observable en el histórico (lanzamientos de nuevos productos, cambios de precio planificados, acciones promocionales futuras).

**R8 — Feature: Jerarquía de producto como regularización**

Usar `CR_GamaProducto` y `CR_TipoProducto` (de Familias Articulos) para hacer target encoding a nivel de familia en lugar de solo a nivel de SKU. Esto ayuda en SKUs con poco historial al "tomar prestada" experiencia de productos similares.

**R9 — Modelo dedicado para Canarias**

Ya identificado en It8 y planificado en `create_it10.py`. Canarias tiene:
- Estacionalidad inversa (turismo verano = más demanda de ciclismo)
- Ausencia del efecto "temporada de carreras peninsular"
- Distribución logística diferente (insular)

El script `create_it10.py` ya está escrito — **ejecutarlo y evaluar** es prioritario.

### 5.3 Cambios estratégicos (alto impacto, esfuerzo alto)

**R10 — Incorporar el calendario de promociones futuras**

La variable `por_descuento2` tiene el mayor impacto conocido (×4.7x sobre demanda). Si el equipo comercial puede compartir el calendario de promociones planificadas, el modelo tendría acceso a la señal más potente con anticipación.

**R11 — Considerar modelos de series temporales puros para SKUs tipo A**

Para los SKUs A (alta rotación, patrones estables), un ensemble de:
- CatBoost (modelo actual, captura efectos de features)
- **Prophet** o **N-BEATS** (captura estacionalidad múltiple explícitamente)

El ensemble de los dos puede reducir el error al capturar lo que cada modelo pierde.

**R12 — Métricas de negocio**

Adicional al MAPE/MAE técnico, reportar:
- **Coste de sobreproducción** (stock no vendido × margen)
- **Coste de infraproducción** (rotura de stock × margen perdido)
- **Días de inventario** proyectados con el modelo vs sin él

Esto permite justificar el ROI del proyecto más allá de los indicadores técnicos.

---

## 6. Hoja de Ruta Recomendada

| Prioridad | Acción | Notebooks afectados | MAPE esperado |
|---|---|---|---|
| 🔴 Crítico | Ejecutar `create_it10.py` (modelo Canarias) | NB17 | Canarias: 30% → 22% |
| 🔴 Crítico | Unificar features It1-14 en arquitectura NB15 | NB17 | Nacional: 61% → 40-45% |
| 🔴 Crítico | Excluir/controlar año 2020 en datos | NB16 (fix) | B Tradicional: 43% → ~38% |
| 🟠 Alto | Separar FLEET del modelo principal | NB18 | FLEET: fuera del MAPE agregado |
| 🟠 Alto | Walk-forward validation 4 semanas | NB18 | Visibilidad real del error |
| 🟠 Alto | Mejorar baseline naïve (Croston) | NB18 | Naïve: 69% → ~45% |
| 🟡 Medio | Modelo Hurdle para B/C | NB19 | B/C: 52% → 44% |
| 🟡 Medio | Incorporar `PrevisionVentasAA` como feature | NB19 | Global: −3 a −5 pp |
| 🟡 Medio | Target encoding por familia de producto | NB19 | Mejora en SKUs nuevos/raros |
| 🟢 Largo plazo | Calendario de promociones (input externo) | — | −10 a −15 pp |
| 🟢 Largo plazo | Ensemble CatBoost + Prophet para tipo A | — | −5 pp tipo A |

**Objetivo alcanzable en las próximas 3-4 iteraciones: MAPE Tradicional <30%, Nacional <45%**

---

## 7. Síntesis de Puntos Fuertes y Débiles

### Puntos Fuertes ✅

1. **Metodología iterativa rigurosa**: 17 notebooks bien numerados, 3 documentos de conclusiones intermedias, evolución trazable.
2. **Elección de algoritmo correcta**: CatBoost es el mejor algoritmo para datos tabulares con muchas categorías y distribuciones asimétricas.
3. **Transformación log1p**: Decisión técnicamente excelente para demanda con cola larga.
4. **Detección temprana del lever de descuentos**: Identificado en It1-5, incorporado en It6.
5. **Arquitectura de dos capas (producción + distribución)**: Conceptualmente sólida y alineada con el negocio.
6. **Filtrado de obsoletos**: Implementado en NB15-16.
7. **Benchmark vs previsión interna**: Demostrar que ML supera en 2.8x-3x a la previsión manual es un argumento de negocio poderoso.
8. **Separación A vs B/C**: La mejor decisión técnica del proyecto.

### Puntos Débiles ⚠️

1. **Regresión de MAPE en NB15-16**: Se pasa de 18.3% a 42.8-88.6%. No documentado ni justificado en los notebooks.
2. **Dos líneas de desarrollo inconexas**: Features de It1-14 no incorporadas en NB15-16.
3. **El horizonte de 4 semanas no está validado explícitamente**.
4. **FLEET incluido en el MAPE agregado**: Penaliza la métrica global con un problema fundamentalmente diferente.
5. **Año 2020 (COVID) en historial**: Sin control del sesgo por pandemia.
6. **Modelo Hurdle para B/C nunca implementado**: Propuesto en It8, prometedor, abandonado.
7. **Falta notebook de conclusiones para It9**.
8. **Acumulación de datasets intermedios**: ~500 MB de CSVs intermedios sin documentar claramente su proveniencia y vigencia.
9. **Métricas de negocio ausentes**: Solo MAPE/MAE técnicos, sin traducción a impacto económico.

---

## 8. Valoración Final

| Dimensión | Puntuación | Comentario |
|---|---|---|
| Formulación del problema | 8/10 | Bien definido, alineado al negocio, SKU-nivel correcto |
| Calidad de datos | 7/10 | Buenos datos internos, externos limitados, obsoletos filtrados en NB15 |
| Metodología ML | 7/10 | CatBoost, log1p, segmentación A/B/C son decisiones excelentes |
| Features engineering | 6/10 | Progresivo pero discontinuado al reescribir en NB15 |
| Resultados técnicos | 5/10 | MAPE 18.3% (It8) es bueno, pero regresión en NB15-16 preocupa |
| Consistencia de iteraciones | 5/10 | Dos líneas inconexas, falta conclusión It9, v2 no mejora v1 |
| Alineación negocio-modelo | 7/10 | Canal+Nacional en NB15 es más alineado con producción |
| Documentación | 6/10 | Buena en general, huecos en NB14-16 |

### **Puntuación Global: 6.5 / 10**

El proyecto está bien concebido, ha generado insights valiosos y tiene potencial para llegar a MAPE <25% en el segmento principal. La principal debilidad no es técnica sino de gestión del desarrollo: la ruptura estratégica de NB15 sin incorporar el conocimiento acumulado ha supuesto una regresión importante. Con las correcciones propuestas en la sección 5, el objetivo de MAPE <30% en canal Tradicional es alcanzable en 2-3 iteraciones.

---

*Documento generado: 30 de marzo de 2026 — Head of Data & ML, CRUZBER*
