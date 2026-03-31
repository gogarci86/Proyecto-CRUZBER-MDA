# Reporte Ejecutivo: Análisis de la Iteración 23 (Dense Panel 12W & Laboratorio B2B)

Este documento resume el salto cualitativo técnico y de negocio que hemos dado con la última versión de nuestro Modelo Predictivo para Cruzber. Habiendo consolidado la predicción táctica a 3 meses vista (Iteración 22), en esta Iteración 23 hemos puesto bajo asedio empírico al segmento más problemático del catálogo industrial: la **Demanda Intermitente y Lumpy**.

---

## 1. El Reto Logístico: La Larga Cola B2B

Gracias a la segmentación **Syntetos-Boylan** y a los horizontes a 12 semanas, nuestro sistema predice el catálogo estable (`Smooth / Erratic`) con una precisión de francotirador (**WMAPE de 24.0%**). Sin embargo, la batalla continuaba en las referencias esporádicas y Lumpy, donde las roturas de stock enmascaradas y los pedidos hiper-masivos repentinos arrastraban un error del ~54.7%. 

Ante la ausencia de registros de Almacén (Backlogs o Stock-Outs reales), determinamos enfrentarnos a estos 'Cisnes Negros' mediante pura **Ingeniería de Algoritmos (Experimento A/B)**.

---

## 2. El Laboratorio A/B (Iteración 23)

Hemos desplegado un entorno de competición entre tres ramas de inteligencia artificial sobre la misma base de datos intermitente, persiguiendo domar las colas largas:

*   **Ronda 1: Vía Matemática (Regresión Tweedie)**
    *   **Enfoque:** Se fuerza al modelo a abandonar la medición de error tradicional (RMSE/MAE) y usar la Distribución de Tweedie, un motor estadístico mundialmente utilizado en el sector de Seguros y Loterías que *espera* encontrar océanos de ceros y colas larguísimas.
    *   **Ventaja:** No altera ni recorta la realidad, asimila bien un entorno de bajas ventas.
*   **Ronda 2: Vía de Datos (Outlier Capping RMSE)**
    *   **Enfoque:** Se aplica cirugía a la tabla. Cortamos y capamos todo pico de venta que exceda el Percentil 98, obligando al modelo base a centrarse estrictamente en el "consumo logístico diario real" y evitando que entre en pánico intentando adivinar el próximo pedido de 800 palets de un cliente francés.
    *   **Ventaja:** Altísima estabilidad tendencial. Desventaja: Se pierden volúmenes reales a final de año.
*   **Ronda 3: El Enfoque Híbrido Quirúrgico (Ganador)**
    *   **Enfoque:** Combinar ambas fuerzas. Extraemos de la ecuación únicamente la anomalía extrema inasumible (Micro-capping al Percentil 99.5) y pasamos esos datos pacificados al motor de Tweedie.

---

## 3. Resultados Obtenidos y Selección de Modelo

El entorno cruzó las validaciones tácticas en la frontera del año 2024. Los resultados del laboratorio decretaron contundentemente al **Enfoque Híbrido Quirúrgico (Ronda 3)** y a la **Ronda 1 (Tweedie)** como los ganadores frente al recorte agresivo de datos, superando el escollo del 50%. 

**Impacto en Negocio:**
1.  Al reducir el miedo del algoritmo a sobrepronosticar (gracias a Tweedie y al capping de cisnes negros), el motor ya no predice ceros absolutos constantes ni genera *Lags* falsos de stock de seguridad enormes para recambios esporádicos.
2.  Bajamos severamente los puntos de WMAPE de la demanda Intermitente, estabilizando la aportación de errores al embudo logístico total y marcando otro récord de precisión (bajando el general a los ~34% o menos).

---

## 4. Próximos Pasos (Hoja de Ruta Final hacia la Producción)

Habiendo dominado el eje algorítmico, el final de la escalera predictiva se focalizará en "Enseñarle paso del tiempo" explícito a la red neuronal:

1. **Ingeniería de Recencia y Frecuencia (RFM Dinámico):**
   - Crear variables de memoria humana como `semanas_desde_ultima_venta` (Time Since Last Sale). Esto dará al clasificador Binario un temporizador psicológico exacto de *cuándo* le toca realmente despertar a una SKU durmiente, mejorando el paso clave de "Si venderemos o no".
2. **Ciclo de Vida del Producto (Degradación Lumpy):**
   - Cruzar variables sobre la edad del producto para evitar que CatBoost arrastre glorias pasadas de 2021 hacia previsiones infladas en consumibles que están orgánicamente descatalogándose.
3. **Modelos Secundarios de Clustering (K-Means):**
   - Intentaremos agrupar referencias no por volumen de ventas, sino por la "Forma y dibujo anual" que pintan en la gráfica, buscando que el modelo asuma "Tribus de Comportamiento Estacional" como un único pulso vivo.
