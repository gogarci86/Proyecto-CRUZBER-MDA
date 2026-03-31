# Diccionario de Variables: Modelo Predictivo CRUZBER (Dense Panel)
*Revisión de la Iteración 22/23 - Horizonte 12 Semanas*

El modelo se alimenta de una matriz densificada de **42 variables independientes (Features)** para predecir el futuro de miles de SKUs de forma simultánea. A continuación se desglosa el significado empírico y el "por qué" logístico de cada una.

---

### 📆 1. Bloque de Calendario y Estacionalidad (10 variables)
El objetivo de este bloque es que el algoritmo tenga noción del tiempo cronológico, detectando temporadas altas y parones poblacionales en España.

| Variable | Tipo | Descripción de Negocio | Por qué la usamos |
| :--- | :--- | :--- | :--- |
| `semana_anio` | Numérica (1-52) | El número de la semana actual dentro del año. | Es el indicador básico de la posición en el año civil. |
| `anio` | Numérica | El año en curso (ej. 2023). | Permite al modelo trazar tendencias macro (¿Cruzber vende globalmente más este año que hace tres?). |
| `mes` | Numérica (1-12) | El mes correspondiente a la semana actual. | Agrupación temporal "humana" para asociar comportamientos (Ej. Agosto = Vacaciones). |
| `trimestre` | Numérica (1-4) | El trimestre del año (Q1, Q2, Q3, Q4). | Estacionalidad comercial. Muchos distribuidores B2B hacen pedidos gordos a inicio de trimestre por cierres trimestrales de presupuesto. |
| `semana_del_mes` | Numérica (1-4) | Si es la primera o la última semana del mes. | Refleja ciclos de facturación: a veces los clientes B2B piden en la semana 1 cuando abren presupuesto mensual. |
| `es_fin_mes` | Binaria (0/1) | Vale 1 si es la semana de cierre de mes. | Detecta "Tensiones de cierre", donde comerciales de zona pueden empujar promociones para llegar a cuota. |
| `sem_sin` | Trigonométrica | Seno de la semana matemática. | Convierte el calendario en una "esfera". Evita que la IA crea que hay un abismo entre Diciembre y Enero. |
| `sem_cos` | Trigonométrica | Coseno de la semana matemática. | Pareja de `sem_sin` para completar la coordenada del calendario circular. |
| `temporada_alta` | Binaria (0/1) | Bandera fijada a 1 para meses de Abril a Septiembre. | Agrupa estadísticamente los meses calientes de outdoor (porta-bicis, ocio, turismo) frente al invierno (solo recambios). |
| `dias_laborables_semana` | Numérica (1-5) | Días que la fábrica y logística física operan descontando festivos. | **Crítica:** Justifica de inmediato caídas masivas en las líneas de pedido causadas por Puentes Nacionales o Semana Santa, evitando el pánico estadístico del modelo. |

---

### 🧠 2. Bloque de Memoria Auto-Regresiva (14 variables)
La "Memoria Muscular" de la predicción. Observa el desempeño pasado del mismo SKU.

| Variable | Tipo | Descripción de Negocio | Por qué la usamos |
| :--- | :--- | :--- | :--- |
| `lag_4w` / `lag_8w` / `lag_12w` | Numérica | Unidades que vendió este mismo artículo hace 1, 2 y 3 meses exactos. | Muestra si el producto viene de un pico reciente o está inactivo. Fija la tendencia a corto/medio plazo. |
| `lag_52w` | Numérica | Unidades vendidas exactamente en la misma semana del año pasado. | **El espejo interanual.** Captura picos de promociones recurrentes o necesidades puntuales del mismo mes del año anterior. |
| `roll_4w` / `roll_8w` / `12w` | Numérica | La Media Aritmética de ventas de los últimos X meses. | Alisa la curva. Si vendes [0, 40, 0, 0], el `roll` te dice que de *ritmo* base llevas 10. Elimina el ruido diario. |
| `roll_std_8w` / `12w` | Numérica | Desviación típica (Volatilidad) de las ventas recientes. | Si es alta, le indica a la probabilidad matemática que este artículo se vende a "latigazos" esporádicos. Fundamental para el perfil Lumpy. |
| `ewm_4w` / `8w` / `12w` | Numérica | Media Móvil Suavizada de forma Exponencial (EMA). | Al revés que un `roll` (que trata todos los días igual), la EMA le da mucho más valor a "lo que pasó ayer" que a "lo que pasó hace 2 meses". Reacciona antes a caídas súbitas del mercado. |
| `tendencia_4v4` | Numérica | Ratio o Cociente entre la Media Móvil actual vs la Media del mes anterior. | Es el acelerador del crecimiento a corto plazo: Si es > 1, el producto está cogiendo tracción rápida esta temporada. |
| `ratio_yoy` | Numérica | Ratio Year-over-Year (crecimiento sobre el año pasado). | Mide la salud estructural del artículo: ¿Este SKU está muriendo en el mercado (ratio < 1) o creciendo (ratio > 1) independientemente del mes en el que estemos? |

---

### 🌍 3. Bloque de Macroentorno y Clima (6 variables)
Conecta las líneas de facturación frías del ERP con el "Mundo Real" logístico en España.

| Variable | Tipo | Descripción de Negocio | Por qué la usamos |
| :--- | :--- | :--- | :--- |
| `temp_media` | Numérica | Temperatura promedio de España esa semana. | Estimula modelos predictivos de productos vacacionales. |
| `precip_mm` | Numérica | Precipitaciones medias de España esa semana. | Frena modelos outdoor y genera disrupciones en envíos logísticos por carretera (Flete). |
| `viento_max` | Numérica | Vel. del viento. | Correlaciona con consumo de cofres de techo aerodinámicos o accesorios paravientos. |
| `num_pruebas_cicl` / `dias_pruebas_cicl` | Numérica | Cantidad de pruebas de ciclismo profesional vivas esa semana. | Mide el "Market Exposure" pasivo en TV y prensa, que estimula al consumidor B2C a través del detallista B2B. |
| `hubo_prueba_cicl` | Binaria (0/1) | ¿Había alguna prueba ciclista ese fin de semana? | |

---

### 💰 4. Bloque Económico, S&OP y Taxonomía (12 variables)
Fija las reglas del negocio: "Cuánto vale hacerlo, cuánto esperamos vender y de qué familia es".

| Variable | Tipo | Descripción de Negocio | Por qué la usamos |
| :--- | :--- | :--- | :--- |
| `por_descuento2` | Numérica | % Medio de descuento aplicado esa semana a ese ítem. | Las IA son ingenuas; si ven un pico de ventas dirán "Qué gran producto". Con esta línea entienden: "Ah, se vendieron muchos porque estaban en liquidación/oferta". |
| `precio_unit` / `tarifa_nacional` | Numérica | Precio del componente y tarifa base del catálogo. | Dota a la IA de sensibilidad de Precio (Elasticidad). Es más fácil vender 100 ganchos de 2€ que 100 cofres de 400€. |
| `prevision_semanal` | Numérica | La Previsión Anual del Director Comercial partida por 52. | Si la IA no sabe qué hacer con un repuesto nuevo, se apoya en esto como "Faro" inicial para arrancar el pronóstico. |
| `factor_crecimiento` | Numérica | Multiplicador manual del Directorio. | Si Marketing sabe que va a empujar la Familia "Cofres 400L", fuerza un sesgo alcista general. |
| `CR_GamaProducto` / `CR_TipoProducto` | Categórica (Texto) | Gama y Tipo oficial del producto (Ej. "Aluminio_Pro"). | Ramifica los cerebros del algoritmo. Manda a los plásticos por un "árbol de decisión" y al aluminio estructurado por otro. |
| `CR_MaterialAgrupacion` | Categórica (Texto) | Elemento base (Acero, Plástico...). | Correlaciona con tiempos de extrusión de materia prima que afectan logísticamente a periodos enteros en roturas de stock B2B latentes. |
| `te_codigo_articulo` / `te_cr_gama` / `te_area_comp` | Target Encode (Num.) | Representación numérica matemática del nombre de la familia. | Evita enviar "texto puro" al regresor matemático. Permite sustituir "Cofre_Élite" por su peso y volumen estadístico equivalente histórico (Ej. *35.2*), salvando memoria RAM frente a One-Hot. |
