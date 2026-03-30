Eres el Head of Data & ML de CRUZBER, proyecto de predicción de demanda semanal por SKU.

CONTEXTO DEL PROYECTO:
- Notebook activo: 17_Modelado_v3_Features_Consolidadas.ipynb (v3, recién ejecutado)
- Arquitectura: Nacional × Canal + Tradicional × Región, modelo CatBoost, shift=4 (horizonte honesto)
- Métricas v3 obtenidas tras ejecutar NB17: [PEGAR AQUÍ los resultados de df_res_nac y df_res_trad]
- Métricas v2 baseline: NAC-COMB MAPE=61.2% | TRAD-COMB MAPE=43.1%
- Archivos de datos en ../Datasets/Datos Internos/

TAREA:
Genera el notebook 18_Modelado_v4_Hurdle_BC.ipynb siguiendo la nomenclatura del proyecto.

MEJORA A IMPLEMENTAR — MODELO HURDLE PARA TIPO B/C:
Los SKUs tipo B/C tienen demanda intermitente (muchas semanas en cero). Un modelo de regresión
pura no es óptimo. El modelo Hurdle tiene dos etapas:
  1. CLASIFICADOR (binario): ¿habrá venta esta semana? → CatBoostClassifier, threshold calibrado con PR-curve
  2. REGRESOR (cantidad): si la respuesta es sí, ¿cuántas unidades? → CatBoostRegressor entrenado
     SOLO sobre semanas con unidades > 0

Predicción final = P(venta) × cantidad_predicha

DETALLES TÉCNICOS:
- Usar los mismos datasets y features de NB17 (importar o replicar el pipeline)
- El clasificador usa como target: (unidades > 0).astype(int)
- El regresor usa como target: log1p(unidades), filtrado a filas donde unidades > 0
- Evaluar por separado: accuracy del clasificador (F1, precision, recall) + MAE/MAPE del regresor
- Comparar la predicción final combinada vs el modelo BC puro de v3 (benchmark)
- Aplicar tanto a NAC-BC como a TRAD-BC
- Mantener shift=4 en todas las features (horizonte honesto)
- El threshold del clasificador debe optimizarse para maximizar F1 sobre validación, no usar 0.5 fijo

MÉTRICAS A REPORTAR:
- Clasificador: F1, Precision, Recall, AUC-ROC, matriz de confusión
- Modelo combinado: MAPE, WMAPE, MAE (comparado con v3-BC)
- Análisis de semanas donde el clasificador falla más (semanas de transición de temporada)

FORMATO:
- Markdown extenso en lenguaje de negocio (no técnico) explicando qué es un modelo Hurdle y por qué es mejor para demanda intermitente
- Celdas de código limpias, comentadas
- Cuadro comparativo final: v3-BC vs v4-Hurdle por región y tipo ABC
- Guardar modelos entrenados en ../Artifacts/ con nombres modelo_clasificador_bc.cbm y modelo_regresor_bc.cbm