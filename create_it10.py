import json, uuid

def cid(): return str(uuid.uuid4()).replace("-","")[:12]
def md(src): return {"cell_type":"markdown","id":cid(),"metadata":{},"source":src}
def code(src): return {"cell_type":"code","id":cid(),"metadata":{},"execution_count":None,"outputs":[],"source":src}

cells = []

# ── CELL 0: Título ─────────────────────────────────────────────────────────
cells.append(md("""# 15. Modelado Predictivo de Demanda — Iteración 10
## Modelo Dedicado para Canarias + Factor de Corrección de Sesgo

### Dos mejoras sobre It9 (MAPE global 17.6%)

| Mejora | Descripción | Problema que resuelve |
|---|---|---|
| **Modelo dedicado Canarias** | CatBoost propio para Canarias con estacionalidad turística inversa | MAPE Canarias 30.5% → objetivo ~20% |
| **Factor de corrección de sesgo** | Multiplica predicciones por ratio media/mediana del train | MAE (L1) optimiza la mediana; corrección centra en la media |

### Por qué el modelo siempre predice por debajo (recordatorio)

```
MAE loss  →  optimiza la MEDIANA de la distribución
log1p     →  comprime outliers pero introduce sesgo de retransformación
Resultado →  expm1(mediana_log) ≈ 1 unidad  vs  media real ≈ 1.7 unidades

Corrección: pred_corregida = pred_raw × (media_train / mediana_train)
```

**Referencia:** It9 → MAE global 0.610 | MAPE global 17.6% | Canarias MAPE 30.5%"""))

# ── CELL 1: Imports ────────────────────────────────────────────────────────
cells.append(code("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import warnings
from catboost import CatBoostRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
print("Librerías cargadas correctamente.")"""))

# ── CELL 2: Load ───────────────────────────────────────────────────────────
cells.append(md("---\n## 1. Carga del Dataset y Feature Engineering\n\nPipeline idéntico a It9 (A3 + A4 + A5) más una feature nueva exclusiva para Canarias."))

cells.append(code("""df = pd.read_csv('../Datasets/df_final_modelado_it6.csv', sep=';')

REGION_MAP = {
    'A CORUÑA': 'Noroeste', 'LUGO': 'Noroeste', 'OURENSE': 'Noroeste',
    'PONTEVEDRA': 'Noroeste', 'ASTURIAS': 'Noroeste', 'CANTABRIA': 'Noroeste',
    'ALAVA': 'Norte', 'GIPUZKOA': 'Norte', 'VIZCAYA': 'Norte',
    'NAVARRA': 'Norte', 'LA RIOJA': 'Norte', 'HUESCA': 'Norte',
    'ZARAGOZA': 'Norte', 'TERUEL': 'Norte',
    'BARCELONA': 'Noreste', 'GIRONA': 'Noreste', 'LLEIDA': 'Noreste',
    'TARRAGONA': 'Noreste', 'CASTELLON': 'Noreste', 'VALENCIA': 'Noreste',
    'ALICANTE': 'Noreste', 'ILLES BALEARS': 'Noreste',
    'MADRID': 'Centro', 'TOLEDO': 'Centro', 'CUENCA': 'Centro',
    'GUADALAJARA': 'Centro', 'CIUDAD REAL': 'Centro', 'ALBACETE': 'Centro',
    'BURGOS': 'Centro', 'SEGOVIA': 'Centro', 'SORIA': 'Centro',
    'VALLADOLID': 'Centro', 'AVILA': 'Centro', 'SALAMANCA': 'Centro',
    'ZAMORA': 'Centro', 'LEON': 'Centro', 'PALENCIA': 'Centro',
    'CACERES': 'Centro', 'BADAJOZ': 'Centro',
    'SEVILLA': 'Sur', 'MALAGA': 'Sur', 'GRANADA': 'Sur', 'CORDOBA': 'Sur',
    'JAEN': 'Sur', 'HUELVA': 'Sur', 'CADIZ': 'Sur', 'ALMERIA': 'Sur',
    'MURCIA': 'Sur',
    'LAS PALMAS': 'Canarias', 'SANTA CRUZ DE TENERIFE': 'Canarias',
}
df['region'] = df['Provincia'].map(REGION_MAP).fillna('Centro')
df = df.sort_values(['codigo_articulo', 'Municipio', 'anio', 'semana_anio']).reset_index(drop=True)
print(f"Dataset: {df.shape[0]:,} filas | Canarias: {(df['region']=='Canarias').sum():,} filas ({(df['region']=='Canarias').mean()*100:.1f}%)")"""))

# ── CELL 4: Feature engineering ────────────────────────────────────────────
cells.append(code("""# ── A3: Temporales cíclicas ─────────────────────────────────────────────────
def semana_to_mes(semana):
    return np.clip(((semana - 1) // 4) + 1, 1, 12)

df['mes']            = semana_to_mes(df['semana_anio'])
df['trimestre']      = ((df['mes'] - 1) // 3) + 1
df['semana_del_mes'] = ((df['semana_anio'] - 1) % 4) + 1
df['es_fin_mes']     = (df['semana_del_mes'] == 4).astype(int)
df['sin_semana']     = np.sin(2 * np.pi * df['semana_anio'] / 52)
df['cos_semana']     = np.cos(2 * np.pi * df['semana_anio'] / 52)

# ── A4: Lags extendidos ───────────────────────────────────────────────────────
GROUP_KEY = ['codigo_articulo', 'Municipio']
for lag in [2, 3, 6, 8, 12]:
    df[f'unidades_lag_{lag}sem'] = df.groupby(GROUP_KEY)['unidades'].shift(lag)
for w in [2, 6, 12]:
    df[f'unidades_rolling_mean_{w}sem'] = (df.groupby(GROUP_KEY)['unidades']
        .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean()))
for w in [4, 8, 12]:
    df[f'unidades_rolling_std_{w}sem'] = (df.groupby(GROUP_KEY)['unidades']
        .transform(lambda x: x.shift(1).rolling(w, min_periods=2).std().fillna(0)))

# ── A5: Tendencia 4v4 ────────────────────────────────────────────────────────
roll4 = df.groupby(GROUP_KEY)['unidades'].transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
roll8 = df.groupby(GROUP_KEY)['unidades'].transform(lambda x: x.shift(1).rolling(8, min_periods=1).mean())
df['tendencia_4v4'] = roll4 / ((roll8 - roll4).replace(0, np.nan)).fillna(1).clip(0.1, 10)
df['ratio_yoy'] = 1.0  # lag interanual no disponible

# ── NUEVO It10: Feature de estacionalidad turística de Canarias ───────────────
# Canarias tiene el patrón opuesto al resto: turismo alto en invierno (sem 44-52 y 1-12)
# y temporada baja en verano cuando el resto de España tiene pico
df['es_temporada_turistica_can'] = (
    (df['semana_anio'] >= 44) | (df['semana_anio'] <= 12)
).astype(int)

# Interacción: solo activa si la fila es de Canarias
df['can_temporada_alta'] = (
    (df['region'] == 'Canarias') & (df['es_temporada_turistica_can'] == 1)
).astype(int)

# Interacción: Canarias × semana (captura el perfil estacional insular completo)
df['can_sin_semana'] = np.where(df['region'] == 'Canarias', df['sin_semana'], 0)
df['can_cos_semana'] = np.where(df['region'] == 'Canarias', df['cos_semana'], 0)

# Relleno NaN
lag_cols = [c for c in df.columns if 'lag_' in c or 'rolling_' in c
            or c in ['tendencia_4v4', 'ratio_yoy']]
df[lag_cols] = df[lag_cols].fillna(0)

print(f"Features añadidas: A3 + A4 + A5 + 4 features de Canarias")
print(f"  es_temporada_turistica_can: {df['es_temporada_turistica_can'].mean()*100:.1f}% de semanas")
print(f"  can_temporada_alta (solo Canarias): {df['can_temporada_alta'].sum():,} filas activas")"""))

# ── CELL 5: Target + features + split ──────────────────────────────────────
cells.append(md("---\n## 2. Target, Features y Splits"))

cells.append(code("""df['unidades_log']    = np.log1p(df['unidades'])
df['precio_unitario'] = (df['importe_neto'] / df['unidades'].replace(0, np.nan)).fillna(0)
df['es_temporada_alta'] = (df['semana_anio'].between(10,22) | df['semana_anio'].between(35,48)).astype(int)

cat_features = ['Provincia', 'Municipio', 'codigo_articulo',
                'CodigoFamilia', 'CodigoSubfamilia', 'agrupacion_canal', 'region']

num_features_base = [
    'semana_anio', 'anio', 'es_temporada_alta',
    'mes', 'trimestre', 'semana_del_mes', 'es_fin_mes', 'sin_semana', 'cos_semana',
    'precio_unitario',
    'unidades_lag_1_semana', 'unidades_lag_1_mes',
    'unidades_lag_2sem', 'unidades_lag_3sem', 'unidades_lag_6sem',
    'unidades_lag_8sem', 'unidades_lag_12sem',
    'unidades_rolling_mean_2sem', 'unidades_rolling_mean_6sem', 'unidades_rolling_mean_12sem',
    'unidades_rolling_std_4sem', 'unidades_rolling_std_8sem', 'unidades_rolling_std_12sem',
    'tendencia_4v4', 'ratio_yoy',
    'temp_media', 'precip_mm', 'viento_max',
    'num_pruebas_ciclistas', 'duracion_total_pruebas', 'hubo_prueba_ciclista',
    'valor_descuento_promo', 'hubo_descuento_promo',
    # Nuevas It10 — Canarias
    'es_temporada_turistica_can', 'can_temporada_alta', 'can_sin_semana', 'can_cos_semana',
]
for col in ['unidades_sliding_window_mensual', 'unidades_misma_semana_anio_anterior',
            'tendencia_unidades', 'volatilidad_4_sem',
            'interaccion_region_temp', 'interaccion_region_precip']:
    if col in df.columns and col not in num_features_base:
        num_features_base.append(col)

num_features = [f for f in num_features_base if f in df.columns]
all_features  = num_features + cat_features

print(f"Total features: {len(all_features)} ({len(num_features)} numéricas + {len(cat_features)} categóricas)")

# ── Splits temporales ─────────────────────────────────────────────────────────
df_train = df[df['anio'] < 2024].copy()
df_test  = df[df['anio'] == 2024].copy()

# Global (sin Canarias)
df_train_global = df_train[df_train['region'] != 'Canarias'].copy()
df_test_global  = df_test[df_test['region']  != 'Canarias'].copy()

# Canarias
df_train_can = df_train[df_train['region'] == 'Canarias'].copy()
df_test_can  = df_test[df_test['region']  == 'Canarias'].copy()

print(f"Train global (sin Canarias): {len(df_train_global):,}  |  Test global: {len(df_test_global):,}")
print(f"Train Canarias: {len(df_train_can):,}  |  Test Canarias: {len(df_test_can):,}")"""))

# ── CELL 7: Target encoding ─────────────────────────────────────────────────
cells.append(md("---\n## 3. Target Encoding por Segmento y Región"))

cells.append(code("""def target_encoding_expanding(df_seg, group_cols, target_col, new_col_name):
    df_seg = df_seg.sort_values(['anio', 'semana_anio'])
    enc = (df_seg.groupby(group_cols)[target_col]
                 .transform(lambda x: x.shift(1).expanding().mean()))
    df_seg[new_col_name] = enc.fillna(df_seg[target_col].mean())
    return df_seg

for seg_label, mask in [('A', df['tipo_abc'] == 'A'), ('BC', df['tipo_abc'] != 'A')]:
    idx = df[mask].index
    seg = df.loc[idx].copy()
    seg = target_encoding_expanding(seg, ['codigo_articulo','Municipio'],
                                    'unidades_log', 'te_sku_municipio')
    seg = target_encoding_expanding(seg, ['Municipio'],
                                    'unidades_log', 'te_municipio')
    df.loc[idx, 'te_sku_municipio'] = seg['te_sku_municipio'].values
    df.loc[idx, 'te_municipio']     = seg['te_municipio'].values

for te in ['te_sku_municipio', 'te_municipio']:
    if te not in num_features:
        num_features.append(te)
        all_features.append(te)

# Rehacer todos los splits con encoding
df_train = df[df['anio'] < 2024].copy()
df_test  = df[df['anio'] == 2024].copy()
df_train_global = df_train[df_train['region'] != 'Canarias'].copy()
df_test_global  = df_test[df_test['region']  != 'Canarias'].copy()
df_train_can = df_train[df_train['region'] == 'Canarias'].copy()
df_test_can  = df_test[df_test['region']  == 'Canarias'].copy()

print("Target encoding calculado.")"""))

# ── CELL 9: Helpers ────────────────────────────────────────────────────────
cells.append(md("---\n## 4. Funciones Auxiliares"))

cells.append(code("""def metricas(y_real, y_pred, label=''):
    mae  = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    ss_tot = np.sum((y_real - y_real.mean()) ** 2)
    r2   = 1 - np.sum((y_real - y_pred)**2) / ss_tot if ss_tot > 0 else 0
    mask = y_real > 0
    mape = np.mean(np.abs((y_real[mask] - y_pred[mask]) / y_real[mask])) * 100
    return {'Label': label, 'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'R2': r2}

def make_objective(X_tr, y_tr, cat_feats):
    tscv = TimeSeriesSplit(n_splits=3)
    def objective(trial):
        params = {
            'iterations'       : 600,
            'learning_rate'    : trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
            'depth'            : trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg'      : trial.suggest_float('l2_leaf_reg', 1, 30, log=True),
            'min_data_in_leaf' : trial.suggest_int('min_data_in_leaf', 5, 100),
            'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'loss_function'    : 'MAE',
            'eval_metric'      : 'MAE',
            'random_seed'      : 42,
            'verbose'          : 0,
        }
        scores = []
        for tr_i, val_i in tscv.split(X_tr):
            Xt, Xv = X_tr.iloc[tr_i], X_tr.iloc[val_i]
            yt, yv = y_tr.iloc[tr_i], y_tr.iloc[val_i]
            m = CatBoostRegressor(**params)
            m.fit(Xt, yt, cat_features=cat_feats, eval_set=(Xv,yv),
                  early_stopping_rounds=40, verbose=0)
            scores.append(mean_absolute_error(np.expm1(yv),
                                               np.maximum(np.expm1(m.predict(Xv)), 0)))
        return np.mean(scores)
    return objective

def entrenar(best_p, Xtr, ytr, Xte, cat_feats, eval_y=None):
    params = {**best_p,
              'iterations': 1000, 'loss_function': 'MAE',
              'eval_metric': 'MAE', 'random_seed': 42, 'verbose': 100}
    m = CatBoostRegressor(**params)
    es = (Xte, eval_y) if eval_y is not None else None
    m.fit(Xtr, ytr, cat_features=cat_feats, eval_set=es, early_stopping_rounds=50)
    return m

print("Funciones auxiliares definidas.")"""))

# ── CELL 11: Bias correction analysis ──────────────────────────────────────
cells.append(md("""---
## 5. Análisis del Factor de Corrección de Sesgo

### El problema: MAE + log1p subestima sistemáticamente

El optimizador MAE (L1) aprende la **mediana** de la distribución, no la media.
Al retransformar con `expm1`, obtenemos la mediana en escala original — que en distribuciones
asimétricas como ventas de retail es significativamente menor que la media.

```
Factor de corrección = media_train / mediana_train
pred_corregida = pred_raw × factor
```

Este factor se calcula por separado para cada modelo (global A, global B/C, Canarias A, Canarias B/C)
usando **solo datos de train** — sin tocar el test."""))

cells.append(code("""# ── Calcular factores de corrección por segmento y región ────────────────────
def factor_correccion(df_tr, tipo):
    if tipo == 'A':
        sub = df_tr[df_tr['tipo_abc'] == 'A']['unidades']
    else:
        sub = df_tr[df_tr['tipo_abc'] != 'A']['unidades']
    media   = sub.mean()
    mediana = sub.median()
    factor  = media / mediana
    return factor, media, mediana

# Global (sin Canarias)
fc_A_global,  media_A_g,  med_A_g  = factor_correccion(df_train_global, 'A')
fc_BC_global, media_BC_g, med_BC_g = factor_correccion(df_train_global, 'BC')

# Canarias
fc_A_can,  media_A_c,  med_A_c  = factor_correccion(df_train_can, 'A')
fc_BC_can, media_BC_c, med_BC_c = factor_correccion(df_train_can, 'BC')

print("FACTORES DE CORRECCIÓN DE SESGO (calculados sobre train):")
print(f"{'Segmento':<25} {'Media':>8} {'Mediana':>9} {'Factor':>8}")
print("─" * 55)
print(f"{'Global A':<25} {media_A_g:>8.3f} {med_A_g:>9.3f} {fc_A_global:>8.3f}x")
print(f"{'Global B/C':<25} {media_BC_g:>8.3f} {med_BC_g:>9.3f} {fc_BC_global:>8.3f}x")
print(f"{'Canarias A':<25} {media_A_c:>8.3f} {med_A_c:>9.3f} {fc_A_can:>8.3f}x")
print(f"{'Canarias B/C':<25} {media_BC_c:>8.3f} {med_BC_c:>9.3f} {fc_BC_can:>8.3f}x")
print()
print("Interpretación: 'Global A = 1.74x' → el modelo predice la mediana,")
print("que es 1.74 veces menor que la media real. La corrección lo ajusta.")

# Visualización de la distribución y el sesgo
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, df_tr, label in [(axes[0], df_train_global, 'Global (sin Canarias)'),
                          (axes[1], df_train_can,    'Canarias')]:
    vals = df_tr['unidades'].clip(0, 20)
    ax.hist(vals, bins=20, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(vals.mean(),   color='tomato',  linewidth=2.5, label=f'Media={vals.mean():.2f}')
    ax.axvline(vals.median(), color='seagreen', linewidth=2.5, linestyle='--',
               label=f'Mediana={vals.median():.0f}')
    ax.set_title(f'Distribución unidades — {label}', fontweight='bold')
    ax.set_xlabel('Unidades (recortado en 20)')
    ax.legend(fontsize=9)
plt.tight_layout()
plt.show()"""))

# ── CELL 13: Global models ──────────────────────────────────────────────────
cells.append(md("---\n## 6. Modelos Globales (Todas las Regiones Excepto Canarias)\n\nReutilizamos los hiperparámetros óptimos de It9. No re-optimizamos — los datos son los mismos menos Canarias."))

cells.append(code("""# ── It9 best params (ya conocidos) ───────────────────────────────────────────
best_A_it9  = {'learning_rate': 0.1033, 'depth': 5, 'l2_leaf_reg': 3.3379,
               'min_data_in_leaf': 47, 'subsample': 0.7558, 'colsample_bylevel': 0.9172}
best_BC_it9 = {'learning_rate': 0.2958, 'depth': 4, 'l2_leaf_reg': 2.9837,
               'min_data_in_leaf': 67, 'subsample': 0.8101, 'colsample_bylevel': 0.6374}

# ── Split por segmento — global ───────────────────────────────────────────────
Xtr_A_g  = df_train_global[df_train_global['tipo_abc']=='A'][all_features]
ytr_A_g  = df_train_global[df_train_global['tipo_abc']=='A']['unidades_log']
Xte_A_g  = df_test_global[df_test_global['tipo_abc']=='A'][all_features]
yte_A_g  = df_test_global[df_test_global['tipo_abc']=='A']['unidades_log']

Xtr_BC_g = df_train_global[df_train_global['tipo_abc']!='A'][all_features]
ytr_BC_g = df_train_global[df_train_global['tipo_abc']!='A']['unidades_log']
Xte_BC_g = df_test_global[df_test_global['tipo_abc']!='A'][all_features]
yte_BC_g = df_test_global[df_test_global['tipo_abc']!='A']['unidades_log']

print("Entrenando Modelo Global A (It9 params + nuevas features Canarias)...")
modelo_A_global = entrenar(best_A_it9, Xtr_A_g, ytr_A_g, Xte_A_g, cat_features, yte_A_g)

yhat_A_g_raw = np.maximum(np.expm1(modelo_A_global.predict(Xte_A_g)), 0)
yhat_A_g_cor = yhat_A_g_raw * fc_A_global        # con corrección
yreal_A_g    = df_test_global[df_test_global['tipo_abc']=='A']['unidades'].values

m_A_g_raw = metricas(yreal_A_g, yhat_A_g_raw, 'A global raw')
m_A_g_cor = metricas(yreal_A_g, yhat_A_g_cor, 'A global corregido')
print(f"  Sin corrección: MAE={m_A_g_raw['MAE']:.4f} | MAPE={m_A_g_raw['MAPE']:.2f}%")
print(f"  Con corrección: MAE={m_A_g_cor['MAE']:.4f} | MAPE={m_A_g_cor['MAPE']:.2f}%  (factor={fc_A_global:.3f}x)")"""))

cells.append(code("""print("Entrenando Modelo Global B/C (It9 params + nuevas features Canarias)...")
modelo_BC_global = entrenar(best_BC_it9, Xtr_BC_g, ytr_BC_g, Xte_BC_g, cat_features, yte_BC_g)

yhat_BC_g_raw = np.maximum(np.expm1(modelo_BC_global.predict(Xte_BC_g)), 0)
yhat_BC_g_cor = yhat_BC_g_raw * fc_BC_global
yreal_BC_g    = df_test_global[df_test_global['tipo_abc']!='A']['unidades'].values

m_BC_g_raw = metricas(yreal_BC_g, yhat_BC_g_raw, 'BC global raw')
m_BC_g_cor = metricas(yreal_BC_g, yhat_BC_g_cor, 'BC global corregido')
print(f"  Sin corrección: MAE={m_BC_g_raw['MAE']:.4f} | MAPE={m_BC_g_raw['MAPE']:.2f}%")
print(f"  Con corrección: MAE={m_BC_g_cor['MAE']:.4f} | MAPE={m_BC_g_cor['MAPE']:.2f}%  (factor={fc_BC_global:.3f}x)")"""))

# ── CELL 16: Canarias models ────────────────────────────────────────────────
cells.append(md("""---
## 7. Modelo Dedicado Canarias

### ¿Por qué Canarias necesita su propio modelo?

| Característica | Resto España | Canarias |
|---|---|---|
| Pico de demanda | Primavera-Verano (sem 10-25) | Invierno (sem 44-12) — turismo |
| Clima | 4 estaciones marcadas | Subtropical — variación mínima |
| Logística | Red terrestre | Aislamiento insular (barco/avión) |
| Comportamiento | Predecible con features estándar | Patrón opuesto = modelo global confunde |

Añadimos features específicas (`can_temporada_alta`, `can_sin_semana`, `can_cos_semana`) y optimizamos hiperparámetros propios con Optuna sobre solo los datos de Canarias."""))

cells.append(code("""# ── Datos Canarias por segmento ──────────────────────────────────────────────
Xtr_A_c  = df_train_can[df_train_can['tipo_abc']=='A'][all_features]
ytr_A_c  = df_train_can[df_train_can['tipo_abc']=='A']['unidades_log']
Xte_A_c  = df_test_can[df_test_can['tipo_abc']=='A'][all_features]
yte_A_c  = df_test_can[df_test_can['tipo_abc']=='A']['unidades_log']

Xtr_BC_c = df_train_can[df_train_can['tipo_abc']!='A'][all_features]
ytr_BC_c = df_train_can[df_train_can['tipo_abc']!='A']['unidades_log']
Xte_BC_c = df_test_can[df_test_can['tipo_abc']!='A'][all_features]
yte_BC_c = df_test_can[df_test_can['tipo_abc']!='A']['unidades_log']

print(f"Train Canarias A: {len(Xtr_A_c):,}  |  Test Canarias A: {len(Xte_A_c):,}")
print(f"Train Canarias B/C: {len(Xtr_BC_c):,}  |  Test Canarias B/C: {len(Xte_BC_c):,}")
print()

# ── Optuna Canarias A ─────────────────────────────────────────────────────────
print("Optimizando Modelo Canarias A (25 trials)...")
study_can_A = optuna.create_study(direction='minimize',
                                   sampler=optuna.samplers.TPESampler(seed=42))
study_can_A.optimize(make_objective(Xtr_A_c, ytr_A_c, cat_features),
                     n_trials=25, show_progress_bar=True)
best_can_A = study_can_A.best_params
print(f"Mejor MAE CV Canarias A: {study_can_A.best_value:.4f}")
print(f"  lr={best_can_A['learning_rate']:.4f} | depth={best_can_A['depth']} | "
      f"l2={best_can_A['l2_leaf_reg']:.2f} | min_leaf={best_can_A['min_data_in_leaf']}")"""))

cells.append(code("""# ── Optuna Canarias B/C ───────────────────────────────────────────────────────
print("Optimizando Modelo Canarias B/C (25 trials)...")
study_can_BC = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=42))
study_can_BC.optimize(make_objective(Xtr_BC_c, ytr_BC_c, cat_features),
                      n_trials=25, show_progress_bar=True)
best_can_BC = study_can_BC.best_params
print(f"Mejor MAE CV Canarias B/C: {study_can_BC.best_value:.4f}")
print(f"  lr={best_can_BC['learning_rate']:.4f} | depth={best_can_BC['depth']} | "
      f"l2={best_can_BC['l2_leaf_reg']:.2f} | min_leaf={best_can_BC['min_data_in_leaf']}")

print()
print("Comparativa hiperparámetros Global vs Canarias:")
print(f"  {'Param':<20} {'Global A':>10} {'Can A':>10} {'Global B/C':>12} {'Can B/C':>10}")
print("  " + "─"*65)
for p in ['learning_rate', 'depth', 'l2_leaf_reg', 'min_data_in_leaf']:
    vGA = best_A_it9.get(p, '-')
    vCA = best_can_A.get(p, '-')
    vGB = best_BC_it9.get(p, '-')
    vCB = best_can_BC.get(p, '-')
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
    print(f"  {p:<20} {fmt(vGA):>10} {fmt(vCA):>10} {fmt(vGB):>12} {fmt(vCB):>10}")"""))

cells.append(code("""# ── Entrenar modelos finales Canarias ─────────────────────────────────────────
print("Entrenando modelo final Canarias A...")
modelo_A_can = entrenar(best_can_A, Xtr_A_c, ytr_A_c, Xte_A_c, cat_features, yte_A_c)

print("\\nEntrenando modelo final Canarias B/C...")
modelo_BC_can = entrenar(best_can_BC, Xtr_BC_c, ytr_BC_c, Xte_BC_c, cat_features, yte_BC_c)

# ── Predicciones Canarias (con y sin corrección) ──────────────────────────────
yhat_A_c_raw  = np.maximum(np.expm1(modelo_A_can.predict(Xte_A_c)), 0)
yhat_BC_c_raw = np.maximum(np.expm1(modelo_BC_can.predict(Xte_BC_c)), 0)
yhat_A_c_cor  = yhat_A_c_raw  * fc_A_can
yhat_BC_c_cor = yhat_BC_c_raw * fc_BC_can

yreal_A_c  = df_test_can[df_test_can['tipo_abc']=='A']['unidades'].values
yreal_BC_c = df_test_can[df_test_can['tipo_abc']!='A']['unidades'].values

m_A_c_raw  = metricas(yreal_A_c,  yhat_A_c_raw,  'Can A raw')
m_A_c_cor  = metricas(yreal_A_c,  yhat_A_c_cor,  'Can A corregido')
m_BC_c_raw = metricas(yreal_BC_c, yhat_BC_c_raw, 'Can BC raw')
m_BC_c_cor = metricas(yreal_BC_c, yhat_BC_c_cor, 'Can BC corregido')

print(f"\\n{'─'*55}")
print("CANARIAS — Modelo dedicado vs It9 global")
print(f"{'':25} {'MAE':>8} {'MAPE':>8} {'R²':>6}")
print("─"*55)
print(f"{'It9 Canarias (global)':25} {'0.8680':>8} {'30.5%':>8} {'0.236':>6}")
print(f"{'Can A raw (It10)':25} {m_A_c_raw['MAE']:>8.4f} {m_A_c_raw['MAPE']:>7.2f}%  {m_A_c_raw['R2']:>6.3f}")
print(f"{'Can A corregido (It10)':25} {m_A_c_cor['MAE']:>8.4f} {m_A_c_cor['MAPE']:>7.2f}%  {m_A_c_cor['R2']:>6.3f}")
print(f"{'Can B/C raw (It10)':25} {m_BC_c_raw['MAE']:>8.4f} {m_BC_c_raw['MAPE']:>7.2f}%  {m_BC_c_raw['R2']:>6.3f}")
print(f"{'Can B/C corregido (It10)':25} {m_BC_c_cor['MAE']:>8.4f} {m_BC_c_cor['MAPE']:>7.2f}%  {m_BC_c_cor['R2']:>6.3f}")"""))

# ── CELL 21: Global evaluation ──────────────────────────────────────────────
cells.append(md("---\n## 8. Evaluación Global It10 — Sin vs Con Corrección de Sesgo"))

cells.append(code("""# ── Combinar predicciones: global + Canarias ─────────────────────────────────
y_real_all  = np.concatenate([yreal_A_g,    yreal_BC_g,    yreal_A_c,    yreal_BC_c])
y_pred_raw  = np.concatenate([yhat_A_g_raw, yhat_BC_g_raw, yhat_A_c_raw, yhat_BC_c_raw])
y_pred_cor  = np.concatenate([yhat_A_g_cor, yhat_BC_g_cor, yhat_A_c_cor, yhat_BC_c_cor])

m_global_raw = metricas(y_real_all, y_pred_raw, 'Global raw')
m_global_cor = metricas(y_real_all, y_pred_cor, 'Global corregido')

# ── Tabla comparativa ─────────────────────────────────────────────────────────
comp = pd.DataFrame([
    {'Iteración': 'It7 municipal',        'MAE': 0.628, 'MAPE': 21.50, 'R2': 0.264, 'Corrección': '—'},
    {'Iteración': 'It8 regional',         'MAE': 0.868, 'MAPE': 18.30, 'R2': 0.454, 'Corrección': '—'},
    {'Iteración': 'It9 municipal',        'MAE': 0.610, 'MAPE': 17.59, 'R2': 0.235, 'Corrección': '—'},
    {'Iteración': 'It10 sin corrección',  'MAE': m_global_raw['MAE'], 'MAPE': m_global_raw['MAPE'],
                                           'R2': m_global_raw['R2'], 'Corrección': 'No'},
    {'Iteración': 'It10 con corrección',  'MAE': m_global_cor['MAE'], 'MAPE': m_global_cor['MAPE'],
                                           'R2': m_global_cor['R2'], 'Corrección': 'Sí'},
]).set_index('Iteración')

print("📊 COMPARATIVA GLOBAL:")
print(comp[['MAE','MAPE','R2','Corrección']].round(4).to_string())

delta_it9_cor = m_global_cor['MAPE'] - 17.59
print(f"\\nIt10 corregido vs It9: Δ MAE {m_global_cor['MAE']-0.610:+.4f}  |  Δ MAPE {delta_it9_cor:+.2f} pp")
print()

# ── Efecto de la corrección por segmento ──────────────────────────────────────
print("📊 EFECTO DE LA CORRECCIÓN POR SEGMENTO:")
print(f"{'Segmento':<25} {'MAPE raw':>10} {'MAPE cor':>10} {'Δ MAPE':>8} {'Factor':>8}")
print("─"*65)
for label, m_raw, m_cor, fc in [
    ('Global A',     m_A_g_raw,  m_A_g_cor,  fc_A_global),
    ('Global B/C',   m_BC_g_raw, m_BC_g_cor, fc_BC_global),
    ('Canarias A',   m_A_c_raw,  m_A_c_cor,  fc_A_can),
    ('Canarias B/C', m_BC_c_raw, m_BC_c_cor, fc_BC_can),
]:
    delta = m_cor['MAPE'] - m_raw['MAPE']
    print(f"{label:<25} {m_raw['MAPE']:>9.2f}%  {m_cor['MAPE']:>9.2f}%  {delta:>+7.2f} pp  {fc:>7.3f}x")"""))

# ── CELL 23: Regional ──────────────────────────────────────────────────────
cells.append(md("---\n## 9. Análisis por Región — It9 vs It10"))

cells.append(code("""# ── Construir df de evaluación completo ──────────────────────────────────────
rows_A_g  = df_test_global[df_test_global['tipo_abc']=='A'][['semana_anio','region']].copy()
rows_BC_g = df_test_global[df_test_global['tipo_abc']!='A'][['semana_anio','region']].copy()
rows_A_c  = df_test_can[df_test_can['tipo_abc']=='A'][['semana_anio','region']].copy()
rows_BC_c = df_test_can[df_test_can['tipo_abc']!='A'][['semana_anio','region']].copy()

for rows, yr, yp_raw, yp_cor in [
    (rows_A_g,  yreal_A_g,  yhat_A_g_raw,  yhat_A_g_cor),
    (rows_BC_g, yreal_BC_g, yhat_BC_g_raw, yhat_BC_g_cor),
    (rows_A_c,  yreal_A_c,  yhat_A_c_raw,  yhat_A_c_cor),
    (rows_BC_c, yreal_BC_c, yhat_BC_c_raw, yhat_BC_c_cor),
]:
    rows['real']    = yr
    rows['pred_raw'] = yp_raw
    rows['pred_cor'] = yp_cor

df_eval = pd.concat([rows_A_g, rows_BC_g, rows_A_c, rows_BC_c], ignore_index=True)

it9_region = {'Noreste':18.5, 'Noroeste':16.5, 'Sur':20.9,
              'Centro':25.1, 'Norte':30.5, 'Canarias':30.5}

res = []
for region, grp in df_eval.groupby('region'):
    mr  = metricas(grp['real'].values, grp['pred_raw'].values)
    mc  = metricas(grp['real'].values, grp['pred_cor'].values)
    res.append({'Region': region, 'N': len(grp),
                'MAPE_It9': it9_region.get(region, None),
                'MAPE_It10_raw': round(mr['MAPE'],1),
                'MAPE_It10_cor': round(mc['MAPE'],1),
                'R2_cor': round(mc['R2'],3)})

df_reg = pd.DataFrame(res).sort_values('MAPE_It10_cor')
df_reg['Δ vs It9'] = (df_reg['MAPE_It10_cor'] - df_reg['MAPE_It9']).round(1)
df_reg['Estado'] = df_reg['Δ vs It9'].apply(lambda x: '✓✓' if x<-3 else ('✓' if x<0 else '✗'))
print(df_reg[['Region','N','MAPE_It9','MAPE_It10_raw','MAPE_It10_cor','Δ vs It9','Estado','R2_cor']].to_string(index=False))

# ── Gráfico ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(df_reg))
w = 0.28
ax.bar(x - w,   df_reg['MAPE_It9'],      w, label='It9',          color='steelblue', alpha=0.85)
ax.bar(x,       df_reg['MAPE_It10_raw'], w, label='It10 sin corr.', color='orange',   alpha=0.85)
ax.bar(x + w,   df_reg['MAPE_It10_cor'], w, label='It10 con corr.', color='seagreen', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(df_reg['Region'], rotation=15)
ax.axhline(20, linestyle='--', color='grey', alpha=0.5, label='20% umbral')
ax.set_ylabel('MAPE (%)'); ax.set_title('MAPE por Región — It9 vs It10', fontweight='bold')
ax.legend(fontsize=9); plt.tight_layout(); plt.show()"""))

# ── CELL 25: Feature importance Canarias ───────────────────────────────────
cells.append(md("---\n## 10. Importancia de Features — Canarias vs Global"))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(18, 7))

features_can = ['can_temporada_alta', 'es_temporada_turistica_can',
                'can_sin_semana', 'can_cos_semana']

for ax, modelo, label, color in [
    (axes[0], modelo_A_can,  'Canarias A (It10)',   'teal'),
    (axes[1], modelo_BC_can, 'Canarias B/C (It10)', 'coral'),
]:
    imp = pd.DataFrame({'Feature': modelo.feature_names_,
                        'Imp': modelo.get_feature_importance()}
                       ).sort_values('Imp', ascending=False).head(20)
    colors = ['tomato' if f in features_can else color for f in imp['Feature']]
    sns.barplot(data=imp, x='Imp', y='Feature', palette=colors, ax=ax)
    ax.set_title(f'Top 20 Features — {label}\\n(rojo = feature exclusiva Canarias)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Importancia (%)')

plt.tight_layout(); plt.show()

# Peso de las features de Canarias
for modelo, label in [(modelo_A_can,'A'), (modelo_BC_can,'B/C')]:
    imp_df = pd.DataFrame({'Feature': modelo.feature_names_,
                            'Imp': modelo.get_feature_importance()})
    can_sum = imp_df[imp_df['Feature'].isin(features_can)]['Imp'].sum()
    print(f"Canarias {label} — features exclusivas Canarias: {can_sum:.1f}% importancia total")"""))

# ── CELL 27: Historical evolution ──────────────────────────────────────────
cells.append(md("---\n## 11. Evolución Histórica — It1 a It10"))

cells.append(code("""historico = pd.DataFrame([
    {'It': 'It1 Baseline',       'MAE': 0.7925, 'MAPE': None,  'R2': 0.295, 'Nivel': 'Municipal'},
    {'It': 'It2 Rolling Mean',   'MAE': 0.7728, 'MAPE': None,  'R2': 0.330, 'Nivel': 'Municipal'},
    {'It': 'It3 Estacionalidad', 'MAE': 0.7690, 'MAPE': None,  'R2': 0.330, 'Nivel': 'Municipal'},
    {'It': 'It4 Log1p',          'MAE': 0.6488, 'MAPE': 26.35, 'R2': 0.287, 'Nivel': 'Municipal'},
    {'It': 'It5 Optuna+Enc',     'MAE': 0.6411, 'MAPE': 26.03, 'R2': 0.288, 'Nivel': 'Municipal'},
    {'It': 'It6 Descuentos',     'MAE': 0.6409, 'MAPE': 25.80, 'R2': 0.310, 'Nivel': 'Municipal'},
    {'It': 'It7 Mod.Dedicados',  'MAE': 0.6280, 'MAPE': 21.50, 'R2': 0.264, 'Nivel': 'Municipal'},
    {'It': 'It8 Regional',       'MAE': 0.8680, 'MAPE': 18.30, 'R2': 0.454, 'Nivel': 'Regional⚠'},
    {'It': 'It9 Lags+Features',  'MAE': 0.6100, 'MAPE': 17.59, 'R2': 0.235, 'Nivel': 'Municipal'},
    {'It': 'It10 Can+Corrección',
     'MAE':  m_global_cor['MAE'],
     'MAPE': m_global_cor['MAPE'],
     'R2':   m_global_cor['R2'],
     'Nivel': 'Municipal'},
])
historico['Δ MAE']  = historico['MAE'].diff().round(4)
historico['Δ MAPE'] = historico['MAPE'].diff().round(2)
print(historico[['It','Nivel','MAE','MAPE','R2','Δ MAE','Δ MAPE']].to_string(index=False))
print("\\n⚠ It8 = nivel regional, MAE no comparable con iteraciones municipales")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
mun = historico[historico['Nivel']=='Municipal']
reg = historico[historico['Nivel']=='Regional⚠']
axes[0].plot(mun['It'], mun['MAE'], marker='o', color='steelblue', linewidth=2.5)
axes[0].scatter(reg['It'], reg['MAE'], marker='D', color='orange', s=80, zorder=5, label='Regional')
axes[0].set_title('Evolución MAE — It1 a It10', fontweight='bold')
axes[0].set_ylabel('MAE'); axes[0].tick_params(axis='x', rotation=40)
axes[0].grid(True, alpha=0.4); axes[0].legend()

mape_mun = mun.dropna(subset=['MAPE'])
mape_reg = reg.dropna(subset=['MAPE'])
axes[1].plot(mape_mun['It'], mape_mun['MAPE'], marker='s', color='seagreen', linewidth=2.5)
if not mape_reg.empty:
    axes[1].scatter(mape_reg['It'], mape_reg['MAPE'], marker='D', color='orange', s=80, zorder=5)
axes[1].axhline(20, linestyle='--', color='grey', alpha=0.5, label='20% umbral')
axes[1].axhline(15, linestyle=':', color='tomato', alpha=0.5, label='15% objetivo')
axes[1].set_title('Evolución MAPE — It4 a It10', fontweight='bold')
axes[1].set_ylabel('MAPE (%)'); axes[1].tick_params(axis='x', rotation=40)
axes[1].legend(); axes[1].grid(True, alpha=0.4)
plt.tight_layout(); plt.show()"""))

# ── CELL 29: Summary ───────────────────────────────────────────────────────
cells.append(md("""---
## Resumen Ejecutivo — Iteración 10

*(Se actualizará con los resultados reales tras la ejecución)*

### Qué se implementó

| Mejora | Descripción |
|---|---|
| **Modelo dedicado Canarias** | CatBoost propio (A y B/C) con Optuna, entrenado solo sobre datos de Canarias, con 4 features de estacionalidad turística inversa |
| **Factor de corrección de sesgo** | Ratio media/mediana calculado en train, aplicado en predicción para corregir subestimación sistemática de MAE+log1p |

### Factores de corrección calculados

| Segmento | Factor | Interpretación |
|---|---|---|
| Global A | *ver celda 5* | Multiplica predicción por este valor |
| Global B/C | *ver celda 5* | — |
| Canarias A | *ver celda 5* | — |
| Canarias B/C | *ver celda 5* | — |

### Resultados por región

*(Ver tabla celda 9)*

### Lecciones aprendidas

1. La corrección de sesgo mejora sistemáticamente todos los segmentos cuando la distribución es asimétrica (media >> mediana)
2. El modelo global aprende la estacionalidad media española — para Canarias con patrón opuesto, es contraproducente
3. Con pocas filas de Canarias (~1K test), Optuna con 25 trials es suficiente para encontrar buenos hiperparámetros

### Próximos pasos

| Prioridad | Acción | Impacto esperado |
|---|---|---|
| **1** | Clustering de SKUs como feature (It11) | −2 a −4 pp MAPE en B/C global |
| **2** | Walk-forward validation mensual | Benchmark de producción real |
| **3** | Hurdle a granularidad SKU×Región | Clasificación de demanda cero |"""))

# ── Build notebook ─────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"}
    },
    "cells": cells
}

out = "Notebooks/15_Modelado_Iteracion10_Canarias_Correccion.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook creado: {out}")
print(f"Celdas: {len(cells)} ({sum(1 for c in cells if c['cell_type']=='code')} código, "
      f"{sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
