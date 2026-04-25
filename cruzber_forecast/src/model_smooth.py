"""
Entrenamiento del modelo CatBoost Tweedie para segmentos Smooth / Erratic.
Incluye la generación de folds temporales y métricas de evaluación.
"""
import unicodedata
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

TARGET = 'target_12w_ahead'

FEATS_NUM = [
    'semana_anio', 'anio', 'mes', 'trimestre', 'semana_del_mes', 'es_fin_mes',
    'sem_sin', 'sem_cos', 'temporada_alta', 'dias_laborables_semana',
    'lag_1w', 'lag_5w', 'lag_9w', 'lag_52w',
    'roll_4w', 'roll_8w', 'roll_12w', 'roll_std_8w', 'roll_std_12w',
    'ewm_4w', 'ewm_8w', 'ewm_12w', 'tendencia_4v4', 'ratio_yoy',
    'por_descuento2', 'precio_unit', 'prevision_semanal', 'factor_crecimiento', 'tarifa_nacional',
    'temp_media', 'precip_mm', 'viento_max',
    'num_pruebas_cicl', 'dias_pruebas_cicl', 'hubo_prueba_cicl',
    'tsls', 'sale_freq_12w',
    'producto_edad_semanas', 'lifecycle_ratio',
    'stockout_freq_12w',
    'te_codigo_articulo', 'te_cr_gama', 'te_area_comp',
]
FEATS_CAT = ['CR_GamaProducto', 'CR_TipoProducto', 'CR_MaterialAgrupacion', 'AreaCompetenciaLc']


def get_feature_lists(df: pd.DataFrame) -> tuple:
    """Filtra FEATS_NUM y FEATS_CAT a columnas realmente presentes."""
    num = [c for c in FEATS_NUM if c in df.columns]
    cat = [c for c in FEATS_CAT if c in df.columns]
    return num, cat


def _normalize_cat_col(series: pd.Series) -> pd.Series:
    return series.astype(str).fillna('NaN').apply(
        lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii')
    )


def generar_folds_tss(df: pd.DataFrame, feats_all: list, feats_cat: list,
                      target: str = TARGET) -> list:
    """
    Genera folds temporales expansivos (2022, 2023, 2024).

    Returns list of (X_tr, y_tr, X_te, y_te, train_df, test_df) per fold.
    Aplica target encoding dinámico dentro de cada fold para evitar leakage.
    """
    df = df.copy()
    te_mappings = {
        'codigo_articulo': 'te_codigo_articulo',
        'CR_GamaProducto': 'te_cr_gama',
        'AreaCompetenciaLc': 'te_area_comp',
    }

    for c in feats_cat:
        if c in df.columns:
            df[c] = _normalize_cat_col(df[c])

    df['period_id'] = df['anio'].astype(str) + '_' + df['semana_anio'].astype(str).str.zfill(2)
    temporales = (
        df[['period_id', 'anio', 'semana_anio']]
        .drop_duplicates()
        .sort_values(['anio', 'semana_anio'])
        .reset_index(drop=True)
    )

    folds = []
    for eval_yr in [2022, 2023, 2024]:
        tr_p = temporales[temporales['anio'] < eval_yr]
        te_p = temporales[temporales['anio'] == eval_yr]

        train_df = df[df['period_id'].isin(tr_p['period_id'])].copy()
        test_df = df[df['period_id'].isin(te_p['period_id'])].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        global_mean = train_df[target].mean()
        for src_col, te_col in te_mappings.items():
            if src_col not in train_df.columns:
                continue
            stats = train_df.groupby(src_col)[target].agg(['mean', 'count']).reset_index()
            smooth = 30
            stats['te'] = (
                (stats['mean'] * stats['count'] + global_mean * smooth)
                / (stats['count'] + smooth)
            )
            mapping = stats.set_index(src_col)['te']
            train_df[te_col] = train_df[src_col].map(mapping).fillna(global_mean)
            test_df[te_col] = test_df[src_col].map(mapping).fillna(global_mean)

        feats_present = [f for f in feats_all if f in train_df.columns]
        X_tr = train_df[feats_present].copy()
        y_tr = train_df[target].copy()
        X_te = test_df[feats_present].copy()
        y_te = test_df[target].copy()

        folds.append((X_tr, y_tr, X_te, y_te, train_df, test_df))

    return folds


def print_metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray,
                  df_test: pd.DataFrame = None) -> None:
    mae = np.mean(np.abs(y_true - y_pred))
    wmape = np.sum(np.abs(y_true - y_pred)) / max(np.sum(np.abs(y_true)), 1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    bias = np.mean(y_pred - y_true)
    print(f'  [{label}]  MAE={mae:.2f}  WMAPE={wmape*100:.1f}%  R²={r2:.3f}  Bias={bias:+.2f}')


def train_smooth(df_reg: pd.DataFrame, feats_all: list, feats_cat: list,
                 cfg: dict) -> dict:
    """
    Entrena CatBoost Tweedie sobre el segmento Smooth / Erratic.

    Returns dict con model, predictions, intervals, folds.
    """
    p = cfg['frozen_params']['smooth']
    min_horizon = cfg['forecast']['min_horizon']
    eval_buffer = cfg['forecast']['eval_buffer']
    seed = cfg['forecast']['seed']

    folds = generar_folds_tss(df_reg, feats_all, feats_cat)
    cat_idx = [feats_all.index(c) for c in feats_cat if c in feats_all]

    params = {
        'learning_rate': p['learning_rate'],
        'depth': p['depth'],
        'l2_leaf_reg': p['l2_leaf_reg'],
        'loss_function': f'Tweedie:variance_power={p["tweedie_vp"]}',
        'iterations': p['iterations'],
        'early_stopping_rounds': p['early_stopping_rounds'],
        'random_seed': seed,
        'verbose': 0,
    }

    X_tr, y_tr, X_te, y_te, train_df, test_df = folds[-1]

    sem_abs = X_tr['anio'] * 53 + X_tr['semana_anio']
    max_s = sem_abs.max()
    eval_mask = sem_abs > (max_s - eval_buffer)
    train_mask = sem_abs <= (max_s - eval_buffer - min_horizon)

    pool_tr = Pool(X_tr[train_mask], np.log1p(y_tr[train_mask].clip(0)), cat_features=cat_idx)
    pool_val = Pool(X_tr[eval_mask], np.log1p(y_tr[eval_mask].clip(0)), cat_features=cat_idx)
    pool_te = Pool(X_te, cat_features=cat_idx)

    model = CatBoostRegressor(**params)
    model.fit(pool_tr, eval_set=pool_val, use_best_model=True)
    pred = np.expm1(model.predict(pool_te)).clip(0)

    preds_val = np.expm1(model.predict(pool_val)).clip(0)
    residuals = y_tr[eval_mask].values - preds_val
    q_lo = np.quantile(residuals, 0.10)
    q_hi = np.quantile(residuals, 0.90)

    test_df = test_df.copy()
    test_df['pred'] = pred
    test_df['real'] = y_te.values
    test_df['pred_p10'] = (pred + q_lo).clip(0)
    test_df['pred_p90'] = (pred + q_hi).clip(0)
    test_df['bias'] = pred - y_te.values

    print_metrics('Smooth/Erratic', y_te.values, pred)

    return {
        'model': model,
        'pred': pred,
        'y_te': y_te.values,
        'test_df': test_df,
        'X_te': X_te,
        'cat_idx': cat_idx,
        'folds': folds,
        'feats_all': feats_all,
        'feats_cat': feats_cat,
    }
