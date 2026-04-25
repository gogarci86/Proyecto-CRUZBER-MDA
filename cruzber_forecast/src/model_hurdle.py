"""
Modelo Hurdle para segmentos Intermittent / Lumpy.
Fase 1: CatBoostClassifier (probabilidad de venta).
Fase 2: CatBoostRegressor Quantile(0.65) (volumen condicional).
Predicción suavizada: pred = prob × vol.
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from .model_smooth import generar_folds_tss, print_metrics

TARGET = 'target_12w_ahead'


def _feature_selection(df_hrd: pd.DataFrame, feats_all: list, feats_cat: list,
                        cfg: dict) -> tuple:
    """
    Selección rápida de features por importancia (CatBoost 200 iter).
    Devuelve (feats_keep, FEATS_CAT_H, cat_idx_clean).
    """
    sample_pct = cfg['hurdle']['sample_pct']
    fs_min = cfg['hurdle']['fs_importance_min']
    seed = cfg['forecast']['seed']

    np.random.seed(seed)
    unique_skus = df_hrd['codigo_articulo'].unique()
    sample_skus = np.random.choice(
        unique_skus,
        size=int(len(unique_skus) * sample_pct),
        replace=False,
    )
    df_sample = df_hrd[df_hrd['codigo_articulo'].isin(sample_skus)].copy()

    folds_sample = generar_folds_tss(df_sample, feats_all, feats_cat)
    cat_idx = [feats_all.index(c) for c in feats_cat if c in feats_all]

    X_tr_fs, y_tr_fs, _, _, _, _ = folds_sample[-1]
    fs_model = CatBoostRegressor(
        iterations=200, depth=6, learning_rate=0.05, verbose=0, random_seed=seed
    )
    fs_model.fit(Pool(X_tr_fs, y_tr_fs, cat_features=cat_idx))

    imp = fs_model.get_feature_importance()
    imp_df = pd.DataFrame({'feature': feats_all, 'importance': imp})
    feats_keep = imp_df[imp_df['importance'] > fs_min]['feature'].tolist()
    for c in feats_cat:
        if c not in feats_keep:
            feats_keep.append(c)

    feats_cat_h = [c for c in feats_cat if c in feats_keep]
    cat_idx_clean = [feats_keep.index(c) for c in feats_cat_h]

    print(f'    Feature selection: {len(feats_all)} → {len(feats_keep)} features')
    return feats_keep, feats_cat_h, cat_idx_clean


def train_hurdle(df_hrd: pd.DataFrame, feats_all: list, feats_cat: list,
                 cfg: dict) -> dict:
    """
    Entrena el modelo Hurdle completo y devuelve artefactos necesarios.

    Parameters
    ----------
    df_hrd : Subconjunto Intermittent + Lumpy (excluye zero_only_or_newborn).
    feats_all : Lista completa de features numéricas + categóricas.
    feats_cat : Features categóricas.
    cfg : Configuración del pipeline.

    Returns dict con modelos, predicciones, features seleccionadas y folds.
    """
    p = cfg['frozen_params']
    min_horizon = cfg['forecast']['min_horizon']
    eval_buffer = cfg['forecast']['eval_buffer']
    seed = cfg['forecast']['seed']

    feats_keep, feats_cat_h, cat_idx_clean = _feature_selection(
        df_hrd, feats_all, feats_cat, cfg
    )

    folds_full = generar_folds_tss(df_hrd, feats_keep, feats_cat_h)

    clf_params = {
        'iterations': p['hurdle_classifier']['iterations'],
        'learning_rate': p['hurdle_classifier']['learning_rate'],
        'depth': p['hurdle_classifier']['depth'],
        'l2_leaf_reg': p['hurdle_classifier']['l2_leaf_reg'],
        'loss_function': 'Logloss',
        'early_stopping_rounds': p['hurdle_classifier']['early_stopping_rounds'],
        'random_seed': seed,
        'verbose': 0,
    }
    reg_params = {
        'iterations': p['hurdle_regressor']['iterations'],
        'learning_rate': p['hurdle_regressor']['learning_rate'],
        'depth': p['hurdle_regressor']['depth'],
        'l2_leaf_reg': p['hurdle_regressor']['l2_leaf_reg'],
        'loss_function': f'Quantile:alpha={p["hurdle_regressor"]["quantile_alpha"]}',
        'early_stopping_rounds': p['hurdle_regressor']['early_stopping_rounds'],
        'random_seed': seed,
        'verbose': 0,
    }

    X_tr, y_tr, X_te, y_te, train_df, test_df = folds_full[-1]

    sem_abs = X_tr['anio'] * 53 + X_tr['semana_anio']
    max_s = sem_abs.max()
    eval_mask = sem_abs > (max_s - eval_buffer)
    train_mask = sem_abs <= (max_s - eval_buffer - min_horizon)

    y_tr_bin = (y_tr > 0).astype(int)

    pool_tr_clf = Pool(X_tr[train_mask], y_tr_bin[train_mask], cat_features=cat_idx_clean)
    pool_val_clf = Pool(X_tr[eval_mask], y_tr_bin[eval_mask], cat_features=cat_idx_clean)
    pool_te = Pool(X_te, cat_features=cat_idx_clean)

    model_clf = CatBoostClassifier(**clf_params)
    model_clf.fit(pool_tr_clf, eval_set=pool_val_clf, use_best_model=True)
    prob_te = model_clf.predict_proba(pool_te)[:, 1]

    pool_tr_reg = Pool(
        X_tr[train_mask], np.log1p(y_tr[train_mask].clip(0)), cat_features=cat_idx_clean
    )
    pool_val_reg = Pool(
        X_tr[eval_mask], np.log1p(y_tr[eval_mask].clip(0)), cat_features=cat_idx_clean
    )

    model_reg = CatBoostRegressor(**reg_params)
    model_reg.fit(pool_tr_reg, eval_set=pool_val_reg, use_best_model=True)
    vol_te = np.expm1(model_reg.predict(pool_te)).clip(0)

    pred = (prob_te * vol_te).clip(0)

    test_df = test_df.copy()
    test_df['pred'] = pred
    test_df['real'] = y_te.values
    test_df['pred_p10'] = pred * 0.7
    test_df['pred_p90'] = pred * 1.4
    test_df['bias'] = pred - y_te.values

    print_metrics('Hurdle (Int./Lumpy)', y_te.values, pred)

    return {
        'model_clf': model_clf,
        'model_reg': model_reg,
        'pred': pred,
        'y_te': y_te.values,
        'test_df': test_df,
        'X_te': X_te,
        'feats_keep': feats_keep,
        'feats_cat_h': feats_cat_h,
        'cat_idx_clean': cat_idx_clean,
        'folds_full': folds_full,
    }
