"""
Pipeline de forecasting CRUZBER — punto de entrada principal.

Uso:
    python -m cruzber_forecast.main
    python cruzber_forecast/main.py
"""
import time
import yaml
import pandas as pd
from pathlib import Path

from src.data_loader import load_all_sources, filter_b2b
from src.dense_panel import build_dense_panel
from src.features import add_all_features
from src.classification import classify_syntetos_boylan, subsegment_lumpy
from src.model_smooth import get_feature_lists, train_smooth
from src.model_hurdle import train_hurdle
from src.baseline import compute_baselines, compute_croston_sba
from src.hybrid_strategy import apply_hybrid_strategy
from src.evaluation import evaluate_global, error_analysis, walk_forward, overfitting_check
from src.export import export_xlsx


def load_config(path: str = 'config.yaml') -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    t0 = time.time()
    cfg = load_config()
    test_year = cfg['forecast']['test_year']

    # ── 1. Carga de datos ────────────────────────────────────────────────
    print('\n[1] Cargando fuentes de datos...')
    df_raw, df_art, df_art_full, df_clima_nac, df_cicl_agg = load_all_sources(cfg)
    df_nac = filter_b2b(df_raw)

    # ── 2. Dense panel ───────────────────────────────────────────────────
    print('\n[2] Construyendo Dense Panel...')
    df_agg, semanas_unicas = build_dense_panel(df_nac, df_clima_nac, df_cicl_agg, cfg)

    # ── 3. Feature engineering ───────────────────────────────────────────
    print('\n[3] Generando features...')
    df_agg = add_all_features(df_agg, df_art_full, cfg)

    # ── 4. Clasificación Syntetos-Boylan ─────────────────────────────────
    print('\n[4] Clasificando demanda (Syntetos-Boylan)...')
    df_agg, demand_stats = classify_syntetos_boylan(df_agg, cfg)
    df_agg = subsegment_lumpy(df_agg, demand_stats)

    # ── 5. Preparación de segmentos ──────────────────────────────────────
    print('\n[5] Preparando segmentos...')
    feats_num, feats_cat = get_feature_lists(df_agg)
    feats_all = feats_num + feats_cat

    df_reg = df_agg[df_agg['sb_class'].isin(['Smooth', 'Erratic'])].copy()
    df_hrd = df_agg[
        df_agg['sb_class'].isin(['Intermittent', 'Lumpy'])
        & (df_agg['sb_reliability'] != 'zero_only_or_newborn')
    ].copy()
    df_newborn = df_agg[
        df_agg['sb_class'].isin(['Intermittent', 'Lumpy'])
        & (df_agg['sb_reliability'] == 'zero_only_or_newborn')
    ].copy()

    print(f'    Smooth/Erratic: {df_reg["codigo_articulo"].nunique()} SKUs')
    print(f'    Hurdle: {df_hrd["codigo_articulo"].nunique()} SKUs')
    print(f'    Newborns (pred=0): {df_newborn["codigo_articulo"].nunique()} SKUs')

    # ── 6. Baselines ─────────────────────────────────────────────────────
    print('\n[6] Calculando baselines...')
    df_agg = compute_baselines(df_agg, cfg)
    df_agg = compute_croston_sba(df_agg, cfg)

    # ── 7. Modelo Smooth / Erratic ───────────────────────────────────────
    print('\n[7] Entrenando modelo Smooth/Erratic...')
    smooth_out = train_smooth(df_reg, feats_all, feats_cat, cfg)
    test_R = smooth_out['test_df']

    # ── 8. Modelo Hurdle ─────────────────────────────────────────────────
    print('\n[8] Entrenando modelo Hurdle (Intermittent/Lumpy)...')
    hurdle_out = train_hurdle(df_hrd, feats_all, feats_cat, cfg)
    test_H = hurdle_out['test_df']

    # ── 9. Evaluación global ─────────────────────────────────────────────
    print('\n[9] Evaluación global...')
    df_eval = evaluate_global(test_R, test_H)
    _ = error_analysis(df_eval)
    _ = walk_forward(df_eval)
    _ = overfitting_check(
        smooth_out['model'], hurdle_out['model_clf'], hurdle_out['model_reg'],
        smooth_out['folds'][-1][0], smooth_out['folds'][-1][1],
        smooth_out['pred'], smooth_out['y_te'],
        hurdle_out['folds_full'][-1][0], hurdle_out['folds_full'][-1][1],
        hurdle_out['pred'], hurdle_out['y_te'],
        smooth_out['cat_idx'], hurdle_out['cat_idx_clean'],
    )

    # ── 10. Construcción de df_final ─────────────────────────────────────
    print('\n[10] Ensamblando predicciones...')
    test_newborn = df_newborn[df_newborn['anio'] == test_year].copy()
    test_newborn['pred'] = 0.0
    test_newborn['pred_p10'] = 0.0
    test_newborn['pred_p90'] = 0.0
    test_newborn['real'] = test_newborn['target_12w_ahead']
    test_newborn['bias'] = -test_newborn['real']

    df_final = pd.concat([test_R, test_H, test_newborn], ignore_index=True)

    # Traer predicciones alternativas desde df_agg
    alt_cols = ['codigo_articulo', 'semana_anio', 'baseline_naive', 'ma4_pred',
                'croston_pred', 'sba_pred']
    alt_preds = df_agg[df_agg['anio'] == test_year][alt_cols].copy()
    df_final = df_final.merge(alt_preds, on=['codigo_articulo', 'semana_anio'], how='left')
    for col in ['baseline_naive', 'ma4_pred', 'croston_pred', 'sba_pred']:
        df_final[col] = df_final[col].fillna(0).clip(lower=0)
    df_final['lumpy_subtype'] = df_final['lumpy_subtype'].fillna('n/a')

    # ── 11. Estrategia híbrida ───────────────────────────────────────────
    print('\n[11] Aplicando estrategia híbrida...')
    df_final = apply_hybrid_strategy(df_final)

    # ── 12. Exportación ──────────────────────────────────────────────────
    print('\n[12] Exportando...')
    export_xlsx(df_final, df_nac, df_clima_nac, df_art, cfg)

    elapsed = time.time() - t0
    print(f'\n  Pipeline completado en {elapsed/60:.1f} min')


if __name__ == '__main__':
    main()
