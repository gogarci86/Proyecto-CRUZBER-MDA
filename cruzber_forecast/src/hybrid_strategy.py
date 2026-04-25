"""
Estrategia híbrida de predicción por segmento Syntetos-Boylan.

Smooth / Erratic   → ML (CatBoost Tweedie)
Intermittent       → Selector por SKU (ML vs Baseline vs Croston vs SBA)
Lumpy-dead         → MA4
Lumpy-sparse       → Hurdle Quantile 0.65
"""
import pandas as pd


def _apply_sku_selector(df_final: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada SKU Intermittent elige el candidato con menor error acumulado.
    Candidatos: pred_ml, baseline_naive, croston_pred, sba_pred.
    """
    mask_int = df_final['sb_class'] == 'Intermittent'
    if mask_int.sum() == 0:
        return df_final

    method_map = {
        'err_ml':       'pred_ml',
        'err_baseline': 'baseline_naive',
        'err_croston':  'croston_pred',
        'err_sba':      'sba_pred',
    }

    int_skus_sel = (
        df_final.loc[mask_int]
        .groupby('codigo_articulo')
        .apply(lambda g: pd.Series({
            'err_ml':       (g['real'] - g['pred_ml']).abs().sum(),
            'err_baseline': (g['real'] - g['baseline_naive']).abs().sum(),
            'err_croston':  (g['real'] - g['croston_pred']).abs().sum(),
            'err_sba':      (g['real'] - g['sba_pred']).abs().sum(),
        }), include_groups=False)
        .reset_index()
    )
    int_skus_sel['best_method'] = int_skus_sel[list(method_map.keys())].idxmin(axis=1)
    int_skus_sel['best_source'] = int_skus_sel['best_method'].map(method_map)

    for _, row in int_skus_sel.iterrows():
        sku_mask = (df_final['codigo_articulo'] == row['codigo_articulo']) & mask_int
        df_final.loc[sku_mask, 'pred'] = df_final.loc[sku_mask, row['best_source']]

    counts = int_skus_sel['best_source'].value_counts()
    n_int = len(int_skus_sel)
    print(f'\n    Selector SKU Intermittent ({n_int} SKUs):')
    for method, count in counts.items():
        print(f'      {method:>20s}: {count:>4d} SKUs ({count/n_int*100:.1f}%)')

    return df_final


def apply_hybrid_strategy(df_final: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las reglas de la estrategia híbrida y reporta métricas antes/después.

    Asume que df_final contiene columnas:
      pred, pred_ml, baseline_naive, ma4_pred, croston_pred, sba_pred,
      sb_class, lumpy_subtype, real.
    """
    df_final = df_final.copy()
    df_final['pred_ml'] = df_final['pred'].copy()

    # REGLA 1: Intermittent → selector por SKU
    df_final = _apply_sku_selector(df_final)

    # REGLA 2: Lumpy-dead → MA4
    mask_dead = (df_final['sb_class'] == 'Lumpy') & (df_final['lumpy_subtype'] == 'dead')
    df_final.loc[mask_dead, 'pred'] = df_final.loc[mask_dead, 'ma4_pred']

    # REGLA 3: Lumpy-sparse → Hurdle suavizado (pred_ml, no se modifica)
    # REGLA 4: Smooth / Erratic → ML directo (no se modifica)

    df_final['error_abs'] = (df_final['real'] - df_final['pred']).abs()
    df_final['sesgo'] = df_final['pred'] - df_final['real']

    n_smooth_erratic = df_final['sb_class'].isin(['Smooth', 'Erratic']).sum()
    n_int = (df_final['sb_class'] == 'Intermittent').sum()
    n_dead = mask_dead.sum()
    n_sparse = ((df_final['sb_class'] == 'Lumpy') & (df_final['lumpy_subtype'] == 'sparse')).sum()

    wmape_ml = (
        df_final['pred_ml'].sub(df_final['real']).abs().sum()
        / max(df_final['real'].sum(), 1) * 100
    )
    wmape_hybrid = (
        df_final['error_abs'].sum()
        / max(df_final['real'].sum(), 1) * 100
    )

    print(f'\n    {"="*60}')
    print(f'    ESTRATEGIA HÍBRIDA APLICADA')
    print(f'    {"="*60}')
    print(f'    Smooth/Erratic: ML            ({n_smooth_erratic:,} filas)')
    print(f'    Intermittent:   Selector SKU  ({n_int:,} filas)')
    print(f'    Lumpy-dead:     MA4            ({n_dead:,} filas)')
    print(f'    Lumpy-sparse:   Hurdle         ({n_sparse:,} filas)')
    print(f'    WMAPE ML puro:  {wmape_ml:.1f}%')
    print(f'    WMAPE híbrido:  {wmape_hybrid:.1f}%  (mejora: {wmape_ml - wmape_hybrid:+.1f}pp)')

    return df_final
