"""
Predicciones de referencia: baseline naive (MM12W), MA4, Croston y SBA.
"""
import numpy as np
import pandas as pd


def compute_baselines(df_agg: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Añade baseline_naive (MM 12W) y ma4_pred (MA4 × 12) a df_agg.
    Ambas respetan lag_safety_gap para evitar leakage.
    """
    h = cfg['forecast']['lag_safety_gap']

    df_agg = df_agg.copy()
    df_agg['baseline_naive'] = (
        df_agg.groupby('codigo_articulo')['unidades']
        .transform(lambda x: x.shift(h).rolling(12, min_periods=1).sum())
        .fillna(0).clip(lower=0)
    )
    df_agg['ma4_pred'] = (
        df_agg.groupby('codigo_articulo')['unidades']
        .transform(lambda x: x.shift(h).rolling(4, min_periods=1).mean() * 12)
        .fillna(0).clip(lower=0)
    )

    test_year = cfg['forecast']['test_year']
    bl_mean = df_agg.loc[df_agg['anio'] == test_year, 'baseline_naive'].mean()
    ma4_mean = df_agg.loc[df_agg['anio'] == test_year, 'ma4_pred'].mean()
    print(f'    baseline_naive media test: {bl_mean:.2f}  |  ma4_pred media test: {ma4_mean:.2f}')

    return df_agg


def _croston_forecast(series: np.ndarray, alpha: float = 0.1, horizon: int = 12) -> float:
    """Croston (1972): predicción acumulada para las próximas `horizon` semanas."""
    q = 0
    z_hat = p_hat = None
    for val in series:
        q += 1
        if val > 0:
            if z_hat is None:
                z_hat, p_hat = val, q
            else:
                z_hat = alpha * val + (1 - alpha) * z_hat
                p_hat = alpha * q + (1 - alpha) * p_hat
            q = 0
    if z_hat is None or p_hat is None or p_hat == 0:
        return 0.0
    return max(z_hat / p_hat * horizon, 0)


def _sba_forecast(series: np.ndarray, alpha: float = 0.1, horizon: int = 12) -> float:
    """SBA (Syntetos-Boylan 2005): Croston con corrección de sesgo."""
    return _croston_forecast(series, alpha, horizon) * (1 - alpha / 2)


def compute_croston_sba(df_agg: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Calcula predicciones Croston y SBA para cada SKU Intermittent × semana de test.
    Usa únicamente datos históricos anteriores a cada semana (anti-leakage).
    """
    alpha = cfg['croston']['alpha']
    horizon = cfg['croston']['horizon']
    h = cfg['forecast']['lag_safety_gap']
    test_year = cfg['forecast']['test_year']

    int_skus = df_agg[df_agg['sb_class'] == 'Intermittent']['codigo_articulo'].unique()
    test_semanas = sorted(df_agg[df_agg['anio'] == test_year]['semana_anio'].unique())

    print(f'    Croston/SBA: {len(int_skus)} SKUs × {len(test_semanas)} semanas...')
    rows = []
    for sku in int_skus:
        sku_df = df_agg[df_agg['codigo_articulo'] == sku].sort_values(['anio', 'semana_anio'])
        for semana in test_semanas:
            mask_hist = (
                (sku_df['anio'] < test_year)
                | ((sku_df['anio'] == test_year) & (sku_df['semana_anio'] < semana - h))
            )
            series = sku_df.loc[mask_hist, 'unidades'].values
            if len(series) < 12:
                c_pred = s_pred = 0.0
            else:
                c_pred = _croston_forecast(series, alpha, horizon)
                s_pred = _sba_forecast(series, alpha, horizon)
            rows.append({
                'codigo_articulo': sku,
                'semana_anio': semana,
                'croston_pred': c_pred,
                'sba_pred': s_pred,
            })

    df_cs = pd.DataFrame(rows)

    for col in ['croston_pred', 'sba_pred']:
        if col in df_agg.columns:
            df_agg = df_agg.drop(columns=[col])
    df_agg = df_agg.merge(df_cs, on=['codigo_articulo', 'semana_anio'], how='left')
    df_agg['croston_pred'] = df_agg['croston_pred'].fillna(0).clip(lower=0)
    df_agg['sba_pred'] = df_agg['sba_pred'].fillna(0).clip(lower=0)

    int_test = df_agg[(df_agg['anio'] == test_year) & (df_agg['sb_class'] == 'Intermittent')]
    real_int = int_test['target_12w_ahead'].values
    denom = max(np.abs(real_int).sum(), 1)
    wmape_bl = np.abs(real_int - int_test['baseline_naive'].values).sum() / denom * 100
    wmape_cr = np.abs(real_int - int_test['croston_pred'].values).sum() / denom * 100
    wmape_sba = np.abs(real_int - int_test['sba_pred'].values).sum() / denom * 100
    print(f'    Intermittent — Baseline: {wmape_bl:.1f}%  Croston: {wmape_cr:.1f}%  SBA: {wmape_sba:.1f}%')

    return df_agg
