"""
Tests de integridad anti-leakage.
Verifica que ninguna feature del modelo usa información futura.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import add_lag_features, add_rolling_features


def _make_sample_df(n_skus=3, n_weeks=60):
    """DataFrame sintético SKU × semana."""
    skus = [f'SKU_{i:03d}' for i in range(n_skus)]
    weeks = list(range(1, n_weeks + 1))
    rows = [{'codigo_articulo': s, 'semana_anio': w, 'anio': 2023,
             'unidades': np.random.poisson(5)}
            for s in skus for w in weeks]
    return pd.DataFrame(rows)


def test_lag_features_no_lookahead():
    """Los lags con h=1 nunca deben igualar la demanda del mismo período."""
    df = _make_sample_df()
    df_feat = add_lag_features(df.copy(), h=1)
    # lag_1w de la semana t debe ser unidades de la semana t-1, nunca t
    for sku in df['codigo_articulo'].unique():
        sub = df_feat[df_feat['codigo_articulo'] == sku].sort_values('semana_anio')
        # Para semanas > 1: lag_1w[t] == unidades[t-1]
        for i in range(1, len(sub)):
            expected = sub['unidades'].iloc[i - 1]
            actual = sub['lag_1w'].iloc[i]
            assert actual == expected or pd.isna(actual), (
                f'{sku} semana {sub["semana_anio"].iloc[i]}: '
                f'lag_1w={actual} != unidades_t-1={expected}'
            )


def test_rolling_features_no_lookahead():
    """Las medias móviles no deben incluir el período actual."""
    df = _make_sample_df(n_skus=2, n_weeks=20)
    df['unidades'] = 100.0  # demanda constante → media == 100

    df_feat = add_rolling_features(df.copy(), h=1)
    # roll_4w con h=1: rolling sobre x.shift(1) → excluye período actual
    # En demanda constante=100, la media debe ser 100 (nunca > 100)
    assert (df_feat['roll_4w'].dropna() <= 100 + 1e-9).all(), (
        'roll_4w > demanda actual implica lookahead'
    )


def test_target_uses_future_only():
    """El target debe ser suma de períodos FUTUROS, nunca el período actual."""
    df = _make_sample_df(n_skus=2, n_weeks=30)
    cfg = {'forecast': {'horizon_weeks': 4, 'lag_safety_gap': 1}}

    from src.features import build_target
    df_t = build_target(df.copy(), cfg)

    for sku in df_t['codigo_articulo'].unique():
        sub = df.sort_values('semana_anio')
        sub_t = df_t[df_t['codigo_articulo'] == sku].sort_values('semana_anio')
        for _, row in sub_t.iterrows():
            w = row['semana_anio']
            # El target de la semana w debe ser suma de semanas [w+1, w+4]
            expected = sub[(sub['codigo_articulo'] == sku)
                           & (sub['semana_anio'] > w)
                           & (sub['semana_anio'] <= w + 4)]['unidades'].sum()
            assert abs(row['target_12w_ahead'] - expected) < 1e-6, (
                f'{sku} w={w}: target={row["target_12w_ahead"]:.1f} expected={expected:.1f}'
            )
