"""
Tests del output final.
Verifica estructura, rangos y consistencia del DataFrame exportado.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hybrid_strategy import apply_hybrid_strategy


def _make_df_final(n_skus=10, n_weeks=5):
    """DataFrame sintético que simula df_final post-concat."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_skus):
        sku = f'ART{i:04d}'
        sb = ['Smooth', 'Erratic', 'Intermittent', 'Lumpy'][i % 4]
        subtype = 'n/a' if sb != 'Lumpy' else (['dead', 'sparse'][i % 2])
        for w in range(1, n_weeks + 1):
            real = float(rng.poisson(10))
            rows.append({
                'codigo_articulo': sku,
                'semana_anio': w,
                'anio': 2024,
                'sb_class': sb,
                'lumpy_subtype': subtype,
                'tipo_abc': 'B',
                'real': real,
                'pred': float(rng.poisson(10)),
                'pred_p10': float(rng.poisson(5)),
                'pred_p90': float(rng.poisson(15)),
                'baseline_naive': float(rng.poisson(9)),
                'ma4_pred': float(rng.poisson(8)),
                'croston_pred': float(rng.poisson(7)),
                'sba_pred': float(rng.poisson(7)),
            })
    return pd.DataFrame(rows)


def test_hybrid_strategy_no_negative_preds():
    """La estrategia híbrida no debe producir predicciones negativas."""
    df = _make_df_final()
    df_out = apply_hybrid_strategy(df)
    assert (df_out['pred'] >= 0).all(), 'Predicciones negativas después de la estrategia híbrida'


def test_hybrid_strategy_lumpy_dead_uses_ma4():
    """Los SKUs Lumpy-dead deben usar ma4_pred como predicción."""
    df = _make_df_final()
    df_out = apply_hybrid_strategy(df)

    mask = (df_out['sb_class'] == 'Lumpy') & (df_out['lumpy_subtype'] == 'dead')
    if mask.sum() > 0:
        pd.testing.assert_series_equal(
            df_out.loc[mask, 'pred'].reset_index(drop=True),
            df_out.loc[mask, 'ma4_pred'].reset_index(drop=True),
            check_names=False,
        )


def test_hybrid_strategy_smooth_erratic_unchanged():
    """Los SKUs Smooth/Erratic no deben modificarse (pred == pred_ml)."""
    df = _make_df_final()
    df_out = apply_hybrid_strategy(df)

    mask = df_out['sb_class'].isin(['Smooth', 'Erratic'])
    if mask.sum() > 0:
        pd.testing.assert_series_equal(
            df_out.loc[mask, 'pred'].reset_index(drop=True),
            df_out.loc[mask, 'pred_ml'].reset_index(drop=True),
            check_names=False,
        )


def test_error_abs_consistency():
    """error_abs debe ser |real - pred|, nunca negativo."""
    df = _make_df_final()
    df_out = apply_hybrid_strategy(df)

    expected = (df_out['real'] - df_out['pred']).abs()
    pd.testing.assert_series_equal(
        df_out['error_abs'].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )
    assert (df_out['error_abs'] >= 0).all()
