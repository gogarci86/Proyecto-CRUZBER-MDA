"""
Tests de reproducibilidad.
Verifica que dos ejecuciones consecutivas producen resultados idénticos.
"""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classification import classify_syntetos_boylan, subsegment_lumpy
from src.features import add_time_features, add_lag_features, add_rolling_features


def _make_panel(n_skus=5, n_weeks=80, seed=42):
    rng = np.random.default_rng(seed)
    skus = [f'ART{i:04d}' for i in range(n_skus)]
    rows = []
    for s in skus:
        for w in range(1, n_weeks + 1):
            anio = 2022 if w <= 52 else 2023
            sem = w if w <= 52 else w - 52
            rows.append({
                'codigo_articulo': s,
                'anio': anio,
                'semana_anio': sem,
                'unidades': int(rng.poisson(3)),
            })
    return pd.DataFrame(rows)


def test_classification_deterministic():
    """classify_syntetos_boylan devuelve exactamente las mismas clases en dos llamadas."""
    df = _make_panel()
    cfg = {
        'forecast': {'test_year': 2023},
        'classification': {
            'adi_threshold': 1.32,
            'cv2_threshold': 0.49,
            'min_periods_stable': 52,
            'min_count_stable': 10,
        },
    }
    df1, _ = classify_syntetos_boylan(df.copy(), cfg)
    df2, _ = classify_syntetos_boylan(df.copy(), cfg)

    pd.testing.assert_series_equal(
        df1['sb_class'].reset_index(drop=True),
        df2['sb_class'].reset_index(drop=True),
    )


def test_feature_engineering_deterministic():
    """El feature engineering produce valores idénticos en dos pasadas."""
    df = _make_panel()
    df1 = add_time_features(df.copy())
    df1 = add_lag_features(df1, h=1)
    df1 = add_rolling_features(df1, h=1)

    df2 = add_time_features(df.copy())
    df2 = add_lag_features(df2, h=1)
    df2 = add_rolling_features(df2, h=1)

    for col in ['mes', 'lag_1w', 'roll_4w', 'ewm_4w']:
        pd.testing.assert_series_equal(
            df1[col].reset_index(drop=True),
            df2[col].reset_index(drop=True),
            check_names=False,
        )


def test_config_frozen_params_present():
    """El fichero config.yaml contiene los parámetros congelados de ambos modelos."""
    import yaml
    cfg_path = Path(__file__).parent.parent / 'config.yaml'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    assert 'frozen_params' in cfg
    assert 'smooth' in cfg['frozen_params']
    assert 'hurdle_classifier' in cfg['frozen_params']
    assert 'hurdle_regressor' in cfg['frozen_params']

    # Parámetros mínimos presentes
    for key in ['learning_rate', 'depth', 'l2_leaf_reg']:
        assert key in cfg['frozen_params']['smooth'], f'smooth.{key} missing'
        assert key in cfg['frozen_params']['hurdle_classifier'], f'clf.{key} missing'
        assert key in cfg['frozen_params']['hurdle_regressor'], f'reg.{key} missing'
