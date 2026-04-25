"""
Clasificación estadística de demanda (Syntetos-Boylan) y subsegmentación Lumpy.
"""
import numpy as np
import pandas as pd


def classify_syntetos_boylan(df_agg: pd.DataFrame, cfg: dict) -> tuple:
    """
    Clasifica SKUs según Syntetos-Boylan usando ADI y CV².

    Usa solo el subconjunto de entrenamiento para evitar leakage.
    Returns
    -------
    df_agg : DataFrame con columnas sb_class, sb_reliability, ADI, CV²
    demand_stats : DataFrame con estadísticas por SKU (para subsegmentación)
    """
    test_year = cfg['forecast']['test_year']
    adi_th = cfg['classification']['adi_threshold']
    cv2_th = cfg['classification']['cv2_threshold']
    min_periods = cfg['classification']['min_periods_stable']
    min_count = cfg['classification']['min_count_stable']

    train_subset = df_agg[df_agg['anio'] < test_year].copy()

    demand_stats = (
        train_subset[train_subset['unidades'] > 0]
        .groupby('codigo_articulo')
        .agg(
            mean_demand=('unidades', 'mean'),
            std_demand=('unidades', 'std'),
            count_demand=('codigo_articulo', 'count'),
        )
    )
    demand_stats['CV2'] = (demand_stats['std_demand'] / demand_stats['mean_demand']) ** 2

    total_periods = train_subset.groupby('codigo_articulo').size()
    demand_stats['total_periods'] = total_periods
    demand_stats['ADI'] = demand_stats['total_periods'] / demand_stats['count_demand']
    demand_stats['CV2'] = demand_stats['CV2'].fillna(0)

    def _classify_demand(row):
        adi, cv2 = row['ADI'], row['CV2']
        if pd.isna(adi) or np.isinf(adi):
            return 'Lumpy'
        if adi < adi_th and cv2 < cv2_th:
            return 'Smooth'
        elif adi < adi_th and cv2 >= cv2_th:
            return 'Erratic'
        elif adi >= adi_th and cv2 < cv2_th:
            return 'Intermittent'
        else:
            return 'Lumpy'

    def _classify_reliability(row):
        if row['count_demand'] == 0:
            return 'zero_only_or_newborn'
        elif row['total_periods'] >= min_periods and row['count_demand'] >= min_count:
            return 'stable'
        else:
            return 'unstable'

    demand_stats['sb_class'] = demand_stats.apply(_classify_demand, axis=1)
    demand_stats['sb_reliability'] = demand_stats.apply(_classify_reliability, axis=1)

    df_agg = df_agg.merge(
        demand_stats[['sb_class', 'sb_reliability', 'ADI', 'CV2']],
        on='codigo_articulo', how='left'
    )
    df_agg['sb_class'] = df_agg['sb_class'].fillna('Lumpy')
    df_agg['sb_reliability'] = df_agg['sb_reliability'].fillna('zero_only_or_newborn')

    dist = df_agg.drop_duplicates('codigo_articulo')['sb_class'].value_counts()
    print(f'    Clasificación Syntetos-Boylan:\n{dist.to_string()}')

    return df_agg, demand_stats


def subsegment_lumpy(df_agg: pd.DataFrame, demand_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Subsegmenta el cluster Lumpy en 'dead' y 'sparse'.

    dead : activity_density < 5% o zero_only_or_newborn
    sparse : Lumpy con señal aprovechable (density >= 5%)
    """
    density_threshold = 0.05

    demand_stats = demand_stats.copy()
    demand_stats['activity_density'] = (
        demand_stats['count_demand'] / demand_stats['total_periods']
    )

    def _classify_lumpy_subtype(row):
        if row['sb_class'] != 'Lumpy':
            return 'n/a'
        if row.get('sb_reliability', '') == 'zero_only_or_newborn':
            return 'dead'
        elif row['activity_density'] < density_threshold:
            return 'dead'
        else:
            return 'sparse'

    demand_stats['lumpy_subtype'] = demand_stats.apply(_classify_lumpy_subtype, axis=1)

    lumpy_cols = demand_stats[['lumpy_subtype', 'activity_density']].copy()
    for col in ['lumpy_subtype', 'activity_density']:
        if col in df_agg.columns:
            df_agg = df_agg.drop(columns=[col])
    df_agg = df_agg.merge(lumpy_cols, on='codigo_articulo', how='left')
    df_agg['lumpy_subtype'] = df_agg['lumpy_subtype'].fillna('n/a')
    df_agg['activity_density'] = df_agg['activity_density'].fillna(0)

    lumpy_mask = demand_stats['sb_class'] == 'Lumpy'
    dist = demand_stats.loc[lumpy_mask, 'lumpy_subtype'].value_counts()
    print(f'    Subsegmentación Lumpy:\n{dist.to_string()}')

    return df_agg
