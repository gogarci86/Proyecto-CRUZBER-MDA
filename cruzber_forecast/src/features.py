"""
Ingeniería de variables: 47 features organizados en 6 bloques.
Todo shift respeta LAG_SAFETY_GAP para garantizar integridad anti-leakage.
"""
import numpy as np
import pandas as pd

GRP = 'codigo_articulo'


# ── Bloque 1: Calendario ──────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['mes'] = ((df['semana_anio'] - 1) // 4 + 1).clip(1, 12).astype(int)
    df['trimestre'] = ((df['mes'] - 1) // 3 + 1).astype(int)
    df['semana_del_mes'] = ((df['semana_anio'] - 1) % 4 + 1).astype(int)
    df['es_fin_mes'] = (df['semana_del_mes'] == 4).astype(int)
    df['sem_sin'] = np.sin(2 * np.pi * df['semana_anio'] / 52.18)
    df['sem_cos'] = np.cos(2 * np.pi * df['semana_anio'] / 52.18)
    df['temporada_alta'] = df['semana_anio'].isin(range(14, 40)).astype(int)
    return df


# ── Bloque 2: Autoregresivos ──────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    df = df.copy()
    for lag in [h, h + 4, h + 8, 52]:
        df[f'lag_{lag}w'] = df.groupby(GRP)['unidades'].transform(lambda x: x.shift(lag))
    return df


def add_rolling_features(df: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    df = df.copy()
    for w in [4, 8, 12]:
        df[f'roll_{w}w'] = df.groupby(GRP)['unidades'].transform(
            lambda x: x.shift(h).rolling(w, min_periods=1).mean()
        )
    for w in [8, 12]:
        df[f'roll_std_{w}w'] = df.groupby(GRP)['unidades'].transform(
            lambda x: x.shift(h).rolling(w, min_periods=2).std().fillna(0)
        )
    for span in [4, 8, 12]:
        df[f'ewm_{span}w'] = df.groupby(GRP)['unidades'].transform(
            lambda x: x.shift(h).ewm(span=span, adjust=False, min_periods=1).mean()
        )
    return df


def add_ratio_features(df: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    df = df.copy()
    roll4 = df.groupby(GRP)['unidades'].transform(
        lambda x: x.shift(h).rolling(4, min_periods=1).mean()
    )
    roll8 = df.groupby(GRP)['unidades'].transform(
        lambda x: x.shift(h).rolling(8, min_periods=1).mean()
    )
    roll4b = roll8 - roll4
    df['tendencia_4v4'] = (roll4 / roll4b.replace(0, np.nan)).fillna(1.0).clip(0.1, 10.0)

    lag52_honest = df.groupby(GRP)['unidades'].transform(lambda x: x.shift(max(52, h)))
    df['ratio_yoy'] = (df[f'lag_{h}w'] / (lag52_honest + 0.1)).clip(0.0, 20.0)
    return df


# ── Bloque 3: Comportamiento temporal ────────────────────────────────────

def add_tsls_features(df: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    """Semanas desde la última venta + frecuencia de venta reciente."""
    df = df.copy()
    df['had_sale'] = (df['unidades'] > 0).astype(int)

    def tsls_vectorized(s):
        mask = s > 0
        cumcount = mask.cumsum()
        last_sale_idx = cumcount.where(mask).ffill()
        result = cumcount - last_sale_idx
        result[last_sale_idx.isna()] = float('nan')
        return result

    df['tsls'] = df.groupby(GRP)['unidades'].transform(tsls_vectorized)
    df['tsls'] = df['tsls'].fillna(52).clip(upper=104)
    df['sale_freq_12w'] = df.groupby(GRP)['had_sale'].transform(
        lambda x: x.shift(h).rolling(12, min_periods=1).sum()
    )
    df.drop(columns=['had_sale'], inplace=True)
    return df


def add_lifecycle_features(df: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    """Edad del producto y ratio ventas recientes vs históricas."""
    df = df.copy()
    first_sale = df[df['unidades'] > 0].groupby(GRP).agg(
        first_period=('anio', 'first'),
        first_week=('semana_anio', 'first'),
    ).reset_index()
    df = df.merge(first_sale, on=GRP, how='left')
    df['producto_edad_semanas'] = (
        (df['anio'] - df['first_period'].fillna(df['anio'])) * 52
        + (df['semana_anio'] - df['first_week'].fillna(df['semana_anio']))
    ).clip(lower=0)
    roll_recent = df.groupby(GRP)['unidades'].transform(
        lambda x: x.shift(h).rolling(12, min_periods=1).mean()
    )
    roll_old = df.groupby(GRP)['unidades'].transform(
        lambda x: x.shift(h).rolling(52, min_periods=1).mean()
    )
    df['lifecycle_ratio'] = (roll_recent / (roll_old + 0.1)).clip(0, 10).fillna(1.0)
    df.drop(columns=['first_period', 'first_week'], inplace=True)
    return df


def add_stockout_proxy(df: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    """Proxy de rotura de stock: semanas a 0 con historial positivo."""
    df = df.copy()
    avg_recent = df.groupby(GRP)['unidades'].transform(
        lambda x: x.shift(h).rolling(12, min_periods=4).mean()
    )
    df['probable_stockout'] = ((df['unidades'] == 0) & (avg_recent > 2)).astype(int)
    df['stockout_freq_12w'] = df.groupby(GRP)['probable_stockout'].transform(
        lambda x: x.shift(h).rolling(12, min_periods=1).sum()
    )
    df.drop(columns=['probable_stockout'], inplace=True)
    return df


# ── Bloque 4: Atributos de catálogo ──────────────────────────────────────

def add_product_attrs(df_agg: pd.DataFrame, df_art_full: pd.DataFrame) -> pd.DataFrame:
    """Merge de atributos estáticos del maestro de artículos."""
    art_attrs = df_art_full[
        ['codigo_articulo', 'tipo_abc', 'factor_crecimiento', 'prevision_ventas_aa',
         'tarifa_nacional', 'precio_unit', 'AreaCompetenciaLc',
         'CR_GamaProducto', 'CR_TipoProducto', 'CR_MaterialAgrupacion']
    ].drop_duplicates('codigo_articulo')

    df_agg = df_agg.merge(art_attrs, on='codigo_articulo', how='left')
    df_agg['tipo_abc'] = df_agg['tipo_abc'].fillna('C').astype(str)
    df_agg['factor_crecimiento'] = df_agg['factor_crecimiento'].fillna(1.0)
    df_agg['prevision_ventas_aa'] = df_agg['prevision_ventas_aa'].fillna(0.0)
    df_agg['tarifa_nacional'] = df_agg['tarifa_nacional'].fillna(0.0)
    df_agg['precio_unit'] = df_agg['precio_unit'].fillna(0.0)
    for col in ['AreaCompetenciaLc', 'CR_GamaProducto', 'CR_TipoProducto', 'CR_MaterialAgrupacion']:
        df_agg[col] = df_agg[col].fillna('DESCONOCIDO').astype(str)
    df_agg['prevision_semanal'] = df_agg['prevision_ventas_aa'] / 52.0
    return df_agg


# ── Bloque 5: Target encoding (placeholder) ───────────────────────────────

def setup_target_encoding(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Inicializa columnas de TE (se recalculan dinámicamente en cada fold)."""
    df_agg = df_agg.copy()
    df_agg['te_codigo_articulo'] = 0.0
    df_agg['te_cr_gama'] = 0.0
    df_agg['te_area_comp'] = 0.0
    return df_agg


# ── Bloque 6: Target ──────────────────────────────────────────────────────

def build_target(df_agg: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Target: suma acumulada de unidades en las próximas horizon_weeks semanas."""
    h = cfg['forecast']['horizon_weeks']
    df_agg = df_agg.sort_values(['codigo_articulo', 'anio', 'semana_anio'])
    df_agg['target_12w_ahead'] = df_agg.groupby('codigo_articulo')['unidades'].transform(
        lambda x: sum(x.shift(-i) for i in range(1, h + 1))
    )
    df_agg = df_agg.dropna(subset=['target_12w_ahead']).copy()
    return df_agg


# ── Orquestador ───────────────────────────────────────────────────────────

def add_all_features(df_agg: pd.DataFrame, df_art_full: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Aplica todos los bloques de feature engineering en orden."""
    h = cfg['forecast']['lag_safety_gap']
    df_agg = add_time_features(df_agg)
    df_agg = add_lag_features(df_agg, h=h)
    df_agg = add_rolling_features(df_agg, h=h)
    df_agg = add_ratio_features(df_agg, h=h)
    df_agg = add_tsls_features(df_agg, h=h)
    df_agg = add_lifecycle_features(df_agg, h=h)
    df_agg = add_stockout_proxy(df_agg, h=h)
    df_agg = add_product_attrs(df_agg, df_art_full)
    df_agg = setup_target_encoding(df_agg)
    df_agg = build_target(df_agg, cfg)
    print(f'    Features generadas. Shape: {df_agg.shape}')
    return df_agg
