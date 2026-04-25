"""
Agregación semanal SKU × nacional y densificación (Dense Panel).
Convierte el dataset sparse (solo semanas con ventas) en un panel denso
donde cada SKU tiene una fila por semana, con ceros explícitos.
"""
import pandas as pd
import numpy as np
from .data_loader import FESTIVOS_FIJOS, VIERNES_SANTOS


def _get_festivos_espana(anios: list) -> set:
    festivos = set()
    for y in anios:
        for m, d in FESTIVOS_FIJOS:
            festivos.add((y, m, d))
        if y in VIERNES_SANTOS:
            vs = pd.Timestamp(VIERNES_SANTOS[y])
            festivos.add((vs.year, vs.month, vs.day))
    return festivos


def _dias_laborables_iso(year: int, week: int, festivos_set: set) -> int:
    try:
        lunes = pd.Timestamp.fromisocalendar(int(year), int(week), 1)
    except ValueError:
        return 5
    count = 0
    for delta in range(5):
        dia = lunes + pd.Timedelta(days=delta)
        if (dia.year, dia.month, dia.day) not in festivos_set:
            count += 1
    return count


def build_dense_panel(df_nac: pd.DataFrame, df_clima_nac: pd.DataFrame,
                      df_cicl_agg: pd.DataFrame, cfg: dict) -> tuple:
    """
    Construye el Dense Panel: matriz SKU × semana con ceros explícitos.

    Returns
    -------
    df_agg : DataFrame
        Panel denso con aggregados semanales + festivos + clima + ciclismo.
    semanas_unicas : DataFrame
        Grilla de semanas con dias_laborables_semana calculados.
    """
    train_years = cfg['forecast']['train_years']
    test_year = cfg['forecast']['test_year']
    all_years = train_years + [test_year]

    # Días laborables por semana
    festivos_set = _get_festivos_espana(list(range(min(all_years), max(all_years) + 2)))
    semanas_unicas = df_nac[['anio', 'semana_anio']].drop_duplicates().copy()
    semanas_unicas['dias_laborables_semana'] = semanas_unicas.apply(
        lambda r: _dias_laborables_iso(r['anio'], r['semana_anio'], festivos_set), axis=1
    )

    # Agregación semanal
    GROUP_NAC = ['anio', 'semana_anio', 'codigo_articulo']

    df_agg = (
        df_nac.groupby(GROUP_NAC, as_index=False)
        .agg(unidades=('Unidades', 'sum'), importe_neto=('ImporteNeto', 'sum'))
    )

    def wmean_desc(g):
        w = g['Unidades'].abs()
        v = g['pct_desc2']
        denom = w.sum()
        return (v * w).sum() / denom if denom > 0 else 0.0

    desc_agg = (
        df_nac.groupby(GROUP_NAC)
        .apply(wmean_desc)
        .reset_index(name='por_descuento2')
    )
    df_agg = df_agg.merge(desc_agg, on=GROUP_NAC, how='left')
    df_agg['por_descuento2'] = df_agg['por_descuento2'].fillna(0.0)
    df_agg['unidades'] = df_agg['unidades'].clip(lower=0)

    print(f'    Sparse: {len(df_agg):,} filas')

    # Densificación (producto cartesiano SKU × semana)
    semanas_grid = semanas_unicas[['anio', 'semana_anio']].copy()
    semanas_grid['key'] = 1
    unique_skus = df_nac['codigo_articulo'].unique()
    skus_grid = pd.DataFrame({'codigo_articulo': unique_skus, 'key': 1})
    dense_grid = semanas_grid.merge(skus_grid, on='key').drop(columns=['key'])

    df_agg = dense_grid.merge(df_agg, on=['codigo_articulo', 'anio', 'semana_anio'], how='left')
    df_agg['unidades'] = df_agg['unidades'].fillna(0)
    df_agg['importe_neto'] = df_agg['importe_neto'].fillna(0)
    df_agg['por_descuento2'] = df_agg['por_descuento2'].fillna(0)

    print(f'    Dense: {len(df_agg):,} filas ({df_agg["codigo_articulo"].nunique()} SKUs)')

    # Merge calendario, clima y ciclismo
    df_agg = df_agg.merge(semanas_unicas, on=['anio', 'semana_anio'], how='left')
    df_agg = df_agg.merge(df_clima_nac, on=['anio', 'semana_anio'], how='left')
    df_agg = df_agg.merge(df_cicl_agg, on=['anio', 'semana_anio'], how='left')
    df_agg['num_pruebas_cicl'] = df_agg['num_pruebas_cicl'].fillna(0).astype(int)
    df_agg['dias_pruebas_cicl'] = df_agg['dias_pruebas_cicl'].fillna(0)
    df_agg['hubo_prueba_cicl'] = df_agg['hubo_prueba_cicl'].fillna(0).astype(int)

    df_agg = df_agg.sort_values(['codigo_articulo', 'anio', 'semana_anio']).reset_index(drop=True)
    return df_agg, semanas_unicas
