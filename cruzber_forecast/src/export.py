"""
Exportación del output final a Excel BI-ready.
Incluye desagregación provincial top-down, impacto financiero y semáforo de confianza.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def _build_provincial_profile(df_nac: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Calcula cuota provincial histórica por SKU a partir del período de entrenamiento."""
    train_years = cfg['forecast']['train_years']
    df_train = df_nac[df_nac['anio'].isin(train_years)].copy()

    prov_sku = (
        df_train.groupby(['codigo_articulo', 'Provincia'])['Unidades']
        .sum().reset_index()
    )
    prov_sku.columns = ['codigo_articulo', 'Provincia', 'uds_prov']
    prov_sku['uds_prov'] = prov_sku['uds_prov'].clip(lower=0)

    total_sku = prov_sku.groupby('codigo_articulo')['uds_prov'].sum().reset_index()
    total_sku.columns = ['codigo_articulo', 'uds_total']
    prov_sku = prov_sku.merge(total_sku, on='codigo_articulo')
    prov_sku['share'] = prov_sku['uds_prov'] / prov_sku['uds_total'].replace(0, 1)

    top1 = (
        prov_sku.sort_values(['codigo_articulo', 'uds_prov'], ascending=[True, False])
        .drop_duplicates('codigo_articulo')[['codigo_articulo', 'Provincia', 'share']]
    )
    top1.columns = ['codigo_articulo', 'top1_prov', 'share_top1_prov']

    n_provs = (
        prov_sku[prov_sku['uds_prov'] > 0]
        .groupby('codigo_articulo')['Provincia'].nunique().reset_index()
    )
    n_provs.columns = ['codigo_articulo', 'n_provs_activas']

    hhi = (
        prov_sku.groupby('codigo_articulo')
        .apply(lambda g: (g['share'] ** 2).sum())
        .reset_index()
    )
    hhi.columns = ['codigo_articulo', 'hhi_prov']

    perfil = top1.merge(n_provs, on='codigo_articulo', how='left')
    perfil = perfil.merge(hhi, on='codigo_articulo', how='left')
    perfil['n_provs_activas'] = perfil['n_provs_activas'].fillna(0).astype(int)
    perfil['hhi_prov'] = perfil['hhi_prov'].fillna(1.0)
    perfil['share_top1_prov'] = perfil['share_top1_prov'].fillna(0.0)

    return perfil


def _enrich_maestro(df_final: pd.DataFrame, df_art: pd.DataFrame,
                    cfg: dict) -> pd.DataFrame:
    """Añade descripción, familia y coste de escandallo desde el maestro."""
    base = Path(cfg['data']['base_dir']) / cfg['data']['internal_dir']
    try:
        desc_art = pd.read_excel(
            base / 'MaestroArticulos.xlsx',
            usecols=['CodigoArticulo', 'DescripcionArticulo', 'CodigoFamilia', 'CosteEscandallo'],
        )
        desc_art['codigo_articulo'] = desc_art['CodigoArticulo'].astype(str).str.strip()
        desc_art['descripcion'] = desc_art['DescripcionArticulo'].fillna('Sin descripcion').astype(str)
        desc_art['familia'] = desc_art['CodigoFamilia'].fillna('N/A').astype(str)
        desc_art['coste_escandallo'] = pd.to_numeric(
            desc_art['CosteEscandallo'], errors='coerce'
        ).fillna(0.0)
        desc_art = desc_art[
            ['codigo_articulo', 'descripcion', 'familia', 'coste_escandallo']
        ].drop_duplicates('codigo_articulo')
    except Exception as e:
        print(f'    Aviso — descripcion/coste no disponibles: {e}')
        desc_art = pd.DataFrame(columns=['codigo_articulo', 'descripcion', 'familia', 'coste_escandallo'])

    maestro = df_art[['codigo_articulo', 'precio_unit']].drop_duplicates('codigo_articulo')
    maestro = maestro.merge(desc_art, on='codigo_articulo', how='left')
    maestro['descripcion'] = maestro['descripcion'].fillna('Sin descripcion')
    maestro['familia'] = maestro['familia'].fillna('N/A')
    maestro['coste_escandallo'] = maestro['coste_escandallo'].fillna(0.0)

    df_final = df_final.merge(maestro, on='codigo_articulo', how='left', suffixes=('', '_m'))
    for col in ['precio_unit', 'descripcion', 'familia', 'coste_escandallo']:
        if col + '_m' in df_final.columns:
            df_final[col] = df_final[col + '_m'].fillna(df_final.get(col, 0))
            df_final.drop(columns=[col + '_m'], inplace=True)
    df_final['precio_unit'] = df_final['precio_unit'].fillna(0.0)

    return df_final


def export_xlsx(df_final: pd.DataFrame, df_nac: pd.DataFrame, df_clima_nac: pd.DataFrame,
                df_art: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Construye el output final BI-ready y lo exporta a Excel.

    Steps
    -----
    A. Perfil provincial (top1, n_provs, HHI, share)
    B. Desagregación top-down (pred_top1_prov, pred_resto_provs)
    C. Enriquecimiento desde maestro (descripcion, familia, coste_escandallo)
    D. Impacto financiero (ventas_riesgo_eur, capital_inmovilizado_eur)
    E. Semáforo de confianza
    F. Cobertura
    G. Export Excel

    Returns df_output con las columnas finales.
    """
    test_year = cfg['forecast']['test_year']
    output_file = cfg['data']['output_file']

    # A. Perfil provincial
    perfil = _build_provincial_profile(df_nac, cfg)
    df_final = df_final.merge(perfil, on='codigo_articulo', how='left')
    df_final['top1_prov'] = df_final['top1_prov'].fillna('SIN_DATOS')
    df_final['n_provs_activas'] = df_final['n_provs_activas'].fillna(0).astype(int)
    df_final['hhi_prov'] = df_final['hhi_prov'].fillna(1.0)
    df_final['share_top1_prov'] = df_final['share_top1_prov'].fillna(0.0)

    temp_media_global = df_clima_nac['temp_media'].mean() if 'temp_media' in df_clima_nac.columns else 15.3
    df_final['temp_media_top1'] = temp_media_global
    df_final['temp_max_top1'] = 20.057
    df_final['temp_range_top1'] = 9.257

    # B. Desagregación top-down
    df_final['pred_top1_prov'] = df_final['pred'] * df_final['share_top1_prov']
    df_final['pred_resto_provs'] = df_final['pred'] * (1 - df_final['share_top1_prov'])

    # C. Enriquecimiento maestro
    df_final = _enrich_maestro(df_final, df_art, cfg)

    # D. Impacto financiero
    df_final['subestimacion_uds'] = (-df_final['sesgo']).clip(lower=0)
    df_final['sobreestimacion_uds'] = df_final['sesgo'].clip(lower=0)
    df_final['ventas_riesgo_eur'] = df_final['subestimacion_uds'] * df_final['precio_unit']
    df_final['capital_inmovilizado_eur'] = (
        df_final['sobreestimacion_uds'] * df_final['coste_escandallo']
    )

    # E. Semáforo de confianza
    def _confianza(row):
        if row['sb_class'] == 'Smooth' and row.get('sb_reliability', '') == 'stable':
            return 'Alta'
        elif row['sb_class'] in ['Smooth', 'Erratic']:
            return 'Media'
        return 'Baja'

    df_final['confianza'] = df_final.apply(_confianza, axis=1)

    # F. Cobertura
    df_final['cobertura_pct'] = (
        df_final['pred'] / df_final['real'].clip(lower=0.01) * 100
    ).clip(upper=999)

    # G. Export
    OUTPUT_COLS = [
        'anio', 'semana_anio', 'codigo_articulo', 'descripcion', 'familia',
        'tipo_abc', 'sb_class', 'sb_reliability', 'confianza',
        'top1_prov', 'n_provs_activas', 'hhi_prov', 'share_top1_prov',
        'temp_media_top1', 'temp_max_top1', 'temp_range_top1',
        'real', 'pred', 'pred_p10', 'pred_p90', 'cobertura_pct',
        'pred_top1_prov', 'pred_resto_provs',
        'error_abs', 'sesgo',
        'precio_unit', 'coste_escandallo',
        'ventas_riesgo_eur', 'capital_inmovilizado_eur',
    ]

    missing = [c for c in OUTPUT_COLS if c not in df_final.columns]
    if missing:
        print(f'    Aviso — columnas faltantes (excluidas): {missing}')
    cols = [c for c in OUTPUT_COLS if c in df_final.columns]
    df_output = df_final[cols].copy()

    for col in ['pred', 'pred_p10', 'pred_p90', 'pred_top1_prov', 'pred_resto_provs',
                'error_abs', 'sesgo', 'cobertura_pct',
                'ventas_riesgo_eur', 'capital_inmovilizado_eur']:
        if col in df_output.columns:
            df_output[col] = df_output[col].round(2)

    df_output.to_excel(output_file, sheet_name='predicciones', index=False)

    print(f'\n    Exportado: {len(df_output):,} filas x {len(cols)} columnas → {output_file}')
    print(f'    sb_class:  {df_output["sb_class"].value_counts().to_dict()}')
    print(f'    confianza: {df_output["confianza"].value_counts().to_dict()}')
    ventas_riesgo = df_output['ventas_riesgo_eur'].sum() if 'ventas_riesgo_eur' in df_output.columns else 0
    capital_inm = df_output['capital_inmovilizado_eur'].sum() if 'capital_inmovilizado_eur' in df_output.columns else 0
    print(f'    Ventas en riesgo: {ventas_riesgo:,.0f} €  |  Capital inmovilizado: {capital_inm:,.0f} €')

    return df_output
