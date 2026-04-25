"""
Carga de las 5 fuentes de datos + filtrado B2B.
Encapsula todo el acceso a disco: ningún otro módulo lee archivos directamente.
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

# ── Festivos nacionales fijos (mes, día) ──────────────────────────────────
FESTIVOS_FIJOS = [
    (1, 1), (1, 6), (5, 1), (8, 15), (10, 12),
    (11, 1), (12, 6), (12, 8), (12, 25),
]

VIERNES_SANTOS = {
    2020: '2020-04-10', 2021: '2021-04-02', 2022: '2022-04-15',
    2023: '2023-04-07', 2024: '2024-03-29', 2025: '2025-04-18',
}

MESES_ES = {
    'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5,
    'junio': 6, 'julio': 7, 'agosto': 8, 'septiembre': 9,
    'octubre': 10, 'noviembre': 11, 'diciembre': 12,
}

REGION_MAP = {
    'GALICIA': 'Noroeste',
    'ASTURIAS': 'Norte', 'CANTABRIA': 'Norte',
    'PAIS VASCO': 'Norte', 'NAVARRA': 'Norte', 'LA RIOJA': 'Norte',
    'ARAGON': 'Noreste', 'CATALUÑA': 'Noreste', 'ISLAS BALEARES': 'Noreste',
    'COMUNIDAD DE MADRID': 'Centro',
    'CASTILLA Y LEON': 'Centro', 'CASTILLA-LA MANCHA': 'Centro', 'EXTREMADURA': 'Centro',
    'COMUNIDAD VALENCIANA': 'Este', 'REGION DE MURCIA': 'Sur',
    'ANDALUCIA': 'Sur',
    'CANARIAS': 'Canarias', 'CEUTA': 'Sur', 'MELILLA': 'Sur',
}


def parse_fecha_es(s: str) -> pd.Timestamp:
    """Convierte 'viernes, 30 de julio de 2021' → pd.Timestamp."""
    try:
        _, resto = str(s).split(', ', 1)
        day, _, month_es, _, year = resto.strip().split()
        return pd.Timestamp(year=int(year), month=MESES_ES[month_es.lower()], day=int(day))
    except Exception:
        return pd.NaT


def load_all_sources(cfg: dict) -> tuple:
    """
    Carga y limpia las 5 fuentes de datos.

    Returns
    -------
    df_raw_merged : DataFrame
        Albaranes con merges de cliente/provincia/canal aplicados (todos los años, incluye Fleet).
    df_art : DataFrame
        Maestro de artículos con atributos básicos.
    df_art_full : DataFrame
        Maestro artículos unido con familias CR_*.
    df_clima_nac : DataFrame
        Clima semanal agregado a nivel nacional.
    df_cicl_agg : DataFrame
        Eventos ciclistas por semana.
    """
    base = Path(cfg['data']['base_dir'])
    internal = base / cfg['data']['internal_dir']

    print('  Cargando transacciones...')
    df_raw = pd.read_excel(internal / 'LineasAlbaranCliente.xlsx')
    df_raw['fecha'] = df_raw['FechaAlbaran'].astype(str).apply(parse_fecha_es)
    df_raw = df_raw.dropna(subset=['fecha'])
    df_raw['anio'] = df_raw['fecha'].dt.isocalendar().year.astype(int)
    df_raw['semana_anio'] = df_raw['fecha'].dt.isocalendar().week.astype(int)
    df_raw['codigo_articulo'] = df_raw['CodigoArticulo'].astype(str).str.strip()
    df_raw['Unidades'] = pd.to_numeric(df_raw['Unidades'], errors='coerce').fillna(0)
    df_raw['ImporteNeto'] = pd.to_numeric(df_raw['ImporteNeto'], errors='coerce').fillna(0)
    df_raw['pct_desc2'] = pd.to_numeric(df_raw['%Descuento2'], errors='coerce').fillna(0)
    print(f'    Transacciones: {len(df_raw):,} filas')

    print('  Cargando maestro artículos...')
    df_art = pd.read_excel(
        internal / 'MaestroArticulos.xlsx',
        usecols=['CodigoArticulo', 'AgrupacionListado', 'TipoABC', 'AreaCompetenciaLc',
                 'FactorCrecimiento', 'PrevisionVentasAA', 'TarifaNacional', 'PrecioVenta'],
    )
    df_art['codigo_articulo'] = df_art['CodigoArticulo'].astype(str).str.strip()
    df_art['tipo_abc'] = df_art['TipoABC'].fillna('C').astype(str).str.upper().str[:1]
    df_art['factor_crecimiento'] = pd.to_numeric(df_art['FactorCrecimiento'], errors='coerce').fillna(1.0)
    df_art['prevision_ventas_aa'] = pd.to_numeric(df_art['PrevisionVentasAA'], errors='coerce').fillna(0.0)
    df_art['tarifa_nacional'] = pd.to_numeric(df_art['TarifaNacional'], errors='coerce').fillna(0.0)
    df_art['precio_unit'] = pd.to_numeric(df_art['PrecioVenta'], errors='coerce').fillna(0.0)

    print('  Cargando familias...')
    df_fam = pd.read_excel(
        internal / 'Familias Articulos.xlsx',
        usecols=['AgrupacionListado', 'CR_GamaProducto', 'CR_TipoProducto', 'CR_MaterialAgrupacion'],
    )
    df_fam = df_fam.dropna(subset=['AgrupacionListado'])
    df_fam['AgrupacionListado'] = pd.to_numeric(df_fam['AgrupacionListado'], errors='coerce')
    df_fam = df_fam.dropna(subset=['AgrupacionListado'])

    df_art_full = df_art.merge(df_fam, on='AgrupacionListado', how='left')
    for col in ['CR_GamaProducto', 'CR_TipoProducto', 'CR_MaterialAgrupacion']:
        df_art_full[col] = df_art_full[col].fillna('DESCONOCIDO').astype(str)
    df_art_full['AreaCompetenciaLc'] = df_art_full['AreaCompetenciaLc'].fillna('SIN_AREA').astype(str)

    print('  Cargando maestro clientes y provincias...')
    df_cli = pd.read_excel(
        internal / 'MaestroClientes.xlsx',
        usecols=['CodigoCliente', 'Municipio', 'Provincia', 'CodigoNacion'],
    )
    df_prov = pd.read_excel(
        internal / 'MaestroProvincias.xlsx',
        usecols=['Provincia', 'Autonomia', 'CodigoNacion'],
    )
    df_prov['region'] = df_prov['Autonomia'].map(REGION_MAP).fillna('Otros')

    df_can = pd.read_excel(internal / 'Agrupacion Canales venta.xlsx', header=0)
    df_can.columns = ['canal_raw', 'agrupacion_canal', 'tipo_agrupacion'] + list(df_can.columns[3:])
    df_can = df_can[['canal_raw', 'agrupacion_canal']].dropna(subset=['canal_raw'])

    print('  Cargando datos exógenos (clima, ciclismo)...')
    df_clima = pd.read_csv(base / 'clima_semanal_openmeteo.csv')
    df_clima.columns = [c.lower() for c in df_clima.columns]
    df_clima_nac = (
        df_clima.groupby(['year', 'semana'])
        .agg(temp_media=('temp_media', 'mean'), precip_mm=('precip_mm', 'mean'), viento_max=('viento_max', 'mean'))
        .reset_index()
        .rename(columns={'year': 'anio', 'semana': 'semana_anio'})
    )

    df_cicl = pd.read_excel(base / 'Calendario Ciclismo 22_24.xlsx')
    df_cicl.columns = [c.strip() for c in df_cicl.columns]
    df_cicl_agg = (
        df_cicl.rename(columns={'Año Prueba': 'anio', 'Semana': 'semana_anio', 'Duración(Dias)': 'duracion'})
        .groupby(['anio', 'semana_anio'])
        .agg(num_pruebas_cicl=('anio', 'count'), dias_pruebas_cicl=('duracion', 'sum'))
        .reset_index()
    )
    df_cicl_agg['hubo_prueba_cicl'] = 1

    # Merges lookup en df_raw
    df_raw = df_raw.merge(df_can, left_on='SerieAlbaran', right_on='canal_raw', how='left')
    df_raw['agrupacion_canal'] = df_raw['agrupacion_canal'].fillna('Otros')
    df_raw = df_raw.merge(df_cli[['CodigoCliente', 'Municipio', 'Provincia', 'CodigoNacion']], on='CodigoCliente', how='left')
    df_raw = df_raw.merge(df_prov[['Provincia', 'Autonomia', 'region']].drop_duplicates('Provincia'), on='Provincia', how='left')

    return df_raw, df_art, df_art_full, df_clima_nac, df_cicl_agg


def filter_b2b(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra al mercado nacional B2B representativo:
    - Solo España (CodigoNacion == 108)
    - Excluye Canal FLEET (licitaciones, no predecibles)
    - Excluye 2020 (anomalías COVID-19)

    Returns df_nac listo para agregación semanal.
    """
    df_es = df_raw[df_raw['CodigoNacion'] == 108].copy()
    df_nac = df_es[df_es['agrupacion_canal'] != 'FLEET'].copy()
    df_nac = df_nac[df_nac['anio'] >= 2021].copy()
    print(f'    Nacional B2B (sin Fleet, desde 2021): {len(df_nac):,} filas')
    return df_nac
