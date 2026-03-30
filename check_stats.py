import pandas as pd
df = pd.read_csv(r'c:\Users\03946542\OneDrive - Mango\ISDI_MDA13\Proyecto-CRUZBER-MDA\Datasets\df_final_modelado.csv', sep=';')
print("Mean units per SKU/week/location:", df['unidades'].mean())
print("Median units per SKU/week/location:", df['unidades'].median())
print("Max units per SKU/week/location:", df['unidades'].max())
print("\nMean units per Segment ABC:")
print(df.groupby('tipo_abc')['unidades'].mean())

print("\nMedian units per Segment ABC:")
print(df.groupby('tipo_abc')['unidades'].median())

print("\nMax units per Segment ABC:")
print(df.groupby('tipo_abc')['unidades'].max())

# Average units per SKU per Week (ignoring location if that's what the user meant, 
# but usually it's per observation in the model)
# The model observation is SKU per Week per Location (Municipio) or just SKU per Week?
# In 02 notebook, it was grouped by ['anio', 'semana_anio', 'Provincia', 'Municipio', 'codigo_articulo']

# If user meant per SKU per Week across all locations:
# df_sku_week = df.groupby(['anio', 'semana_anio', 'codigo_articulo'])['unidades'].sum()
# print("\nMean units per SKU per Week (total):", df_sku_week.mean())
