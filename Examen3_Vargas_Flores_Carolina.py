# ==========================================================================
# TAREA 1: INGESTA, DIAGNÓSTICO Y SUSTITUCIÓN DE CENTINELAS
# ==========================================================================
import pandas as pd
import numpy as np
# 1. Cargar o definir el dataset
data = {
    'ID_Registro': ['REG-001', 'REG-002', 'REG-003', 'REG-004', 'REG-005',
                    'REG-006', 'REG-007', 'REG-008', 'REG-009', 'REG-010', 'REG-010'],
    'Timestamp': ['2026-10-01 08:00:00', '2026-10-01 09:00:00', '2026-10-01 10:00:00',
                  '2026-10-01 11:00:00', '2026-10-01 12:00:00', '2026-10-01 13:00:00',
                  '2026-10-01 14:00:00', '2026-10-01 15:00:00', '2026-10-01 16:00:00',
                  '2026-10-01 17:00:00', '2026-10-01 17:00:00'],
    'Zona_Estacion': ['Centro_Historico', 'centro_historico', 'CENTRO_HISTORICO',
                      'Zona_Oriente', 'Zona_Oriente', 'zona_oriente', 'Zona_Sur',
                      'Zona_Sur', 'Zona_Sur', 'Zona_Sur', 'Zona_Sur'],
    'Concentracion_PM25': [35.4, 42.1, -999.0, 120.5, np.nan, 310.0, 18.2, 20.1, 19.5, 19.5, 19.5],
    'Nivel_NO2': [42.1, 48.5, 51.0, 89.2, 82.0, 195.0, 22.4, -999.0, 25.1, 25.1, 25.1],
    'Indice_Trafico_Vehicular': [85.0, 92.5, 88.0, 98.0, 91.2, 15.0, 45.0, 42.1, np.nan, 41.0, 41.0],
    'Estado_Sensor': ['Normal', 'Normal', 'Alerta', 'Alerta', 'Normal', 'Falla', 'Normal',
                      'Normal', 'Normal', 'Normal', 'Normal']
}
df_raw = pd.DataFrame(data)
display(df_raw.head())
print("Filas y columnas iniciales:", df_raw.shape)
# Identificar columnas numéricas
columnas_numericas = df_raw.select_dtypes(
    include="number"
).columns
# Contar NaN explícitos
print("VALORES NaN POR COLUMNA")
display(df_raw[columnas_numericas].isna().sum())
# Contar los valores centinela
print("VALORES -999.0 POR COLUMNA")
display(df_raw[columnas_numericas].eq(-999.0).sum())
# Contar filas duplicadas
print("Filas totalmente duplicadas:", df_raw.duplicated().sum())
# Crear una copia para conservar los datos originales
df_raw = df_raw.copy()
# Reemplazar -999.0 por NaN
df_raw[columnas_numericas] = df_raw[
    columnas_numericas
].replace(-999.0, np.nan)
# Eliminar filas totalmente duplicadas
df_raw = df_raw.drop_duplicates().reset_index(drop=True)
print("DATAFRAME DESPUÉS DE LA LIMPIEZA INICIAL")
display(df_raw)
print("Dimensiones finales:", df_raw.shape)
print("\nValores faltantes por columna:")
display(df_raw.isna().sum())
# ==========================================================================
# TAREA 2: PIPELINE DE LIMPIEZA, FEATURE ENGINEERING E IMPUTACIÓN
# ==========================================================================
from sklearn.impute import KNNImputer
df_clean = df_raw.copy()
# 1. Homogeneizar las zonas
mapa_zonas = {
    "centro_historico": "Centro_Historico",
    "zona_oriente": "Zona_Oriente",
    "zona_sur": "Zona_Sur"
}
df_clean["Zona_Estacion"] = (
    df_clean["Zona_Estacion"]
    .str.strip()
    .str.lower()
    .map(mapa_zonas))
# 2. Convertir Timestamp y obtener Hora_Dia
df_clean["Timestamp"] = pd.to_datetime(
    df_clean["Timestamp"], errors="coerce")
df_clean["Hora_Dia"] = df_clean["Timestamp"].dt.hour
# 3. Detectar valores atípicos mediante IQR
columna = "Concentracion_PM25"
q1 = df_clean[columna].quantile(0.25)
q3 = df_clean[columna].quantile(0.75)
iqr = q3 - q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr
outliers = ((df_clean[columna] < limite_inferior) |
    (df_clean[columna] > limite_superior))
print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)
print("Límite inferior:", limite_inferior)
print("Límite superior:", limite_superior)
print("Cantidad de outliers:", int(outliers.sum()))
# Mostrar los registros identificados antes de reemplazarlos
display(df_clean.loc[outliers, [
    "ID_Registro", "Concentracion_PM25", "Estado_Sensor"]])
# Sustituir los valores atípicos por NaN
df_clean.loc[outliers, columna] = np.nan
# 4. Imputar las tres variables numéricas
columnas_imputar = ["Concentracion_PM25",
    "Nivel_NO2",
    "Indice_Trafico_Vehicular"]
imputer = KNNImputer(n_neighbors=2)
df_clean[columnas_imputar] = imputer.fit_transform(
    df_clean[columnas_imputar])
# 5. Verificar los resultados
print("DATAFRAME LIMPIO E IMPUTADO")
display(df_clean)
print("Valores faltantes por columna:")
display(df_clean.isna().sum())
print("Zonas normalizadas:")
print(df_clean["Zona_Estacion"].unique())

# ==========================================================================
# TAREA 3: EDA Y ANÁLISIS DE CORRELACIÓN
# ==========================================================================
# 1. Estadísticas de forma para Concentracion_PM25
pm25 = df_clean["Concentracion_PM25"]
resultados = pd.Series({
    "Media": pm25.mean(),
    "Mediana": pm25.median(),
    "Desviación estándar": pm25.std(),
    "Skewness (asimetría)": pm25.skew(),
    "Kurtosis (curtosis)": pm25.kurt()})
print("ESTADÍSTICAS DESCRIPTIVAS DE PM2.5")
display(resultados.to_frame(name="Valor").round(4))
# 2. Matrices de correlación Pearson y Spearman
df_numerico = df_clean.select_dtypes(include=np.number)
# Correlación de Pearson
corr_pearson = df_numerico.corr(method="pearson")
print("MATRIZ DE CORRELACIÓN DE PEARSON")
display(corr_pearson.round(3))
# Correlación de Spearman
corr_spearman = df_numerico.corr(method="spearman")
print("MATRIZ DE CORRELACIÓN DE SPEARMAN")
display(corr_spearman.round(3))
# Buscar pares con correlación absoluta >= 0.85
def buscar_correlaciones_fuertes(matriz, umbral=0.85):
    mascara = np.triu(
        np.ones(matriz.shape, dtype=bool), k=1)
    pares = matriz.where(mascara).stack()
    fuertes = pares[pares.abs() >= umbral]
    if fuertes.empty:
        print(f"No hay pares con |r| >= {umbral}.")
    else:
        for (var1, var2), valor in fuertes.items():
            print(f"{var1} y {var2}: {valor:.3f}")
print("PEARSON")
buscar_correlaciones_fuertes(corr_pearson)
print("\nSPEARMAN")

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 6))
sns.heatmap(corr_pearson,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1,
    center=0)
plt.title("Matriz de correlación de Pearson - EcoTránsito CDMX")
plt.tight_layout()
plt.show()