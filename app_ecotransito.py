# ==========================================================================
# TAREA 4: SCRIPT DE STREAMLIT (app_ecotransito.py)
# ==========================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from pathlib import Path
# ==========================================================
# 1. CONFIGURACIÓN DE PÁGINA
# ==========================================================
st.set_page_config(
    page_title="Sistema EcoTránsito CDMX",
    page_icon="🌎",
    layout="wide"
)

st.title("Sistema EcoTránsito CDMX")
st.subheader("Diagnóstico de calidad del aire y flujo vehicular")

# ==========================================================
# 2. CARGA Y LIMPIEZA DE LOS DATOS
# ==========================================================
@st.cache_data
def cargar_y_limpiar_datos():
    #df = pd.read_csv("cdmx_air_traffic_dirty.csv")
    RUTA_CSV = Path(__file__).parent / "cdmx_air_traffic_dirty.csv"

if not RUTA_CSV.exists():
    raise FileNotFoundError(
        f"No se encontró el archivo: {RUTA_CSV}"
    )

df = pd.read_csv(RUTA_CSV)

    # Sustituir centinelas y eliminar duplicados
    columnas_numericas = df.select_dtypes(include="number").columns
    df[columnas_numericas] = df[columnas_numericas].replace(-999.0, np.nan)
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Homogeneizar las zonas
    mapa_zonas = {
        "centro_historico": "Centro_Historico",
        "zona_oriente": "Zona_Oriente",
        "zona_sur": "Zona_Sur"
    }
    df["Zona_Estacion"] = (
        df["Zona_Estacion"]
        .str.strip()
        .str.lower()
        .map(mapa_zonas)
    )
    
    # Convertir fechas y extraer hora
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Hora_Dia"] = df["Timestamp"].dt.hour
    
    # Detectar valores atípicos de PM2.5 mediante IQR
    columna = "Concentracion_PM25"
    q1 = df[columna].quantile(0.25)
    q3 = df[columna].quantile(0.75)
    iqr = q3 - q1
    inferior = q1 - 1.5 * iqr
    superior = q3 + 1.5 * iqr
    outliers = (df[columna] < inferior) | (df[columna] > superior)
    df.loc[outliers, columna] = np.nan
    
    # Imputar las variables numéricas solicitadas
    columnas_imputar = [
        "Concentracion_PM25",
        "Nivel_NO2",
        "Indice_Trafico_Vehicular"
    ]
    imputador = KNNImputer(n_neighbors=2)
    df[columnas_imputar] = imputador.fit_transform(df[columnas_imputar])
    
    return df

try:
    df = cargar_y_limpiar_datos()
except FileNotFoundError:
    st.error("No se encontró cdmx_air_traffic_dirty.csv. Guarda el archivo CSV en la misma carpeta que app_ecotransito.py.")
    st.stop()

# ==========================================================
# 3. FILTRO DINÁMICO POR ZONA
# ==========================================================
st.sidebar.header("Filtros")
zonas_disponibles = sorted(df["Zona_Estacion"].dropna().unique().tolist())
zonas_seleccionadas = st.sidebar.multiselect(
    "Selecciona las zonas:",
    options=zonas_disponibles,
    default=zonas_disponibles
)

df_filtrado = df[df["Zona_Estacion"].isin(zonas_seleccionadas)].copy()
if df_filtrado.empty:
    st.warning("Selecciona al menos una zona para ver los datos.")
    st.stop()

# ==========================================================
# 4. INDICADORES KPI
# ==========================================================
col1, col2, col3 = st.columns(3)
col1.metric("Total de lecturas activas", len(df_filtrado))
col2.metric("Promedio de PM2.5", f"{df_filtrado['Concentracion_PM25'].mean():.2f} μg/m³")
col3.metric("Índice máximo de tráfico", f"{df_filtrado['Indice_Trafico_Vehicular'].max():.2f}")

st.divider()

# ==========================================================
# 5. PESTAÑAS DE VISUALIZACIÓN
# ==========================================================
tab1, tab2 = st.tabs(["Tráfico y PM2.5", "Correlación"])

# Pestaña 1: gráfico interactivo de dispersión
with tab1:
    st.subheader("Tráfico vehicular frente a PM2.5")
    fig_scatter = px.scatter(
        df_filtrado,
        x="Indice_Trafico_Vehicular",
        y="Concentracion_PM25",
        color="Zona_Estacion",
        hover_data=[
            "ID_Registro",
            "Timestamp",
            "Nivel_NO2",
            "Estado_Sensor"
        ],
        title="Relación entre tráfico y contaminación",
        labels={
            "Indice_Trafico_Vehicular": "Índice de tráfico",
            "Concentracion_PM25": "PM2.5 (μg/m³)",
            "Zona_Estacion": "Zona"
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# Pestaña 2: mapa de calor
with tab2:
    st.subheader("Matriz de correlación")
    columnas_corr = [
        "Concentracion_PM25",
        "Nivel_NO2",
        "Indice_Trafico_Vehicular",
        "Hora_Dia"
    ]
    matriz_corr = df_filtrado[columnas_corr].corr(method="pearson")
    
    fig_heatmap, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        matriz_corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax
    )
    ax.set_title("Correlación entre variables numéricas")
    plt.tight_layout()
    st.pyplot(fig_heatmap)
    plt.close(fig_heatmap)

# ==========================================================
# 6. TABLA DE DATOS
# ==========================================================
with st.expander("Consultar los datos procesados"):
    st.dataframe(df_filtrado, use_container_width=True)

st.caption("EcoTránsito CDMX | Análisis exploratorio de calidad del aire y tráfico")
