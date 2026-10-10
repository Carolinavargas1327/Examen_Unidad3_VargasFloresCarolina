# ==========================================================================
# TAREA 4: SCRIPT DE STREAMLIT (app_ecotransito.py)
# ==========================================================================
# ESCRIBE TU CÓDIGO AQUÍ:
import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
# 1. Configuración de página
st.set_page_config( page_title="Sistema EcoTránsito CDMX",
    page_icon="🌎",
    layout="wide")
st.title("Sistema EcoTránsito CDMX")
st.subheader("Diagnóstico de calidad del aire y flujo vehicular")
# 2. Carga de datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("cdmx_air_traffic_clean.csv",
        parse_dates=["Timestamp"])
try:
    df = cargar_datos()
except FileNotFoundError:
    st.error("No se encontró el archivo cdmx_air_traffic_clean.csv. "
        "Guarda el CSV en la misma carpeta que app_ecotransito.py.")
    st.stop()
# 3. Carga y filtrado de datos con barra lateral
st.sidebar.header("Filtrados")
zonas = sorted(df["Zona_Estacion"].dropna().unique())
zonas_seleccionadas = st.sidebar.multiselect( "Selecciona las zonas:",
    options=zonas,
    default=zonas)
df_filtrado = df[ df["Zona_Estacion"].isin(zonas_seleccionadas)].copy()
if df_filtrado.empty:
    st.warning("Selecciona al menos una zona.")
    st.stop()
# 4. Métricas principales (KPIs)
col1, col2, col3 = st.columns(3)
col1.metric( "Total de lecturas activas",
    len(df_filtrado))
col2.metric("Promedio de PM2.5", f"{df_filtrado['Concentracion_PM25'].mean():.2f} μg/m³")
col3.metric("Índice máximo de tráfico", f"{df_filtrado['Indice_Trafico_Vehicular'].max():.2f}")
st.divider()
# 5. Pestañas interactivas de visualización
tab1, tab2 = st.tabs(["Tráfico y PM2.5", "Correlación"])
with tab1:
    st.subheader("Tráfico vehicular frente a PM2.5")
    fig = px.scatter( df_filtrado,
        x="Indice_Trafico_Vehicular",
        y="Concentracion_PM25",
        color="Zona_Estacion",
        hover_data=["ID_Registro", "Nivel_NO2"],
        labels={
            "Indice_Trafico_Vehicular": "Índice de tráfico",
            "Concentracion_PM25": "PM2.5 (μg/m³)",
            "Zona_Estacion": "Zona"
        },
        title="Relación entre tráfico y contaminación")
    st.plotly_chart(fig, use_container_width=True)
with tab2:
    st.subheader("Matriz de correlación")
    columnas = ["Concentracion_PM25",
        "Nivel_NO2",
        "Indice_Trafico_Vehicular",
        "Hora_Dia"]
    matriz = df_filtrado[columnas].corr()
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matriz,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax)
    ax.set_title("Correlación entre variables")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
# 6. Tabla de datos
with st.expander("Ver datos procesados"):
    st.dataframe(df_filtrado, use_container_width=True)
# 7. Pie de página
st.caption("Proyecto académico: EcoTránsito CDMX")

# ==========================================================================