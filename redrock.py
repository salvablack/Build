import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from PIL import Image

# Configuración inicial de la página para un look profesional
st.set_page_config(
    page_title="R E D R O C K",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para profesionalismo
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #4CAF50; color: white; border: none; padding: 10px 20px; text-align: center; font-size: 16px; border-radius: 5px;}
    .stButton>button:hover {background-color: #45a049;}
    .stDataFrame {border: 1px solid #ddd; border-radius: 5px; padding: 10px;}
    .stAlert {border-radius: 5px;}
    h1, h2, h3 {color: #333;}
    </style>
""", unsafe_allow_html=True)

# Sidebar para navegación y opciones
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=Logo+Empresa", use_column_width=True)  # Reemplaza con tu logo si tienes
    st.title("Opciones de Análisis")
    st.markdown("---")
    st.info("Sube tu CSV y configura el análisis.")
    
    # Opciones dinámicas (se activan después de subir el archivo)
    if 'df' in st.session_state:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'datetime']).columns.tolist()
        
        st.subheader("Configura Pivot Table")
        valor = st.selectbox("Columna de Valor (numérica)", numeric_cols, index=0 if numeric_cols else None)
        indice = st.selectbox("Filas (Índice)", cat_cols, index=0 if cat_cols else None)
        columna = st.selectbox("Columnas (opcional)", ['Ninguna'] + cat_cols, index=0)
        agg_func = st.selectbox("Función de Agregación", ['sum', 'mean', 'count', 'min', 'max', 'std'], index=0)
        
        st.subheader("Configura Gráfica")
        graph_type = st.selectbox("Tipo de Gráfica", ['Barra', 'Línea', 'Pastel', 'Dispersión', 'Heatmap'], index=0)
        stacked = st.checkbox("Apilado (para barra/línea)", value=False)

# Contenido principal
st.title("🛠️ R E D R O C K")
st.markdown("Sube tu archivo CSV para analizar datos, generar tablas pivot dinámicas y visualizaciones interactivas. Soporta análisis descriptivo, pivots personalizados y exportaciones.")

# Sección de subida de archivo
uploaded_file = st.file_uploader("Selecciona o arrastra tu archivo CSV aquí", type=["csv"], help="Archivos CSV hasta 200MB. Asegúrate de que tenga encabezados.")

if uploaded_file is not None:
    try:
        # Leer el CSV y guardarlo en session_state para persistencia
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success(f"✅ Archivo '{uploaded_file.name}' cargado exitosamente. Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas.")
        
        # Análisis básico automático
        with st.expander("📊 Vista Previa y Análisis Básico", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Vista Previa")
                st.dataframe(df.head(10), use_container_width=True)
            with col2:
                st.subheader("Estadísticas Descriptivas")
                st.dataframe(df.describe(), use_container_width=True)
                
            st.subheader("Información del DataFrame")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        # Procesamiento de fechas si aplica
        date_cols = df.select_dtypes(include=['object']).columns[df.select_dtypes(include=['object']).apply(lambda col: pd.to_datetime(col, errors='coerce').notnull().all())].tolist()
        if date_cols:
            for col in date_cols:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[f'{col}_Mes'] = df[col].dt.month
                df[f'{col}_Año'] = df[col].dt.year
            st.info(f"📅 Columnas de fechas detectadas y procesadas: {', '.join(date_cols)}")
        
        # Generar Pivot Table basado en selecciones de sidebar
        if 'valor' in locals() and valor and indice:
            try:
                pivot_params = {
                    'values': valor,
                    'index': indice,
                    'aggfunc': agg_func,
                    'fill_value': 0
                }
                if columna != 'Ninguna':
                    pivot_params['columns'] = columna
                
                pivot_table = pd.pivot_table(df, **pivot_params)
                
                st.subheader("🔄 Tabla Pivot Generada")
                st.dataframe(pivot_table, use_container_width=True)
                
                # Exportar pivot a CSV
                csv = pivot_table.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Descargar Pivot como CSV",
                    data=csv,
                    file_name="pivot_table.csv",
                    mime="text/csv"
                )
                
                # Generar Gráfica
                st.subheader("📈 Gráfica Basada en la Pivot")
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.set(style="whitegrid")
                
                if graph_type == 'Barra':
                    if columna != 'Ninguna':
                        pivot_table.plot(kind='bar', stacked=stacked, ax=ax)
                    else:
                        sns.barplot(data=pivot_table.reset_index(), x=indice, y=valor, ax=ax)
                elif graph_type == 'Línea':
                    pivot_table.plot(kind='line', marker='o', ax=ax)
                elif graph_type == 'Pastel':
                    if columna == 'Ninguna':
                        pivot_table.plot(kind='pie', y=valor, ax=ax, autopct='%1.1f%%')
                    else:
                        st.warning("Pastel no soporta columnas múltiples.")
                elif graph_type == 'Dispersión':
                    if len(pivot_table.columns) >= 2:
                        sns.scatterplot(data=pivot_table.reset_index(), x=pivot_table.columns[0], y=pivot_table.columns[1], ax=ax)
                    else:
                        st.warning("Dispersión requiere al menos dos columnas numéricas.")
                elif graph_type == 'Heatmap':
                    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', ax=ax)
                
                ax.set_title(f"{graph_type} de {valor} por {indice}" + (f" y {columna}" if columna != 'Ninguna' else ""))
                ax.set_xlabel(indice)
                ax.set_ylabel(valor)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                st.pyplot(fig)
                
                # Exportar gráfica como PNG
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                st.download_button(
                    label="🖼️ Descargar Gráfica como PNG",
                    data=buf,
                    file_name="grafica.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"❌ Error al generar pivot o gráfica: {str(e)}. Verifica las selecciones.")
        
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {str(e)}. Asegúrate de que sea un CSV válido.")
else:
    st.info("👆 Sube un archivo CSV para comenzar el análisis.")

# Pie de página
st.markdown("---")
st.caption("Desarrollado con Streamlit | Versión 1.0 | © 2025 Tu Empresa")