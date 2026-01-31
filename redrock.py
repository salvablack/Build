import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
import tempfile
import os
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle,
    Spacer, PageBreak, Image
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4, landscape   # ← agregado landscape
from reportlab.lib.units import cm

# ================= CONFIG =================
st.set_page_config(page_title="R E D R O C K Technologies", layout="wide")
st.title("🟥 R E D R O C K")
st.caption("Data inspection • filtros • métricas • agrupaciones • gráficos + Mucho más...")
REDROCK_RED = "#FF0000"

# ================= STATE =================
if "filters" not in st.session_state:
    st.session_state.filters = []

# ================= SIDEBAR =================
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader(
    "Cargar archivo (CSV o Excel)",
    type=["csv", "xlsx", "xls"]
)

# Detección simple de cambio de archivo para limpiar caché
if uploaded_file is not None:
    current_file_id = uploaded_file.name + "_" + str(uploaded_file.size)
    if "last_file_id" not in st.session_state:
        st.session_state.last_file_id = None
    
    if current_file_id != st.session_state.last_file_id:
        st.cache_data.clear()
        st.session_state.filters = []  # limpiamos también los filtros para evitar confusiones
        st.session_state.last_file_id = current_file_id

if uploaded_file is None:
    st.info("👈 Carga un archivo CSV o Excel para comenzar")
    st.stop()

# Guardar archivo subido temporalmente
file_extension = os.path.splitext(uploaded_file.name)[1].lower()
temp_path = f"temp_uploaded{file_extension}"
with open(temp_path, "wb") as f:
    f.write(uploaded_file.getbuffer())
st.sidebar.success(f"Archivo cargado: {uploaded_file.name} ({file_extension[1:].upper()})")

# Leer el archivo según su formato
@st.cache_data(show_spinner="Leyendo archivo...")
def load_data(file_path, ext):
    if ext == ".csv":
        return pd.read_csv(file_path, low_memory=False, encoding_errors='replace')
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path, engine='openpyxl')
    else:
        st.error("Formato de archivo no soportado.")
        st.stop()

with st.spinner("Cargando datos..."):
    df = load_data(temp_path, file_extension)

# ================= FILTROS =================
def apply_filters(df, filters):
    df_filtered = df.copy()
    for col, op, val in filters:
        if op == "=":
            df_filtered = df_filtered[df_filtered[col] == val]
        elif op == "!=":
            df_filtered = df_filtered[df_filtered[col] != val]
        elif op == ">":
            df_filtered = df_filtered[df_filtered[col] > float(val)]
        elif op == "<":
            df_filtered = df_filtered[df_filtered[col] < float(val)]
        elif op == ">=":
            df_filtered = df_filtered[df_filtered[col] >= float(val)]
        elif op == "<=":
            df_filtered = df_filtered[df_filtered[col] <= float(val)]
        elif op == "contiene":
            df_filtered = df_filtered[df_filtered[col].astype(str).str.contains(val, case=False, na=False)]
    return df_filtered

with st.spinner("Aplicando filtros..."):
    df_filtered = apply_filters(df, st.session_state.filters)
    preview_df = df_filtered.copy()

# ================= COLUMN VISIBILITY =================
st.sidebar.subheader("Columnas visibles")
visible_columns = st.sidebar.multiselect(
    "Selecciona columnas",
    options=df_filtered.columns.tolist(),
    default=df_filtered.columns.tolist(),
    key="visible_cols"
)
df_display = df_filtered[visible_columns].copy()

# ================= FILTROS UI =================
st.sidebar.subheader("Filtros")
with st.sidebar.expander("➕ Agregar filtro"):
    f_col = st.selectbox("Columna", df.columns, key="f_col_new")
    f_op = st.selectbox(
        "Operador",
        ["=", "!=", ">", "<", ">=", "<=", "contiene"],
        key="f_op_new"
    )
    f_val = st.text_input("Valor", key="f_val_new")
    if st.button("Agregar filtro"):
        if f_col and f_val.strip():
            st.session_state.filters.append((f_col, f_op, f_val.strip()))
            st.rerun()
        else:
            st.sidebar.warning("Completa columna y valor")

if st.session_state.filters:
    st.sidebar.markdown("### Filtros activos")
    for i, (col, op, val) in enumerate(st.session_state.filters):
        c1, c2 = st.sidebar.columns([5, 1])
        c1.write(f"{i+1}. **{col}** {op} *{val}*")
        if c2.button("✕", key=f"del_f_{i}"):
            st.session_state.filters.pop(i)
            st.rerun()

# ================= ANÁLISIS =================
st.sidebar.subheader("Análisis")
group_col = st.sidebar.selectbox(
    "Agrupar por",
    ["— Ninguno —"] + df_filtered.columns.tolist(),
    key="group_col"
)
metric_col = st.sidebar.selectbox(
    "Columna métrica",
    df_filtered.columns.tolist(),
    key="metric_col"
)
metric_op_label = st.sidebar.selectbox(
    "Operación",
    ["Conteo", "Suma", "Promedio", "Mínimo", "Máximo"],
    key="metric_op"
)
agg_map = {
    "Conteo": "count",
    "Suma": "sum",
    "Promedio": "mean",
    "Mínimo": "min",
    "Máximo": "max"
}
agg_func = agg_map[metric_op_label]

# Cálculos
df_calc = df_filtered.copy()
grouped_df = None
metric_value = None
if group_col != "— Ninguno —":
    grouped_df = (
        df_calc.groupby(group_col, observed=True)[metric_col]
        .agg(agg_func)
        .reset_index()
        .rename(columns={metric_col: metric_op_label})
        .sort_values(metric_op_label, ascending=False)
        .reset_index(drop=True)
    )
else:
    s = df_calc[metric_col]
    if metric_op_label == "Conteo":
        metric_value = int(s.count())
    elif metric_op_label == "Suma":
        metric_value = float(s.sum()) if not s.empty else 0
    elif metric_op_label == "Promedio":
        metric_value = float(s.mean()) if not s.empty else None
    elif metric_op_label == "Mínimo":
        metric_value = float(s.min()) if not s.empty else None
    elif metric_op_label == "Máximo":
        metric_value = float(s.max()) if not s.empty else None

# ================= VISTA PRINCIPAL =================
st.subheader("Vista previa (primeras 10 filas)")
st.dataframe(preview_df.head(10), use_container_width=True)

st.subheader("Resultado")
if grouped_df is not None and not grouped_df.empty:
    st.dataframe(grouped_df, use_container_width=True)
elif metric_value is not None:
    st.metric(metric_op_label, f"{metric_value:,.2f}" if isinstance(metric_value, float) else metric_value)
else:
    st.info("Selecciona una operación y columna para ver resultados.")

# ================= GRÁFICO =================
st.subheader("Gráfico")
if grouped_df is not None and not grouped_df.empty:
    fig = px.bar(
        grouped_df,
        x=group_col,
        y=metric_op_label,
        title=f"{metric_op_label} por {group_col}",
        color_discrete_sequence=[REDROCK_RED]
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No hay datos agrupados para mostrar gráfico.")

# ================= TABLA FINAL =================
st.subheader("Datos filtrados (columnas seleccionadas)")
st.dataframe(df_display, use_container_width=True)

# ================= FUNCIONES PDF =================
def get_top_15_df(df, group_col, metric_col):
    if df is None or df.empty:
        return None
    return df.sort_values(metric_col, ascending=False).head(15).copy()

def plot_to_png_pdf(df, group_col, metric_op_label):
    if df is None or df.empty:
        return None
    df_top = get_top_15_df(df, group_col, metric_op_label)
    if df_top is None:
        return None
   
    total = len(df)
    titulo = f"{metric_op_label} por {group_col}"
    if total > 15:
        titulo += f" (Top 15 de {total})"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.figure(figsize=(12, 7))
    plt.bar(df_top[group_col].astype(str), df_top[metric_op_label], color=REDROCK_RED)
    plt.title(titulo, fontsize=14, pad=20)
    plt.xlabel(group_col, fontsize=12)
    plt.ylabel(metric_op_label, fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    plt.savefig(tmp.name, dpi=200, bbox_inches='tight')
    plt.close()
    return tmp.name

def df_to_table(data_df, title):
    styles = getSampleStyleSheet()
    elements = [Paragraph(title, styles["Heading2"]), Spacer(1, 12)]
    data = [data_df.columns.tolist()] + data_df.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.black),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ]))
    elements.append(table)
    elements.append(PageBreak())
    return elements

def footer(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(A4[0]/2, 1.5*cm, "Generated by RedRock • 2026")

def generate_pdf():
    buffer = BytesIO()
    tmp_images = []
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=landscape(A4),              # ← Cambio principal: horizontal
        topMargin=2*cm, bottomMargin=2*cm,
        leftMargin=2*cm, rightMargin=2*cm
    )
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("R E D R O C K – Reporte de Datos", styles["Title"]))
    elements.append(Spacer(1, 30))
    if grouped_df is not None and not grouped_df.empty:
        img_path = plot_to_png_pdf(grouped_df, group_col, metric_op_label)
        if img_path:
            tmp_images.append(img_path)
            elements.append(Paragraph("Análisis Gráfico – Top 15", styles["Heading2"]))
            elements.append(Spacer(1, 12))
            elements.append(Image(img_path, width=22*cm, height=10*cm))  # un poco más ancho
            elements.append(PageBreak())
    if grouped_df is not None and not grouped_df.empty:
        top15 = get_top_15_df(grouped_df, group_col, metric_op_label)
        total = len(grouped_df)
        titulo = f"Resultados Agrupados – Top 15 de {total}" if total > 15 else "Resultados Agrupados"
        elements += df_to_table(top15, titulo)
    elements += df_to_table(df_display, "Datos Filtrados (columnas visibles)")
    doc.build(elements, onFirstPage=footer, onLaterPages=footer)
    for img in tmp_images:
        try:
            os.unlink(img)
        except:
            pass
    buffer.seek(0)
    return buffer

# ================= EXPORTAR =================
st.subheader("Exportar")
if st.button("⬇ Generar PDF profesional"):
    with st.spinner("Creando PDF..."):
        pdf_buffer = generate_pdf()
    st.download_button(
        label="Descargar PDF",
        data=pdf_buffer,
        file_name="redrock_reporte.pdf",
        mime="application/pdf"
    )

# Pie de página
st.markdown("---")
st.caption("Desarrollado con Streamlit | Versión 2026 | © Salva Rosales")










