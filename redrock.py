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
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from csv_engine import process_csv

# ================= CONFIG =================
st.set_page_config(page_title="R E D R O C K", layout="wide")
st.title("🟥 R E D R O C K")
st.caption("Data inspection • filtros • métricas • agrupaciones • gráficos")
REDROCK_RED = "#FF0000"

# ================= STATE =================
if "filters" not in st.session_state:
    st.session_state.filters = []

# ================= SIDEBAR =================
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])
if uploaded_file is None:
    st.info("👈 Carga un archivo CSV para comenzar")
    st.stop()

temp_path = "temp_uploaded.csv"
with open(temp_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

# ================= PROCESS =================
with st.spinner("Procesando datos..."):
    preview_df, df_filtered = process_csv(
        temp_path,
        st.session_state.filters
    )

# ================= COLUMN VISIBILITY =================
st.sidebar.subheader("Columnas visibles")
visible_columns = st.sidebar.multiselect(
    "Selecciona columnas",
    df_filtered.columns.tolist(),
    default=df_filtered.columns.tolist()
)
df_display = df_filtered[visible_columns].copy()

# ================= FILTERS =================
st.sidebar.subheader("Filtros")
with st.sidebar.expander("➕ Agregar filtro"):
    f_col = st.selectbox("Columna", df_filtered.columns, key="f_col")
    f_op = st.selectbox(
        "Operador",
        ["=", "!=", ">", "<", ">=", "<=", "contiene"],
        key="f_op"
    )
    f_val = st.text_input("Valor", key="f_val")
    if st.button("Agregar filtro"):
        st.session_state.filters.append((f_col, f_op, f_val))
        st.rerun()

if st.session_state.filters:
    st.sidebar.markdown("### Filtros activos")
    for i, f in enumerate(st.session_state.filters):
        c1, c2 = st.sidebar.columns([4, 1])
        c1.write(f"{i+1}. {f[0]} {f[1]} {f[2]}")
        if c2.button("✕", key=f"del_filter_{i}"):
            st.session_state.filters.pop(i)
            st.rerun()

# ================= ANALYSIS =================
st.sidebar.subheader("Análisis")
group_col = st.sidebar.selectbox(
    "Agrupar por",
    ["— Ninguno —"] + df_filtered.columns.tolist()
)
metric_col = st.sidebar.selectbox(
    "Columna métrica",
    df_filtered.columns.tolist()
)
metric_op_label = st.sidebar.selectbox(
    "Operación",
    ["Conteo", "Suma", "Promedio", "Mínimo", "Máximo"]
)
agg_map = {
    "Conteo": "count",
    "Suma": "sum",
    "Promedio": "mean",
    "Mínimo": "min",
    "Máximo": "max"
}
agg_func = agg_map[metric_op_label]

# ================= CALCULATIONS =================
df_calc = df_filtered.copy()
grouped_df = None
metric_value = None

if group_col != "— Ninguno —":
    grouped_df = (
        df_calc
        .groupby(group_col, observed=True)[metric_col]
        .agg(agg_func)
        .reset_index()
        .rename(columns={metric_col: metric_op_label})
        .sort_values(metric_op_label, ascending=False)
        .reset_index(drop=True)
    )
else:
    s = df_calc[metric_col]
    metric_value = {
        "Conteo": int(s.count()),
        "Suma": float(s.sum()),
        "Promedio": float(s.mean()),
        "Mínimo": float(s.min()),
        "Máximo": float(s.max())
    }[metric_op_label]

# ================= UI =================
st.subheader("Vista previa")
st.dataframe(preview_df.head(10), use_container_width=True)

st.subheader("Resultado")
if grouped_df is not None:
    st.dataframe(grouped_df, use_container_width=True)
else:
    st.metric(metric_op_label, metric_value)

# ================= GRAPH (BROWSER) =================
st.subheader("Gráfico")
if grouped_df is not None and not grouped_df.empty:
    # Opcional: también limitar a Top 15 en la app web (descomenta si lo quieres)
    # df_plot = grouped_df.head(15)
    # titulo = f"{metric_op_label} por {group_col} (Top 15)" if len(grouped_df) > 15 else f"{metric_op_label} por {group_col}"
    fig = px.bar(
        grouped_df,  # o df_plot si activas el límite
        x=group_col,
        y=metric_op_label,
        title=f"{metric_op_label} por {group_col}",
        color_discrete_sequence=[REDROCK_RED]
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    fig = None
    st.info("No hay datos agrupados para el gráfico.")

# ================= TABLE =================
st.subheader("Datos filtrados (columnas visibles)")
st.dataframe(df_display, use_container_width=True)

# ================= AUX FUNCTIONS FOR PDF =================
def get_top_15_df(df, group_col, metric_col):
    """Devuelve solo las 15 filas con mayor valor en la métrica."""
    if df is None or df.empty:
        return df
    return df.sort_values(metric_col, ascending=False).head(15).copy()

def plot_to_png_pdf(df, group_col, metric_op_label):
    """Genera gráfico matplotlib con solo Top 15 y lo guarda como PNG."""
    if df is None or df.empty:
        return None

    df_top = get_top_15_df(df, group_col, metric_op_label)
    total_categorias = len(df)
    titulo = f"{metric_op_label} por {group_col}"
    if total_categorias > 15:
        titulo += f" (Top 15 de {total_categorias} categorías)"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.figure(figsize=(12, 7))
    plt.bar(
        df_top[group_col].astype(str),
        df_top[metric_op_label],
        color=REDROCK_RED
    )
    plt.title(titulo, fontsize=14, pad=20)
    plt.xlabel(group_col, fontsize=12)
    plt.ylabel(metric_op_label, fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout(pad=2.0)
    plt.savefig(tmp.name, dpi=200, bbox_inches='tight')
    plt.close()
    return tmp.name

def df_to_table(data_df, title):
    styles = getSampleStyleSheet()
    elements = [
        Paragraph(title, styles["Heading2"]),
        Spacer(1, 12)
    ]
    data = [data_df.columns.tolist()] + data_df.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.black),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    elements.append(table)
    elements.append(PageBreak())
    return elements

def footer(canvas, doc):
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(
        A4[0] / 2,
        1.5 * cm,
        "Generated by RedRock 2026"
    )

def generate_pdf():
    buffer = BytesIO()
    tmp_images = []
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm
    )
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("R E D R O C K – Reporte de Datos", styles["Title"]))
    elements.append(Spacer(1, 30))

    # Gráfico Top 15
    if grouped_df is not None and not grouped_df.empty:
        img_path = plot_to_png_pdf(grouped_df, group_col, metric_op_label)
        if img_path:
            tmp_images.append(img_path)
            elements.append(Paragraph("Análisis Gráfico – Top 15 categorías", styles["Heading2"]))
            elements.append(Spacer(1, 12))
            elements.append(Image(img_path, width=17 * cm, height=9 * cm))
            elements.append(PageBreak())

    # Tabla Top 15 (recomendado)
    if grouped_df is not None and not grouped_df.empty:
        top15_df = get_top_15_df(grouped_df, group_col, metric_op_label)
        total = len(grouped_df)
        titulo_tabla = f"Resultados Agrupados – Top 15 de {total} categorías" if total > 15 else "Resultados Agrupados"
        elements += df_to_table(top15_df, titulo_tabla)

    # Tabla completa filtrada
    elements += df_to_table(df_display, "Datos Filtrados (columnas visibles)")

    doc.build(elements, onFirstPage=footer, onLaterPages=footer)

    # Limpieza de imágenes temporales
    for img in tmp_images:
        try:
            os.unlink(img)
        except:
            pass

    buffer.seek(0)
    return buffer

# ================= EXPORT =================
st.subheader("Exportar resultados")
if st.button("⬇ Exportar PDF profesional"):
    with st.spinner("Generando PDF..."):
        pdf_buffer = generate_pdf()
    st.download_button(
        "Descargar PDF",
        pdf_buffer,
        "redrock_reporte.pdf",
        mime="application/pdf"
    )

