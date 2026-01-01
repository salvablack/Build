import pandas as pd
import re

# ---------- Limpieza numérica ----------
def clean_numeric_series(series):
    return (
        series.astype(str)
        .str.replace(r"[^\d\-,.]", "", regex=True)
        .str.replace(",", "", regex=False)
        .replace("", pd.NA)
        .astype(float)
    )

# ---------- Motor principal ----------
def process_csv(
    path,
    filters=None,
    selected_columns=None,
    chunksize=100_000
):
    preview = None
    result_chunks = []

    for chunk in pd.read_csv(
        path,
        sep=None,
        engine="python",
        encoding="latin-1",
        chunksize=chunksize
    ):
        # Vista previa (solo una vez)
        if preview is None:
            preview = chunk.head(10)

        # Limpieza automática de columnas numéricas
        for col in chunk.columns:
            if chunk[col].astype(str).str.contains(r"\d").any():
                try:
                    chunk[col] = clean_numeric_series(chunk[col])
                except:
                    pass

        # Aplicar filtros
        if filters:
            for col, op, val in filters:
                if col not in chunk.columns:
                    continue

                if op == "=":
                    chunk = chunk[chunk[col] == val]
                elif op == "!=":
                    chunk = chunk[chunk[col] != val]
                elif op == ">":
                    chunk = chunk[chunk[col] > val]
                elif op == "<":
                    chunk = chunk[chunk[col] < val]
                elif op == ">=":
                    chunk = chunk[chunk[col] >= val]
                elif op == "<=":
                    chunk = chunk[chunk[col] <= val]
                elif op == "contiene":
                    chunk = chunk[
                        chunk[col]
                        .astype(str)
                        .str.contains(val, case=False, na=False)
                    ]

        # Selección de columnas
        if selected_columns:
            chunk = chunk[selected_columns]

        if not chunk.empty:
            result_chunks.append(chunk)

    if not result_chunks:
        return preview, pd.DataFrame()

    final_df = pd.concat(result_chunks, ignore_index=True)
    return preview, final_df

# ---------- Exportación profesional ----------
def export_data(df, path, fmt="csv"):
    """
    fmt: csv | parquet | feather
    """
    if fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8")

    elif fmt == "parquet":
        df.to_parquet(path, index=False)

    elif fmt == "feather":
        df.to_feather(path)

    else:
        raise ValueError(f"Formato no soportado: {fmt}")
