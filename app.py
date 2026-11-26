import csv
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from textlab.preparacion import preparar_dataset

try:
    from textlab.datos import perfilar_dataset
except ModuleNotFoundError as error:
    if error.name not in {"textlab", "textlab.datos"}:
        raise
    perfilar_dataset = None


SAMPLE = """texto,etiqueta
La empresa aumentó sus ganancias,Positivo
El banco mejoró su previsión,Positivo
La compañía ganó nuevos clientes,Positivo
La empresa reportó pérdidas,Negativo
Las ventas bajaron este mes,Negativo
El banco redujo su previsión,Negativo
"""

STYLE = """
<style>
[data-testid="stMainBlockContainer"] {max-width: 1200px; padding-top: 3rem;}
.textlab-header {
    padding: clamp(1.35rem, 4vw, 2.7rem);
    margin-bottom: 1.25rem;
    border-radius: 1.45rem;
    border: 1px solid #ddd8f2;
    background: radial-gradient(ellipse at 96% 12%, #ded3fc 0%, transparent 61%),
                linear-gradient(120deg, #e5efff, #f5f2fb 65%);
}
.textlab-header h1 {
    color: #211d42; margin: 0; padding: 0; font-size: clamp(2.5rem, 6vw, 4.25rem);
    font-weight: 750; letter-spacing: -.065em; line-height: 1.12;
}
.textlab-header p {color: #4b4969; margin: .65rem 0 0; font-size: 1.04rem;}
[data-testid="stMetric"] {
    border: 1px solid #dedbed; border-radius: .85rem; padding: .85rem 1rem;
    background: white;
}
</style>
"""


def read_csv(payload: bytes) -> pd.DataFrame:
    try:
        decoded = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = payload.decode("cp1252")

    try:
        delimiter = csv.Sniffer().sniff(decoded[:8192], delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","

    return pd.read_csv(StringIO(decoded), sep=delimiter)


def preview_distribution(labels: pd.Series) -> dict:
    valid = labels.dropna()
    valid = valid[valid.astype(str).str.strip().ne("")]
    return valid.value_counts(sort=False).to_dict()


def show_distribution(distribution: dict) -> None:
    st.subheader("Distribución por clase")
    if not distribution:
        st.info("La columna seleccionada no contiene etiquetas válidas.")
        return

    labels = [str(label) for label in distribution]
    counts = list(distribution.values())
    chart = go.Figure(
        go.Bar(
            y=labels,
            x=counts,
            orientation="h",
            marker_color="#6253C7",
            text=counts,
            textposition="auto",
            hovertemplate="%{y}: %{x} textos<extra></extra>",
        )
    )
    chart.update_layout(
        height=max(230, min(650, 130 + 35 * len(labels))),
        margin=dict(l=8, r=24, t=12, b=12),
        xaxis=dict(title="Textos", rangemode="tozero", dtick=1 if max(counts) < 10 else None),
        yaxis=dict(type="category", autorange="reversed", automargin=True),
    )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})


def show_training_settings(data: pd.DataFrame, text_column: str, label_column: str) -> None:
    _, summary = preparar_dataset(data, text_column, label_column)

    st.divider()
    st.subheader("Entrenamiento")
    st.write("Regresión logística · TF-IDF")
    st.metric("Filas utilizables", summary["filas_utilizables"])
    st.caption(
        f"Filas excluidas · Sin datos ({summary['filas_sin_datos']}), "
        f"duplicadas ({summary['filas_duplicadas']}) y con etiquetas contradictorias ({summary['filas_en_conflicto']})."
    )
    if summary["filas_utilizables"] == 0:
        st.warning("No quedaron filas utilizables. Revise los datos o elija otras columnas.")
        return

    feature_column, test_column, validation_column = st.columns(3)
    feature_column.selectbox(
        "Grupos de palabras",
        ("Unigramas", "Bigramas", "Trigramas"),
        help="Elija palabras individuales, parejas o grupos de tres palabras consecutivas.",
    )
    test_percent = test_column.slider(
        "Datos reservados para prueba", 10, 50, 20, format="%d %%"
    )
    test_column.caption(f"{100 - test_percent} % para entrenar y {test_percent} % para probar.")
    validation_column.selectbox(
        "Validación cruzada",
        ("Desactivada", "3 folds", "5 folds", "10 folds"),
        help="Evalúa el modelo varias veces con distintas divisiones de los datos de entrenamiento.",
    )

    st.button("Entrenar", type="primary", disabled=True)
    st.caption("El botón se habilitará cuando se conecte el módulo de entrenamiento.")


def main() -> None:
    st.set_page_config(page_title="TextLab · Conjunto de datos", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    with st.sidebar:
        st.title("TextLab")
        st.markdown("**Conjunto de datos**")
        st.caption("Cargue y revise los textos etiquetados para su experimento.")

    st.markdown(
        '<header class="textlab-header"><h1>Conjunto de datos</h1>'
        '<p>Cargue un CSV etiquetado y revise las clases antes de entrenar.</p></header>',
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Archivo CSV", type=["csv"], help="Use un archivo con encabezados en la primera fila."
    )
    use_sample = False
    if uploaded is None:
        use_sample = st.checkbox("Probar con un CSV de ejemplo")
        if not use_sample:
            return

    try:
        data = read_csv(SAMPLE.encode("utf-8") if use_sample else uploaded.getvalue())
    except (UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        st.error("No se pudo leer el CSV. Revise que tenga encabezados y un separador consistente.")
        return

    st.caption(f"Archivo: {'ejemplo sintético' if use_sample else uploaded.name}")
    if data.empty or len(data.columns) < 2:
        st.warning("El CSV debe tener datos y al menos dos columnas: texto y etiqueta.")
        return

    st.subheader("Vista previa")
    st.dataframe(data.head(10), width="stretch", hide_index=True)
    text_field, label_field = st.columns(2)
    text_column = text_field.selectbox("Columna de texto", data.columns)
    label_column = label_field.selectbox("Columna de etiqueta", data.columns, index=1)
    if text_column == label_column:
        st.warning("Seleccione una columna distinta para el texto y otra para la etiqueta.")
        return

    if perfilar_dataset is None:
        st.metric("Filas", len(data))
        show_distribution(preview_distribution(data[label_column]))
        show_training_settings(data, text_column, label_column)
        return

    try:
        profile = perfilar_dataset(data, text_column, label_column)
    except ValueError as error:
        st.error(str(error))
        return

    for column, (label, key) in zip(
        st.columns(4),
        (
            ("Filas", "total_filas"),
            ("Textos faltantes", "textos_faltantes"),
            ("Etiquetas faltantes", "etiquetas_faltantes"),
            ("Filas duplicadas", "filas_duplicadas"),
        ),
    ):
        column.metric(label, profile[key])
    show_distribution(profile["distribucion_clases"])
    show_training_settings(data, text_column, label_column)


if __name__ == "__main__":
    main()
