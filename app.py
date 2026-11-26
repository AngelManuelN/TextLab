from hashlib import sha256
import csv
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

from textlab.entrenamiento import entrenar_modelo
from textlab.preparacion import preparar_dataset
from textlab.validacion_entrenamiento import validar_configuracion_entrenamiento

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
[data-testid="stMainBlockContainer"] {max-width: 1240px; padding-top: 3.2rem; container-type: inline-size;}
[data-testid="stSidebar"] {border-right: 1px solid #e0e1ed;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a {
    display: inline-block; color: #443781; padding: .18rem 0; text-decoration: none;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] a:hover {text-decoration: underline;}
.textlab-header {
    padding: clamp(1.5rem, 5vw, 3.4rem);
    margin-bottom: 1.6rem;
    border-radius: 1.55rem;
    border: 1px solid #d6ddff;
    background: radial-gradient(ellipse at 89% 0%, #dfccff 0%, transparent 54%),
                radial-gradient(ellipse at 14% 125%, #bbddff 0%, transparent 66%),
                linear-gradient(115deg, #e9f0ff, #f1edfa);
    box-shadow: 0 10px 32px #21264f0d;
}
.textlab-header .textlab-brand {
    display: inline-block; padding: .28rem .65rem; margin: 0 0 .65rem;
    border: 1px solid #4c428233; border-radius: .48rem;
    background: #ffffff8c; color: #3c356d;
    font: 600 .8rem ui-monospace, monospace;
}
.textlab-header h1 {
    color: #211d42; margin: 0; padding: 0; font-size: clamp(2.65rem, 7vw, 5rem);
    font-weight: 750; letter-spacing: -.065em; line-height: 1.12;
}
.textlab-header p {color: #413b61; margin: .7rem 0 0; font-size: 1.06rem; max-width: 38rem;}
[data-testid="stMetric"] {
    border: 1px solid #e0def0; border-radius: .9rem; padding: .9rem 1rem;
    background: linear-gradient(165deg, white, #f9f8ff);
    font-variant-numeric: tabular-nums;
}
[data-testid="stFileUploaderDropzone"] {border: 1px dashed #a5a0cc; border-radius: .85rem;}
[data-testid="stBaseButton-primary"]:focus-visible {
    outline: 2px solid #4031a5; outline-offset: 3px;
}
@media (prefers-reduced-motion: no-preference) {
    [data-testid="stBaseButton-primary"]:not(:disabled) {
        transition: transform 140ms cubic-bezier(.2,.65,.3,1);
    }
    [data-testid="stBaseButton-primary"]:not(:disabled):active {transform: scale(.97);}
}
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] {padding-top: 2.5rem;}
    .textlab-header {border-radius: 1.1rem;}
}
@container (max-width: 650px) {
    [data-testid="stHorizontalBlock"] {
        display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {width: 100%; min-width: 0;}
}
@container (max-width: 350px) {
    [data-testid="stHorizontalBlock"] {grid-template-columns: minmax(0, 1fr);}
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
    total = sum(counts)
    textos = [f"{c} ({c / total * 100:.0f} %)" for c in counts]
    chart = go.Figure(
        go.Bar(
            y=labels,
            x=counts,
            orientation="h",
            marker_color="#5647BD",
            text=textos,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{text}<extra></extra>",
        )
    )
    chart.update_layout(
        height=max(230, min(650, 130 + 35 * len(labels))),
        margin=dict(l=8, r=70, t=12, b=12),
        bargap=0.42,
        xaxis=dict(title="Textos", rangemode="tozero", gridcolor="#e7e5f2", dtick=1 if max(counts) < 10 else None),
        yaxis=dict(type="category", autorange="reversed", automargin=True),
    )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})


def show_training_settings(
    data: pd.DataFrame, text_column: str, label_column: str, dataset_id: str, dataset_name: str
) -> dict | None:
    clean_data, summary = preparar_dataset(data, text_column, label_column)

    st.divider()
    st.subheader("Entrenamiento")
    st.write("Regresión logística")
    st.metric("Filas utilizables", summary["filas_utilizables"])
    st.caption(
        f"Filas excluidas · Sin datos ({summary['filas_sin_datos']}), "
        f"duplicadas ({summary['filas_duplicadas']}) y con etiquetas contradictorias ({summary['filas_en_conflicto']})."
    )
    if summary["filas_utilizables"] == 0:
        st.warning("No quedaron filas utilizables. Revise los datos o elija otras columnas.")
        return

    feature_column, representation_column, test_column, validation_column = st.columns(4)
    features = feature_column.selectbox(
        "Grupos de palabras",
        ("Unigramas", "Bigramas", "Trigramas"),
        help="Elija palabras individuales, parejas o grupos de tres palabras consecutivas.",
    )
    representation = representation_column.selectbox("Representación", ("TF-IDF", "Frecuencia", "Binaria"))
    test_percent = test_column.slider(
        "Datos reservados para prueba", 10, 50, 20, format="%d %%"
    )
    test_column.caption(f"{100 - test_percent} % para entrenar y {test_percent} % para probar.")
    folds = validation_column.selectbox(
        "Validación cruzada",
        (None, 3, 5, 10),
        format_func=lambda value: "Desactivada" if value is None else f"{value} folds",
        help="Evalúa el modelo varias veces con distintas divisiones de los datos de entrenamiento.",
    )

    proportion = test_percent / 100
    warnings = validar_configuracion_entrenamiento(clean_data[label_column], proportion, folds)
    for message in warnings:
        st.warning(message)

    settings = (dataset_id, text_column, label_column, proportion, features, representation, folds)
    experiments = st.session_state.setdefault("experimentos", [])
    if st.button("Entrenar", type="primary", disabled=bool(warnings)):
        try:
            size = {"Unigramas": 1, "Bigramas": 2, "Trigramas": 3}[features]
            with st.spinner("Entrenando la regresión logística…"):
                result = entrenar_modelo(
                    clean_data[text_column].astype(str).tolist(),
                    clean_data[label_column].tolist(),
                    proporcion_prueba=proportion,
                    ngram_range=(size, size),
                    representacion={"TF-IDF": "tfidf", "Frecuencia": "frecuencia", "Binaria": "binaria"}[representation],
                    folds=folds,
                )
        except ValueError as error:
            st.error(f"No se pudo entrenar con estos datos. {error}")
            return None
        metrics = calculate_metrics(result["etiquetas_reales"], result["etiquetas_predichas"])
        experiments.append(
            {
                "settings": settings,
                "result": result,
                "summary": {
                    "Experimento": len(experiments) + 1,
                    "Archivo": dataset_name,
                    "Columna de texto": text_column,
                    "Columna de etiqueta": label_column,
                    "Clasificador": "Regresión logística",
                    "Características": features,
                    "Representación": representation,
                    "Prueba (%)": test_percent,
                    "Validación cruzada": f"{folds} folds" if folds else "Desactivada",
                    "Textos de prueba": len(result["etiquetas_reales"]),
                    **{name: value * 100 for name, value in metrics.items()},
                },
            }
        )

    for experiment in reversed(experiments):
        if experiment["settings"] == settings:
            return experiment["result"]
    return None


def calculate_metrics(actual, predicted) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="macro", zero_division=0
    )
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precisión macro": precision,
        "Recall macro": recall,
        "F1 macro": f1,
    }


def show_experiment_history() -> None:
    st.divider()
    st.subheader("Historial de experimentos")
    experiments = st.session_state.get("experimentos", [])
    if not experiments:
        st.info("Entrena un modelo para agregarlo al historial de esta sesión.")
        return

    st.caption("Los experimentos se guardan durante esta sesión. Para comparar métricas, usa el mismo archivo y la misma división de prueba.")
    st.dataframe(
        [experiment["summary"] for experiment in experiments],
        hide_index=True,
        width="stretch",
        column_config={
            name: st.column_config.NumberColumn(f"{name} (%)", format="%.1f")
            for name in ("Accuracy", "Precisión macro", "Recall macro", "F1 macro")
        },
        column_order=(
            "Experimento", "Archivo", "Clasificador", "Características", "Representación",
            "Prueba (%)", "Accuracy", "Precisión macro", "Recall macro", "F1 macro",
            "Textos de prueba", "Validación cruzada", "Columna de texto", "Columna de etiqueta",
        ),
    )


def show_results(result: dict | None) -> None:
    st.divider()
    st.subheader("Resultados")
    if result is None:
        st.info("Ejecuta un experimento para ver los resultados.")
        return

    actual = list(result["etiquetas_reales"])
    predicted = list(result["etiquetas_predichas"])
    if not actual or len(actual) != len(predicted):
        st.warning("Se necesita una predicción por cada etiqueta de prueba para calcular las métricas.")
        return

    classes = unique_labels(actual, predicted)
    st.caption(f"Evaluación con {len(actual)} textos de prueba. El promedio macro da el mismo peso a cada clase.")
    for column, (name, value) in zip(st.columns(4), calculate_metrics(actual, predicted).items()):
        column.metric(name, f"{value:.1%}")

    st.markdown("**Matriz de confusión**")
    st.caption("La diagonal muestra los aciertos. Las demás celdas muestran qué clases se confundieron.")
    labels = [str(label) for label in classes]
    matrix = confusion_matrix(actual, predicted, labels=classes)
    chart = go.Figure(
        go.Heatmap(
            z=matrix.tolist(),
            x=labels,
            y=labels,
            colorscale=[[0, "#f2effb"], [1, "#6253C7"]],
            texttemplate="%{z}",
            hovertemplate="Real: %{y}<br>Predicha: %{x}<br>Textos: %{z}<extra></extra>",
            colorbar=dict(title="Textos"),
        )
    )
    chart.update_layout(
        margin=dict(l=8, r=12, t=12, b=12),
        xaxis=dict(title="Clase predicha", type="category", automargin=True),
        yaxis=dict(title="Clase real", type="category", autorange="reversed", automargin=True),
    )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

    st.markdown("**Desempeño por clase**")
    precision, recall, f1, support = precision_recall_fscore_support(
        actual, predicted, labels=classes, zero_division=0
    )
    per_class = go.Figure()
    for name, values, color in (
        ("Precisión", precision, "#6552c7"),
        ("Recall", recall, "#2776a7"),
        ("F1", f1, "#9372b8"),
    ):
        per_class.add_bar(
            x=labels, y=values.tolist(), name=name, marker_color=color,
            customdata=support.tolist(),
            hovertemplate="%{x}<br>%{y:.1%}<br>Textos de prueba: %{customdata}<extra>%{fullData.name}</extra>",
        )
    per_class.update_layout(
        barmode="group", height=300, margin=dict(l=8, r=12, t=12, b=12),
        legend=dict(orientation="h", y=1.12),
        yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#e7e5f2"),
        xaxis=dict(type="category", automargin=True),
    )
    st.plotly_chart(per_class, width="stretch", config={"displayModeBar": False})

    validation = result.get("validacion_cruzada")
    if validation:
        st.markdown("**Validación cruzada**")
        st.caption(f"{validation['folds']} folds sobre la parte de entrenamiento. La prueba final se evaluó por separado.")
        columns = st.columns(2)
        folds_chart = go.Figure()
        for column, (key, name, color) in zip(
            columns, (("accuracy", "Accuracy", "#6552c7"), ("f1_macro", "F1 macro", "#2776a7"))
        ):
            metric = validation[key]
            column.metric(f"{name} promedio", f"{metric['promedio']:.1%}")
            column.caption(f"Desviación estándar de {metric['desviacion'] * 100:.1f} puntos porcentuales.")
            folds_chart.add_scatter(
                x=list(range(1, validation["folds"] + 1)), y=metric["por_fold"],
                name=name, mode="lines+markers", line=dict(color=color),
                hovertemplate="Fold %{x}<br>%{y:.1%}<extra>%{fullData.name}</extra>",
            )
        folds_chart.update_layout(
            height=280, margin=dict(l=8, r=12, t=12, b=12),
            legend=dict(orientation="h", y=1.12),
            xaxis=dict(title="Fold", dtick=1),
            yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#e7e5f2"),
        )
        st.plotly_chart(folds_chart, width="stretch", config={"displayModeBar": False})


def main() -> None:
    st.set_page_config(page_title="TextLab · manuel_solutions", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)
    with st.sidebar:
        st.title("TextLab")
        st.caption("manuel_solutions")
        st.markdown("[Conjunto de datos](#conjunto-de-datos)  \n[Historial](#historial-de-experimentos)")

    st.markdown(
        '<header class="textlab-header"><p class="textlab-brand">manuel_solutions</p><h1>TextLab</h1>'
        '<p>Entrena y compara modelos de clasificación de textos desde tus datos.</p></header>',
        unsafe_allow_html=True,
    )
    st.subheader("Conjunto de datos")
    uploaded = st.file_uploader(
        "Archivo CSV", type=["csv"], help="Use un archivo con encabezados en la primera fila."
    )
    use_sample = False
    if uploaded is None:
        use_sample = st.checkbox("Probar con un CSV de ejemplo")
        if not use_sample:
            return

    payload = SAMPLE.encode("utf-8") if use_sample else uploaded.getvalue()
    try:
        data = read_csv(payload)
    except (UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
        st.error("No se pudo leer el CSV. Revise que tenga encabezados y un separador consistente.")
        return

    dataset_name = "Ejemplo sintético" if use_sample else uploaded.name
    dataset_id = sha256(payload).hexdigest()
    st.caption(f"Archivo: {dataset_name}")
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

    with st.sidebar:
        st.markdown("[Entrenamiento](#entrenamiento)  \n[Resultados](#resultados)")

    if perfilar_dataset is None:
        st.metric("Filas", len(data))
        show_distribution(preview_distribution(data[label_column]))
        show_results(show_training_settings(data, text_column, label_column, dataset_id, dataset_name))
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
    show_results(show_training_settings(data, text_column, label_column, dataset_id, dataset_name))


if __name__ == "__main__":
    main()
    show_experiment_history()
