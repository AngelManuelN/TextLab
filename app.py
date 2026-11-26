from hashlib import sha256
import csv
from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

from textlab.entrenamiento import entrenar_modelo
from textlab.prediccion import predecir_lote
from textlab.preparacion import preparar_dataset
from textlab.validacion_entrenamiento import validar_configuracion_entrenamiento

from textlab.datos import perfilar_dataset


CLASSIFIERS = {
    "Regresión logística": "logistica",
    "SVM lineal": "svm",
    "Random Forest": "random_forest",
    "Naive Bayes": "naive_bayes",
}
TEXT_MODES = {
    "Palabras": "palabras",
    "Etiquetas gramaticales (POS)": "pos",
    "Stopwords": "stopwords",
    "Palabras de contenido": "palabras_contenido",
}

LANGUAGES = {"Español": "es", "Inglés": "en", "Francés": "fr"}
NGRAMS = {"Unigramas": (1, 1), "Bigramas": (2, 2), "Trigramas": (3, 3)}


STYLE = """
<style>
[data-testid="stMainBlockContainer"] {max-width: 1450px; padding-top: 3rem; container-type: inline-size;}
[data-testid="stSidebar"] {border-right: 1px solid #294a73;}
.textlab-header {
    padding: clamp(1.3rem, 3.4vw, 3rem);
    border: 1px solid #427ab0;
    border-radius: 1.45rem;
    background: radial-gradient(ellipse at 90% 0%, #176ed0 0%, transparent 62%),
                radial-gradient(ellipse at 84% 130%, #0987af 0%, transparent 68%),
                linear-gradient(115deg, #051328, #09275b);
    box-shadow: inset 0 1px #c1e8ff69, 0 12px 32px #03132533;
}
.textlab-header h1 {color: #f3f9ff; margin: 0; padding: 0; font-size: clamp(2.55rem, 6vw, 4.4rem);
    font-weight: 750; letter-spacing: -.065em; line-height: 1.05;}
.textlab-header p {color: #c1e8ff; margin: .55rem 0 0; max-width: 38rem;}
.textlab-header .textlab-brand {color: #a9d6fd; margin: 0 0 .62rem; font-size: .85rem;}
[data-testid="stMetric"] {
    border: 1px solid #365277; border-radius: .9rem; padding: .85rem 1rem;
    background: linear-gradient(155deg, #172d4b, #101f35);
    font-variant-numeric: tabular-nums;
}
[data-testid="stFileUploaderDropzone"] {border: 1px dashed #6583a8; border-radius: .85rem;}
[data-testid="stExpander"] details {border-color: #365277;}
[data-testid="stExpander"] summary:hover {background: #1c3859;}
@media (prefers-reduced-motion: no-preference) {
    [data-testid="stExpander"] details[open] > div {animation: textlab-reveal 180ms ease-out;}
    @keyframes textlab-reveal {from {opacity: .55; transform: translateY(-3px);} to {opacity: 1; transform: none;}}
}
@container (max-width: 760px) {
    [data-testid="stHorizontalBlock"] {display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {width: 100%; min-width: 0;}
}
@container (max-width: 520px) {
    [data-testid="stHorizontalBlock"] {grid-template-columns: minmax(0, 1fr);}
}
</style>
"""


def read_csv(payload: bytes) -> pd.DataFrame:
    if payload.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        decoded = payload.decode("utf-32")
    elif payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        decoded = payload.decode("utf-16")
    else:
        try:
            decoded = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            decoded = payload.decode("cp1252")

    try:
        delimiter = csv.Sniffer().sniff(decoded[:8192], delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","

    return pd.read_csv(StringIO(decoded), sep=delimiter)


def show_distribution(distribution: dict) -> None:
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
            marker_color="#52a8ff",
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
        xaxis=dict(title="Textos", rangemode="tozero", gridcolor="#304461", dtick=1 if max(counts) < 10 else None),
        yaxis=dict(type="category", autorange="reversed", automargin=True),
    )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})


def show_training_settings(
    data: pd.DataFrame, text_column: str, label_column: str, dataset_id: str, dataset_name: str
) -> dict | None:
    clean_data, summary = preparar_dataset(data, text_column, label_column)

    st.subheader("Entrenamiento")
    st.metric("Filas utilizables", summary["filas_utilizables"])
    st.caption(
        f"Filas excluidas · Sin datos ({summary['filas_sin_datos']}), "
        f"duplicadas ({summary['filas_duplicadas']}) y con etiquetas contradictorias ({summary['filas_en_conflicto']})."
    )
    if summary["filas_utilizables"] == 0:
        st.warning("No quedaron filas utilizables. Revise los datos o elija otras columnas.")
        return

    classifier = st.selectbox("Clasificador", tuple(CLASSIFIERS), index=1)
    mode = st.selectbox(
        "Características del texto", tuple(TEXT_MODES),
        help="Los modos lingüísticos usan el analizador del idioma seleccionado.",
    )
    language = st.selectbox("Idioma del texto", tuple(LANGUAGES), disabled=TEXT_MODES[mode] == "palabras",
                            help="Palabras funciona sin analizador. Para POS, stopwords y palabras de contenido, elige el idioma del CSV.")
    if CLASSIFIERS[classifier] in {"logistica", "svm"}:
        value = st.number_input("Regularización inversa (C)", min_value=0.001, value=1.0, step=0.1, format="%.3f")
        parameter = {"C": value}
    elif CLASSIFIERS[classifier] == "random_forest":
        parameter = {"n_arboles": st.number_input("Número de árboles", min_value=1, value=100, step=1)}
    else:
        value = st.number_input("Suavizado (alpha)", min_value=0.001, value=1.0, step=0.1, format="%.3f")
        parameter = {"alpha": value}

    features = st.selectbox(
        "Tamaño de los n-gramas",
        tuple(NGRAMS),
        help="Agrupa uno, dos o tres elementos consecutivos del modo elegido.",
    )
    representation = st.selectbox("Representación", ("TF-IDF", "Frecuencia", "Binaria"))
    test_percent = st.slider(
        "Datos reservados para prueba", 10, 50, 20, format="%d %%"
    )
    st.caption(f"{100 - test_percent} % para entrenar y {test_percent} % para probar.")
    folds = st.selectbox(
        "Validación cruzada",
        (None, 3, 5, 10),
        format_func=lambda value: "Desactivada" if value is None else f"{value} folds",
        help="Evalúa el modelo varias veces con distintas divisiones de los datos de entrenamiento.",
    )

    proportion = test_percent / 100
    warnings = validar_configuracion_entrenamiento(clean_data[label_column], proportion, folds)
    for message in warnings:
        st.warning(message)

    settings = (dataset_id, text_column, label_column, proportion, features, representation, folds,
                classifier, mode, language, tuple(parameter.items()))
    experiments = st.session_state.setdefault("experimentos", [])
    if st.button("Entrenar", type="primary", disabled=bool(warnings)):
        try:
            with st.spinner(f"Entrenando {classifier}…"):
                result = entrenar_modelo(
                    clean_data[text_column].astype(str).tolist(),
                    clean_data[label_column].tolist(),
                    proporcion_prueba=proportion,
                    ngram_range=NGRAMS[features],
                    representacion={"TF-IDF": "tfidf", "Frecuencia": "frecuencia", "Binaria": "binaria"}[representation],
                    folds=folds,
                    clasificador=CLASSIFIERS[classifier],
                    modo_texto=TEXT_MODES[mode],
                    idioma=LANGUAGES[language],
                    incluir_detalles_prueba=True,
                    **parameter,
                )
        except (RuntimeError, ValueError) as error:
            if "empty vocabulary" in str(error).lower():
                st.error("No hay suficientes características para esa selección. Prueba con n-gramas más cortos o con el modo Palabras.")
            else:
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
                    "Clasificador": classifier,
                    "Características": mode,
                    "Idioma": language if TEXT_MODES[mode] != "palabras" else "Libre",
                    "N-gramas": features,
                    "Parámetros": ", ".join(f"{key}={value}" for key, value in parameter.items()),
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
            "Experimento", "Archivo", "Clasificador", "Características", "Idioma", "N-gramas", "Representación",
            "Prueba (%)", "Accuracy", "Precisión macro", "Recall macro", "F1 macro",
            "Textos de prueba", "Validación cruzada", "Parámetros", "Columna de texto", "Columna de etiqueta",
        ),
    )


def show_predictions() -> None:
    st.subheader("Predicción")
    experiments = st.session_state.get("experimentos", [])
    if not experiments:
        st.info("Entrena un modelo para usarlo en una predicción.")
        return

    summaries = [entry["summary"] for entry in experiments]
    chosen = st.selectbox(
        "Modelo entrenado", range(len(experiments)), index=len(experiments) - 1,
        format_func=lambda i: f"Experimento {summaries[i]['Experimento']} · {summaries[i]['Clasificador']} · {summaries[i]['Archivo']}",
    )
    predictor = experiments[chosen]["result"]["pipeline"].predict
    single, batch = st.tabs(["Texto individual", "Archivo CSV"])
    with single:
        text = st.text_area("Texto por clasificar")
        if st.button("Predecir texto", disabled=not text.strip()):
            prediction = predecir_lote([text], predictor)[0]["etiqueta"]
            st.success(f"Clase predicha: {prediction}")

    with batch:
        upload = st.file_uploader("CSV de textos sin etiquetas", type=["csv"], key="prediction_upload")
        if upload is None:
            return
        payload = upload.getvalue()
        try:
            data = read_csv(payload)
        except (UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
            st.error("No se pudo leer el CSV de predicción. Revisa los encabezados y el separador.")
            return
        if data.empty:
            st.warning("El CSV de predicción debe contener al menos un texto.")
            return
        text_column = st.selectbox("Columna de texto para predecir", data.columns)
        texts = data[text_column].fillna("").astype(str)
        has_empty = texts.str.strip().eq("").any()
        if has_empty:
            st.warning("La columna seleccionada contiene textos vacíos. Elige otra columna o corrige el CSV.")
        key = (chosen, sha256(payload).hexdigest(), text_column)
        if st.button("Predecir CSV", disabled=bool(has_empty)):
            predictions = predecir_lote(texts.tolist(), predictor)
            st.session_state["prediccion_lote"] = (key, pd.DataFrame(predictions).rename(columns={"etiqueta": "prediccion"}))
        cached = st.session_state.get("prediccion_lote")
        if cached is not None and cached[0] == key:
            output = cached[1]
            st.dataframe(output, hide_index=True, width="stretch")
            st.download_button(
                "Descargar predicciones", output.to_csv(index=False).encode("utf-8-sig"),
                file_name="textlab-predicciones.csv", mime="text/csv", on_click="ignore",
            )


def show_results(result: dict | None) -> None:
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
    metrics = calculate_metrics(actual, predicted)
    values = [(name, f"{metrics[name]:.1%}") for name in ("Accuracy", "F1 macro", "Precisión macro")]
    values.append(("Inferencia por texto", f"{result.get('inferencia_ms_por_texto', 0):.3f} ms"))
    for column, (name, value) in zip(st.columns(4), values):
        column.metric(name, value)
    left, right = st.columns((1, 1.2))

    with left.expander("Matriz de confusión", expanded=True):
        st.caption("La diagonal muestra los aciertos. Las demás celdas muestran qué clases se confundieron.")
        labels = [str(label) for label in classes]
        matrix = confusion_matrix(actual, predicted, labels=classes)
        chart = go.Figure(
            go.Heatmap(
                z=matrix.tolist(),
                x=labels,
                y=labels,
                colorscale=[[0, "#182d4c"], [1, "#338df6"]],
                texttemplate="%{z}",
                hovertemplate="Real: %{y}<br>Predicha: %{x}<br>Textos: %{z}<extra></extra>",
                colorbar=dict(title="Textos"),
            )
        )
        chart.update_layout(
            height=330, margin=dict(l=8, r=12, t=12, b=12),
            xaxis=dict(title="Clase predicha", type="category", automargin=True),
            yaxis=dict(title="Clase real", type="category", autorange="reversed", automargin=True),
        )
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})

    with right.expander("Desempeño por clase", expanded=True):
        st.caption(f"Recall macro {metrics['Recall macro']:.1%}. Pasa sobre una barra para ver sus valores.")
        precision, recall, f1, support = precision_recall_fscore_support(
            actual, predicted, labels=classes, zero_division=0
        )
        per_class = go.Figure()
        for name, values, color in (
            ("Precisión", precision, "#519cff"),
            ("Recall", recall, "#39d7ee"),
            ("F1", f1, "#a1c8ff"),
        ):
            per_class.add_bar(
                x=labels, y=values.tolist(), name=name, marker_color=color,
                customdata=support.tolist(),
                hovertemplate="%{x}<br>%{y:.1%}<br>Textos de prueba: %{customdata}<extra>%{fullData.name}</extra>",
            )
        per_class.update_layout(
            barmode="group", height=330, margin=dict(l=8, r=12, t=12, b=12),
            legend=dict(orientation="h", y=1.12),
            yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#304461"),
            xaxis=dict(type="category", automargin=True),
        )
        st.plotly_chart(per_class, width="stretch", config={"displayModeBar": False})

    validation = result.get("validacion_cruzada")
    if validation:
        with st.expander("Validación cruzada"):
            st.caption(f"{validation['folds']} folds sobre la parte de entrenamiento. La prueba final se evaluó por separado.")
            columns = st.columns(2)
            folds_chart = go.Figure()
            for column, (key, name, color) in zip(
                columns, (("accuracy", "Accuracy", "#519cff"), ("f1_macro", "F1 macro", "#39d7ee"))
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
                yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#304461"),
            )
            st.plotly_chart(folds_chart, width="stretch", config={"displayModeBar": False})


def show_errors(result: dict | None) -> None:
    if result is None:
        st.info("Entrena un modelo para revisar las clasificaciones incorrectas.")
        return
    errors = pd.DataFrame({
        "Texto": result.get("textos_prueba", [""] * len(result["etiquetas_reales"])),
        "Clase real": result["etiquetas_reales"],
        "Clase predicha": result["etiquetas_predichas"],
    })
    errors = errors.loc[errors["Clase real"] != errors["Clase predicha"]]
    if errors.empty:
        st.success("El modelo acertó en todos los textos de prueba.")
        return
    labels = sorted(set(errors["Clase real"]).union(errors["Clase predicha"]), key=str)
    chosen = st.selectbox("Filtrar por clase", [None, *labels], format_func=lambda x: "Todas" if x is None else str(x))
    if chosen is not None:
        errors = errors.loc[errors["Clase real"].eq(chosen) | errors["Clase predicha"].eq(chosen)]
    st.dataframe(errors, width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title="TextLab · manuel_solutions", layout="wide", initial_sidebar_state="expanded")
    st.markdown(STYLE, unsafe_allow_html=True)
    dataset = None
    result = None
    with st.sidebar:
        st.header("Conjunto de datos")
        uploaded = st.file_uploader("Archivo CSV", type=["csv"])
        use_sample = uploaded is None and st.checkbox("Probar con un CSV de ejemplo")
        if uploaded is not None or use_sample:
            sample = Path(__file__).parent / "datos" / "pruebas" / "carga_valida.csv"
            payload = sample.read_bytes() if use_sample else uploaded.getvalue()
            try:
                data = read_csv(payload)
            except (UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError):
                st.error("No se pudo leer el CSV. Revisa los encabezados y el separador.")
            else:
                if data.empty or len(data.columns) < 2:
                    st.warning("El CSV debe tener filas y al menos dos columnas.")
                else:
                    name = "Ejemplo sintético" if use_sample else uploaded.name
                    text_column = st.selectbox("Columna de texto", data.columns)
                    label_column = st.selectbox("Columna de etiqueta", data.columns, index=1)
                    if text_column == label_column:
                        st.warning("Elige columnas distintas para texto y etiqueta.")
                    else:
                        dataset = (data, text_column, label_column, name)
                        result = show_training_settings(data, text_column, label_column, sha256(payload).hexdigest(), name)

    st.markdown(
        '<header class="textlab-header"><p class="textlab-brand">manuel_solutions</p><h1>TextLab</h1>'
        '<p>Entrena, compara y prueba clasificadores de texto.</p></header>',
        unsafe_allow_html=True,
    )
    show_results(result)
    explore, errors, predictions, history = st.tabs(["Explorador de datos", "Análisis de errores", "Predicción", "Experimentos"])
    with explore:
        if dataset is None:
            st.info("Carga un CSV o activa el ejemplo desde la barra lateral.")
        else:
            data, text_column, label_column, name = dataset
            profile = perfilar_dataset(data, text_column, label_column)
            st.caption(name)
            for column, (title, key) in zip(st.columns(4), (
                ("Filas", "total_filas"), ("Textos faltantes", "textos_faltantes"),
                ("Etiquetas faltantes", "etiquetas_faltantes"), ("Duplicadas", "filas_duplicadas"),
            )):
                column.metric(title, profile[key])
            table, distribution = st.columns((1.2, 1))
            with table.expander("Vista previa del CSV", expanded=True):
                st.dataframe(data.head(100), width="stretch", hide_index=True)
            with distribution.expander("Distribución por clase", expanded=True):
                show_distribution(profile["distribucion_clases"])
    with errors:
        show_errors(result)
    with predictions:
        show_predictions()
    with history:
        show_experiment_history()


if __name__ == "__main__":
    main()
