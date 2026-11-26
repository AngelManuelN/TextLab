from hashlib import sha256
import csv
from io import StringIO
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

from textlab.entrenamiento import entrenar_modelo
from textlab.idioma import detectar_idioma
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
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 95% 0%, #d8e7fe 0%, transparent 50%), #edf3fa;
}
[data-testid="stHeader"] {background:#edf3fad9;backdrop-filter:blur(12px);}
[data-testid="stMainBlockContainer"] {max-width:1440px;padding-top:2.8rem;container-type:inline-size;}
[data-testid="stSidebar"] {background:linear-gradient(168deg,#102e58,#102447);border-right:1px solid #264975;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.68rem;}
.tl-brand {display:flex;align-items:center;gap:.7rem;padding:.3rem 0 .8rem;color:#fff;}
.tl-brand strong {display:block;font-size:1.25rem;line-height:1.25;}
.tl-brand small {display:block;color:#b7cbee;font-size:.76rem;}
.tl-symbol {display:grid;place-items:center;width:2.2rem;height:2.2rem;border-radius:.7rem;color:#fff;
    background:linear-gradient(145deg,#8dbdff,#3e68c8);box-shadow:inset 0 1px #e8f2ff88;font-size:1.2rem;}
.textlab-header {display:flex;align-items:center;gap:.8rem;padding:.75rem 1.05rem;border-radius:1.2rem;
    background:#fff;border:1px solid #dbe6f4;box-shadow:0 4px 18px #254b7710;}
.textlab-header h1 {font-size:1.52rem;letter-spacing:-.04em;line-height:1.1;padding:0;margin:0;color:#142b4f;}
.textlab-header p {font-size:.83rem;margin:.22rem 0 0;color:#506883;}
.tl-intro,.tl-format {border:1px solid #d7e4f3;border-radius:1.2rem;padding:1.2rem 1.35rem;min-height:12.2rem;}
.tl-intro {position:relative;overflow:hidden;background:linear-gradient(140deg,#fff 35%,#e2eeff);}
.tl-intro::after {content:"";position:absolute;inset:0;pointer-events:none;
    background:repeating-radial-gradient(ellipse at 100% 105%,transparent 0 21px,#486fba28 23px 24px,transparent 26px 41px);
    mask-image:linear-gradient(90deg,transparent 58%,black);}
.tl-intro h2,.tl-format h2 {font-size:1.3rem;line-height:1.2;padding:0;margin:.65rem 0 .46rem;}
.tl-intro p,.tl-format p {max-width:31rem;margin:0;line-height:1.5;font-size:.94rem;}
.tl-format {background:#142e55;color:#e9f2ff;}
.tl-format h2 {color:#fff;}
[data-testid="stMetric"],.tl-kpi {border:1px solid #dbe5f2;border-radius:1rem;padding:.83rem 1rem;
    background:linear-gradient(145deg,#fff,#f1f6ff);box-shadow:0 4px 14px #254b770c;}
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background:#1b3a65;border-color:#50729d;color:#f0f5ff;box-shadow:none;
}
[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {color:#f0f5ff;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#c5d8f2;}
.tl-kpi {min-height:7.25rem;}
.tl-kpi.is-primary {background:linear-gradient(145deg,#eaf3ff,#d9e8fd);}
.tl-kpi-icon {display:grid;place-items:center;width:1.8rem;height:1.8rem;border-radius:.55rem;margin-bottom:.3rem;
    background:linear-gradient(145deg,#e6f2ff,#c5d9fb);color:#284b9a;border:1px solid #c6daf7;}
.tl-kpi-label {color:#405b7e;font-size:.82rem;}
.tl-kpi-value {display:block;font-size:clamp(1.45rem,3cqw,2.1rem);font-weight:700;color:#142c52;
    line-height:1.25;font-variant-numeric:tabular-nums;letter-spacing:-.03em;}
[data-testid="stFileUploaderDropzone"] {border:1px dashed #819ac0;border-radius:.9rem;}
[data-testid="stTextArea"] textarea {border-radius:.8rem;line-height:1.55;overflow-wrap:anywhere;}
[data-testid="stBaseButton-primary"] {border-radius:.8rem;font-weight:700;}
[data-testid="stExpander"] details {background:#fff;border:1px solid #d7e3f2;border-radius:1rem;}
[data-testid="stSidebar"] [data-testid="stExpander"] details {background:#183963;border-color:#385580;}
[data-testid="stExpander"] summary {padding:.77rem .95rem;}
[data-testid="stTabs"] [role="tablist"] {gap:.3rem;overflow-x:auto;}
[data-testid="stTabs"] [role="tab"] {padding-inline:.7rem;white-space:nowrap;}
@container (max-width:720px) {
    [data-testid="stHorizontalBlock"] {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {width:100%;min-width:0;}
}
@container (max-width:500px) {
    [data-testid="stHorizontalBlock"] {grid-template-columns:minmax(0,1fr);}
}
</style>
"""


def read_csv(payload: bytes, delimiter: str | None = None, has_header: bool = True, quoted: bool = True) -> pd.DataFrame:
    if payload.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        decoded = payload.decode("utf-32")
    elif payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        decoded = payload.decode("utf-16")
    else:
        try:
            decoded = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            decoded = payload.decode("cp1252")

    if "\x00" in decoded:
        raise UnicodeError("El archivo contiene caracteres nulos. Exporta como UTF-8 o UTF-16 con BOM.")

    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(decoded[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","

    quoting = csv.QUOTE_MINIMAL if quoted else csv.QUOTE_NONE
    reader = csv.reader(StringIO(decoded), delimiter=delimiter, quoting=quoting, strict=True)
    expected = None
    for row in reader:
        if not row:
            continue
        if expected is None:
            expected = len(row)
        elif len(row) > expected:
            raise pd.errors.ParserError(
                f"La línea {reader.line_num} tiene {len(row)} campos y se esperaban {expected}. Revisa el separador o las comillas."
            )

    data = pd.read_csv(StringIO(decoded), sep=delimiter, header=0 if has_header else None,
                       dtype=str, keep_default_na=False, na_values=[""], quoting=quoting)
    if not has_header:
        data.columns = [f"Columna {i + 1}" for i in range(len(data.columns))]
    return data



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
        xaxis=dict(title="Textos", rangemode="tozero", gridcolor="#dce7f5", dtick=1 if max(counts) < 10 else None),
        yaxis=dict(type="category", autorange="reversed", automargin=True),
    )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})


RAPIDO = {
    "train_classifier": "SVM lineal",
    "train_mode": "Palabras",
    "train_size": "Unigramas",
    "train_representation": "TF-IDF",
    "train_test_percent": 20,
    "train_folds": None,
    "train_c": 1.0,
}


def usar_configuracion_rapida():
    st.session_state.update(RAPIDO)


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

    st.button("Configuración rápida", icon=":material/bolt:", width="stretch", on_click=usar_configuracion_rapida)
    classifier = st.selectbox("Clasificador", tuple(CLASSIFIERS), index=1, key="train_classifier")
    mode = st.selectbox(
        "Características del texto", tuple(TEXT_MODES), key="train_mode",
        help="Palabras usa los términos del texto. POS usa categorías gramaticales. Stopwords conserva solo palabras frecuentes. Palabras de contenido conserva sustantivos, verbos, adjetivos y adverbios, entre otros.",
    )
    if CLASSIFIERS[classifier] in {"logistica", "svm"}:
        value = st.number_input("Regularización inversa (C)", min_value=0.001, value=1.0, step=0.1, format="%.3f", key="train_c")
        parameter = {"C": value}
    elif CLASSIFIERS[classifier] == "random_forest":
        parameter = {"n_arboles": st.number_input("Número de árboles", min_value=1, value=100, step=1)}
    else:
        value = st.number_input("Suavizado (alpha)", min_value=0.001, value=1.0, step=0.1, format="%.3f")
        parameter = {"alpha": value}

    features = st.selectbox(
        "Tamaño de los n-gramas",
        tuple(NGRAMS), key="train_size",
        help="Agrupa uno, dos o tres elementos consecutivos del modo elegido.",
    )
    representation = st.selectbox("Representación", ("TF-IDF", "Frecuencia", "Binaria"), key="train_representation",
                                  help="Convierte las características en números: TF-IDF reduce el peso de términos comunes, Frecuencia cuenta apariciones y Binaria indica presencia.")
    test_percent = st.slider(
        "Datos reservados para prueba", 10, 50, 20, format="%d %%", key="train_test_percent"
    )
    st.caption(f"{100 - test_percent} % para entrenar y {test_percent} % para probar.")
    folds = st.selectbox(
        "Validación cruzada",
        (None, 3, 5, 10), key="train_folds",
        format_func=lambda value: "Desactivada" if value is None else f"{value} folds",
        help="Evalúa el modelo varias veces con distintas divisiones de los datos de entrenamiento.",
    )

    proportion = test_percent / 100
    warnings = validar_configuracion_entrenamiento(clean_data[label_column], proportion, folds)
    for message in warnings:
        st.warning(message)

    if folds and TEXT_MODES[mode] != "palabras" and len(clean_data) >= 10000:
        st.caption("El análisis lingüístico se repite en cada fold. Para una prueba rápida, usa 3 folds o desactiva la validación cruzada.")

    language = "Libre"
    language_code = None
    language_blocked = False
    if TEXT_MODES[mode] != "palabras":
        selection = st.selectbox(
            "Idioma del texto", ("Automático", *LANGUAGES),
            help="Puedes corregir la detección eligiendo el idioma del CSV.",
        )
        if selection != "Automático":
            language, language_code = selection, LANGUAGES[selection]
        elif not warnings:
            training_texts, _ = train_test_split(
                clean_data[text_column].astype(str).tolist(), test_size=proportion,
                stratify=clean_data[label_column], random_state=42,
            )
            language_code = detectar_idioma(training_texts)
            language = next((name for name, code in LANGUAGES.items() if code == language_code), None)
            if language:
                st.caption(f"Idioma detectado: {language}.")
            else:
                language_blocked = True
                reason = f"Idioma detectado ({language_code}) sin analizador." if language_code else "El idioma no se pudo determinar con suficiente confianza."
                st.warning(f"{reason} Puedes elegir un idioma manualmente o usar Palabras.")

    settings = (dataset_id, text_column, label_column, proportion, features, representation, folds,
                classifier, mode, language, tuple(parameter.items()))
    experiments = st.session_state.setdefault("experimentos", [])
    matching = next((entry for entry in reversed(experiments) if entry["settings"] == settings), None)
    previous = next((entry for entry in reversed(experiments) if entry["settings"][0] == dataset_id), None)
    if matching is not None:
        st.success(f"Modelo entrenado · Experimento {matching['summary']['Experimento']}")
    elif previous is not None:
        st.info("Cambiaste la configuración. Pulsa Entrenar para aplicarla. El modelo anterior está en Experimentos y Predicción.")

    if st.button("Entrenar", type="primary", icon=":material/play_arrow:", width="stretch", disabled=bool(warnings) or language_blocked):
        with st.status("Preparando los datos…", expanded=True) as progress:
            try:
                result = entrenar_modelo(
                    clean_data[text_column].astype(str).tolist(),
                    clean_data[label_column].tolist(),
                    proporcion_prueba=proportion,
                    ngram_range=NGRAMS[features],
                    representacion={"TF-IDF": "tfidf", "Frecuencia": "frecuencia", "Binaria": "binaria"}[representation],
                    folds=folds,
                    clasificador=CLASSIFIERS[classifier],
                    modo_texto=TEXT_MODES[mode],
                    idioma=language_code or "es",
                    incluir_detalles_prueba=True,
                    al_avanzar=lambda mensaje: progress.update(label=mensaje),
                    **parameter,
                )
            except (RuntimeError, ValueError) as error:
                progress.update(label="Entrenamiento interrumpido", state="error")
                if "empty vocabulary" in str(error).lower():
                    st.error("No hay suficientes características para esa selección. Prueba con n-gramas más cortos o con el modo Palabras.")
                else:
                    st.error(f"No se pudo entrenar con estos datos. {error}")
                return None
            progress.update(label="Modelo entrenado · abre Resultados", state="complete", expanded=False)
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
                    "Idioma": language,
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
        with st.form("prediccion_individual"):
            text = st.text_area("Texto por clasificar", height=230, placeholder="Pega una frase, una reseña o varios párrafos.")
            submitted = st.form_submit_button("Predecir texto", type="primary", icon=":material/bolt:")
        if submitted:
            st.session_state.pop("prediccion_individual_resultado", None)
            if not text.strip():
                st.warning("Escribe un texto para clasificar.")
            else:
                try:
                    prediction = predecir_lote([text], predictor)[0]["etiqueta"]
                except ValueError as error:
                    st.error(f"No se pudo clasificar el texto. {error}")
                else:
                    st.session_state["prediccion_individual_resultado"] = (chosen, text, prediction)
        cached_text = st.session_state.get("prediccion_individual_resultado")
        if cached_text is not None and cached_text[:2] == (chosen, text):
            st.info(f"Clase predicha: {cached_text[2]}", icon=":material/label:")
            st.caption("Resultado del último texto enviado.")
            st.text(text if len(text) <= 240 else text[:237] + "…")

    with batch:
        upload = st.file_uploader("CSV de textos sin etiquetas", type=["csv"], key="prediction_upload")
        if upload is None:
            return
        payload = upload.getvalue()
        try:
            data = read_csv(payload)
        except (UnicodeError, csv.Error, pd.errors.EmptyDataError, pd.errors.ParserError):
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
        if st.button("Predecir CSV", type="primary", icon=":material/bolt:", disabled=bool(has_empty)):
            try:
                predictions = predecir_lote(texts.tolist(), predictor)
            except ValueError as error:
                st.error(f"No se pudo clasificar el CSV. {error}")
                return
            st.session_state["prediccion_lote"] = (key, pd.DataFrame(predictions).rename(columns={"etiqueta": "prediccion"}))
        cached = st.session_state.get("prediccion_lote")
        if cached is not None and cached[0] == key:
            output = cached[1]
            st.dataframe(output, hide_index=True, width="stretch")
            st.download_button(
                "Descargar predicciones", output.to_csv(index=False).encode("utf-8-sig"),
                file_name="textlab-predicciones.csv", mime="text/csv", on_click="ignore", icon=":material/download:",
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
    for index, (column, (name, value), icon) in enumerate(
        zip(st.columns(4), values, ("◎", "◈", "◇", "↯"))
    ):
        variant = " is-primary" if index == 0 else ""
        column.markdown(
            f'<div class="tl-kpi{variant}"><span class="tl-kpi-icon" aria-hidden="true">{icon}</span>'
            f'<span class="tl-kpi-label">{name}</span><strong class="tl-kpi-value">{value}</strong></div>',
            unsafe_allow_html=True,
        )
    left, right = st.columns((1, 1.2))

    with left.expander("Matriz de confusión", expanded=True, icon=":material/grid_view:"):
        st.caption("La diagonal muestra los aciertos. Las demás celdas muestran qué clases se confundieron.")
        labels = [str(label) for label in classes]
        matrix = confusion_matrix(actual, predicted, labels=classes)
        chart = go.Figure(
            go.Heatmap(
                z=matrix.tolist(),
                x=labels,
                y=labels,
                colorscale=[[0, "#eff5ff"], [1, "#3977c7"]],
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

    with right.expander("Desempeño por clase", expanded=True, icon=":material/bar_chart:"):
        st.caption(f"Recall macro {metrics['Recall macro']:.1%}. Pasa sobre una barra para ver sus valores.")
        precision, recall, f1, support = precision_recall_fscore_support(
            actual, predicted, labels=classes, zero_division=0
        )
        per_class = go.Figure()
        for name, values, color in (
            ("Precisión", precision, "#519cff"),
            ("Recall", recall, "#147f93"),
            ("F1", f1, "#7a60b7"),
        ):
            per_class.add_bar(
                x=labels, y=values.tolist(), name=name, marker_color=color,
                customdata=support.tolist(),
                hovertemplate="%{x}<br>%{y:.1%}<br>Textos de prueba: %{customdata}<extra>%{fullData.name}</extra>",
            )
        per_class.update_layout(
            barmode="group", height=330, margin=dict(l=8, r=12, t=12, b=12),
            legend=dict(orientation="h", y=1.12),
            yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#dce7f5"),
            xaxis=dict(type="category", automargin=True),
        )
        st.plotly_chart(per_class, width="stretch", config={"displayModeBar": False})

    validation = result.get("validacion_cruzada")
    if validation:
        with st.expander("Validación cruzada", icon=":material/monitoring:"):
            st.caption(f"{validation['folds']} folds sobre la parte de entrenamiento. La prueba final se evaluó por separado.")
            columns = st.columns(2)
            folds_chart = go.Figure()
            for column, (key, name, color) in zip(
                columns, (("accuracy", "Accuracy", "#519cff"), ("f1_macro", "F1 macro", "#147f93"))
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
                yaxis=dict(tickformat=".0%", range=[0, 1], gridcolor="#dce7f5"),
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
    class_column, direction_column = st.columns(2)
    chosen = class_column.selectbox(
        "Filtrar por clase", [None, *labels], format_func=lambda x: "Todas" if x is None else str(x),
    )
    if chosen is not None:
        direction = direction_column.selectbox(
            "Buscar en", ["Clase real", "Clase predicha", "Ambas"],
            help="Clase real muestra casos que el modelo pasó por alto. Clase predicha muestra casos a los que asignó esa etiqueta por error.",
        )
        if direction == "Ambas":
            errors = errors.loc[errors["Clase real"].eq(chosen) | errors["Clase predicha"].eq(chosen)]
        else:
            errors = errors.loc[errors[direction].eq(chosen)]
    st.caption(f"Errores mostrados: {len(errors)}")
    if errors.empty:
        return
    st.dataframe(errors, width="stretch", hide_index=True)
    with st.expander("Leer un texto completo", icon=":material/article:"):
        selected = st.selectbox("Fila de prueba", errors.index.tolist(), format_func=lambda value: f"{value + 1}")
        st.text_area("Texto completo", value=str(errors.loc[selected, "Texto"]), disabled=True, height=240,
                     key=f"texto_error_{id(result)}_{selected}")


def main() -> None:
    st.set_page_config(page_title="TextLab · manuel_solutions", layout="wide", initial_sidebar_state="expanded")
    st.markdown(STYLE, unsafe_allow_html=True)
    dataset = None
    result = None
    with st.sidebar:
        st.markdown('<div class="tl-brand"><span class="tl-symbol" aria-hidden="true">◈</span>'
                    '<div><strong>TextLab</strong><small>manuel_solutions</small></div></div>', unsafe_allow_html=True)
        st.subheader("Conjunto de datos")
        uploaded = st.file_uploader("Archivo CSV", type=["csv", "tsv", "txt"])
        use_sample = uploaded is None and st.checkbox("Probar con un CSV de ejemplo")
        if uploaded is not None or use_sample:
            sample = Path(__file__).parent / "datos" / "pruebas" / "carga_valida.csv"
            payload = sample.read_bytes() if use_sample else uploaded.getvalue()
            with st.expander("Formato de importación", icon=":material/tune:"):
                separators = {"Automático": None, "Coma": ",", "Punto y coma": ";", "Tabulación": "\t", "Barra vertical": "|"}
                separator = st.selectbox("Separador", tuple(separators))
                has_header = st.checkbox("La primera fila tiene encabezados", value=True)
                quoted = st.checkbox("Las comillas delimitan campos", value=True)
            try:
                data = read_csv(payload, separators[separator], has_header, quoted)
            except (UnicodeError, csv.Error, pd.errors.EmptyDataError, pd.errors.ParserError) as error:
                st.error(f"No se pudo leer el CSV. Revisa el formato de importación. {error}")
            else:
                if data.empty or len(data.columns) < 2:
                    st.warning("Se necesita una columna de texto y otra de etiquetas. Si ves una sola columna, cambia el separador en Formato de importación.")
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
        '<header class="textlab-header"><span class="tl-symbol" aria-hidden="true">◈</span><div><h1>TextLab</h1>'
        '<p>Laboratorio de clasificación de textos</p></div></header>',
        unsafe_allow_html=True,
    )
    if dataset is None:
        left, right = st.columns((1.55, 1))
        left.markdown(
            '<section class="tl-intro"><span class="tl-kpi-icon" aria-hidden="true">▤</span>'
            '<h2>Empieza con tus datos</h2><p>Sube un CSV desde la barra lateral. Elige la columna que contiene '
            'los textos y la que contiene sus categorías.</p></section>', unsafe_allow_html=True,
        )
        right.markdown(
            '<section class="tl-format"><span class="tl-kpi-icon" aria-hidden="true">⇄</span>'
            '<h2>Formato flexible</h2><p>Usa los nombres de columna del archivo original. '
            'Puedes cambiar el separador o indicar que el CSV viene sin encabezados.</p></section>', unsafe_allow_html=True,
        )
        return
    explore, results, errors, predictions, history = st.tabs(
        ["Explorador de datos", "Resultados", "Análisis de errores", "Predicción", "Experimentos"],
        default="Resultados" if result is not None else "Explorador de datos",
    )
    with results:
        show_results(result)
    with explore:
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
