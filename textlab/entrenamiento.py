import numpy as np
from collections import Counter
from time import perf_counter
from math import isfinite
from numbers import Real

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from textlab.caracteristicas import crear_vectorizador

FOLDS_PERMITIDOS = (3, 5, 10)

# Valores que acepta el parámetro "clasificador" (el primero es el predeterminado).
CLASIFICADORES_PERMITIDOS = ("logistica", "svm", "random_forest", "naive_bayes")

# Qué parámetro propio aplica a cada clasificador.
PARAMETROS_POR_CLASIFICADOR = {
    "logistica": ("C",),
    "svm": ("C",),
    "random_forest": ("n_arboles",),
    "naive_bayes": ("alpha",),
}

# Valores iniciales: se usan cuando no se indica el parámetro.
VALORES_INICIALES = {
    "C": 1.0,
    "n_arboles": 100,
    "alpha": 1.0,
}


def _resumir(valores):
    """Valores por fold, promedio y desviación estándar (poblacional)."""
    valores = [float(v) for v in valores]
    return {
        "por_fold": valores,
        "promedio": float(np.mean(valores)),
        "desviacion": float(np.std(valores)),
    }


def _crear_clasificador(clasificador, C, n_arboles, alpha):
    """Crea el clasificador elegido validando nombre y parámetros."""
    if clasificador not in CLASIFICADORES_PERMITIDOS:
        raise ValueError(
            "clasificador debe ser uno de "
            f"{', '.join(CLASIFICADORES_PERMITIDOS)} (recibí {clasificador!r})."
        )

    indicados = {"C": C, "n_arboles": n_arboles, "alpha": alpha}
    validos = PARAMETROS_POR_CLASIFICADOR[clasificador]

    for nombre, valor in indicados.items():
        if valor is not None and nombre not in validos:
            raise ValueError(
                f"El parámetro '{nombre}' no aplica al clasificador "
                f"'{clasificador}'. Parámetros válidos: {', '.join(validos)}."
            )

    valores = {
        nombre: (indicados[nombre] if indicados[nombre] is not None
                 else VALORES_INICIALES[nombre])
        for nombre in validos
    }

    for nombre in ("C", "alpha"):
        if nombre in valores:
            valor = valores[nombre]
            if isinstance(valor, bool) or not isinstance(valor, Real) or not isfinite(valor) or valor <= 0:
                raise ValueError(f"{nombre} debe ser un número positivo y finito (recibí {valor!r}).")
    if "n_arboles" in valores:
        n = valores["n_arboles"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(
                f"n_arboles debe ser un entero mayor o igual a 1 (recibí {n!r})."
            )

    if clasificador == "logistica":
        return LogisticRegression(C=valores["C"], max_iter=1000, random_state=42)
    if clasificador == "svm":
        return LinearSVC(C=valores["C"], random_state=42)
    if clasificador == "random_forest":
        return RandomForestClassifier(
            n_estimators=valores["n_arboles"], random_state=42
        )
    return MultinomialNB(alpha=valores["alpha"])


def entrenar_modelo(
    textos,
    etiquetas,
    proporcion_prueba=0.20,
    ngram_range=(1, 1),
    representacion="tfidf",
    folds=None,
    clasificador="logistica",
    C=None,
    n_arboles=None,
    alpha=None,
    modo_texto="palabras",
    idioma="es",
    incluir_detalles_prueba=False,
    al_avanzar=None,
):
    """Entrena un clasificador de texto con n-gramas y la representación elegida.

    textos: lista de frases utilizables.
    etiquetas: lista de categorías correctas, en el mismo orden que textos.
    proporcion_prueba: fracción reservada para evaluar (0.20 = 20 %).
    ngram_range: tupla (mínimo, máximo) de palabras por n-grama; (1, 1) = unigramas.
    representacion: "tfidf", "frecuencia" o "binaria".
    modo_texto: "palabras", "pos", "stopwords" o "palabras_contenido".
    folds: None (sin validación cruzada), 3, 5 o 10. La validación cruzada
        estratificada usa solo la parte de entrenamiento.
    clasificador: "logistica" (predeterminado), "svm", "random_forest" o
        "naive_bayes".
    C: fuerza de regularización inversa. Solo para "logistica" y "svm".
        Valor inicial: 1.0.
    n_arboles: número de árboles. Solo para "random_forest". Valor inicial: 100.
    alpha: suavizado de Laplace. Solo para "naive_bayes". Valor inicial: 1.0.
        Si C, n_arboles o alpha son None se usa su valor inicial. Indicar un
        parámetro que no corresponde al clasificador lanza ValueError.

    idioma: código del analizador lingüístico.
    incluir_detalles_prueba: incluye textos evaluados y tiempo de inferencia.

    Devuelve un diccionario con:
      - pipeline: vectorizador + clasificador, ya entrenado.
      - etiquetas_reales: categorías correctas de la parte de prueba.
      - etiquetas_predichas: lo que el modelo predijo para esa misma parte.
      - validacion_cruzada: None si folds es None; si no, un diccionario con
        "folds", "accuracy" y "f1_macro". Cada métrica trae "por_fold"
        (lista), "promedio" y "desviacion".
    """
    if folds is not None and folds not in FOLDS_PERMITIDOS:
        raise ValueError(
            f"folds debe ser None, 3, 5 o 10 (recibí {folds!r})."
        )

    if al_avanzar is not None:
        al_avanzar("Preparando la partición de entrenamiento y prueba…")
    modelo = _crear_clasificador(clasificador, C, n_arboles, alpha)

    # 1. Separar entrenamiento y prueba (reparto reproducible y balanceado por clase)
    textos_entrenamiento, textos_prueba, etiquetas_entrenamiento, etiquetas_prueba = (
        train_test_split(
            textos,
            etiquetas,
            test_size=proporcion_prueba,
            stratify=etiquetas,
            random_state=42,  # semilla provisional
        )
    )

    # 2. Armar el pipeline: primero convierte texto a números, luego clasifica
    pipeline = Pipeline(
        [
            (
                "vectorizador",
                crear_vectorizador(
                    ngram_range=ngram_range,
                    representacion=representacion,
                    modo_texto=modo_texto,
                    idioma=idioma,
                ),
            ),
            ("clasificador", modelo),
        ]
    )

    # 3. Validación cruzada estratificada, solo con entrenamiento (opcional)
    validacion_cruzada = None
    if folds is not None:
        conteo = Counter(etiquetas_entrenamiento)
        clase_minima, ejemplos_minimos = min(conteo.items(), key=lambda par: par[1])
        if ejemplos_minimos < folds:
            raise ValueError(
                f"No se puede hacer validación cruzada con {folds} folds: "
                f"la clase '{clase_minima}' tiene solo {ejemplos_minimos} "
                f"ejemplo(s) en el entrenamiento. Usa menos folds o más datos."
            )
        divisor = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        if al_avanzar is not None:
            al_avanzar(f"Validando el modelo · {folds} folds…")
        resultados = cross_validate(
            pipeline,  # cross_validate clona el pipeline completo en cada fold
            textos_entrenamiento,
            etiquetas_entrenamiento,
            cv=divisor,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            error_score="raise",
        )
        validacion_cruzada = {
            "folds": folds,
            "accuracy": _resumir(resultados["test_accuracy"]),
            "f1_macro": _resumir(resultados["test_f1_macro"]),
        }

    # 4. Ajustar el pipeline final SOLO con los datos de entrenamiento
    if al_avanzar is not None:
        al_avanzar("Extrayendo características y entrenando el modelo final…")
    pipeline.fit(textos_entrenamiento, etiquetas_entrenamiento)

    # 5. Predecir la parte de prueba con el pipeline ya entrenado
    if al_avanzar is not None:
        al_avanzar("Evaluando los textos reservados…")
    inicio = perf_counter()
    etiquetas_predichas = pipeline.predict(textos_prueba)
    ms_por_texto = (perf_counter() - inicio) * 1000 / len(textos_prueba)

    resultado = {
        "pipeline": pipeline,
        "etiquetas_reales": list(etiquetas_prueba),
        "etiquetas_predichas": list(etiquetas_predichas),
        "validacion_cruzada": validacion_cruzada,
    }
    if incluir_detalles_prueba:
        resultado["textos_prueba"] = list(textos_prueba)
        resultado["inferencia_ms_por_texto"] = ms_por_texto
    return resultado
