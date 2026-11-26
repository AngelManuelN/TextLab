import numpy as np
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from textlab.caracteristicas import crear_vectorizador

FOLDS_PERMITIDOS = (3, 5, 10)


def _resumir(valores):
    """Valores por fold, promedio y desviación estándar (poblacional)."""
    valores = [float(v) for v in valores]
    return {
        "por_fold": valores,
        "promedio": float(np.mean(valores)),
        "desviacion": float(np.std(valores)),
    }


def entrenar_modelo(
    textos,
    etiquetas,
    proporcion_prueba=0.20,
    ngram_range=(1, 1),
    representacion="tfidf",
    folds=None,
):
    """Entrena una regresión logística con n-gramas y la representación elegida.

    textos: lista de frases utilizables.
    etiquetas: lista de categorías correctas, en el mismo orden que textos.
    proporcion_prueba: fracción reservada para evaluar (0.20 = 20 %).
    ngram_range: tupla (mínimo, máximo) de palabras por n-grama; (1, 1) = unigramas.
    representacion: "tfidf", "frecuencia" o "binaria".
    folds: None (sin validación cruzada), 3, 5 o 10. La validación cruzada
        estratificada usa solo la parte de entrenamiento.

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
                ),
            ),
            ("clasificador", LogisticRegression()),
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
    pipeline.fit(textos_entrenamiento, etiquetas_entrenamiento)

    # 5. Predecir la parte de prueba con el pipeline ya entrenado
    etiquetas_predichas = pipeline.predict(textos_prueba)

    return {
        "pipeline": pipeline,
        "etiquetas_reales": list(etiquetas_prueba),
        "etiquetas_predichas": list(etiquetas_predichas),
        "validacion_cruzada": validacion_cruzada,
    }
