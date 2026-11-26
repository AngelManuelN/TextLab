from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from textlab.caracteristicas import crear_vectorizador


def entrenar_modelo(
    textos,
    etiquetas,
    proporcion_prueba=0.20,
    ngram_range=(1, 1),
    representacion="tfidf",
):
    """Entrena una regresión logística con n-gramas y la representación elegida.

    textos: lista de frases utilizables.
    etiquetas: lista de categorías correctas, en el mismo orden que textos.
    proporcion_prueba: fracción reservada para evaluar (0.20 = 20 %).
    ngram_range: tupla (mínimo, máximo) de palabras por n-grama; (1, 1) = unigramas.
    representacion: "tfidf", "frecuencia" o "binaria".

    Devuelve un diccionario con:
      - pipeline: vectorizador + clasificador, ya entrenado.
      - etiquetas_reales: categorías correctas de la parte de prueba.
      - etiquetas_predichas: lo que el modelo predijo para esa misma parte.
    """
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

    # 3. Ajustar SOLO con los datos de entrenamiento
    pipeline.fit(textos_entrenamiento, etiquetas_entrenamiento)

    # 4. Predecir la parte de prueba con el pipeline ya entrenado
    etiquetas_predichas = pipeline.predict(textos_prueba)

    return {
        "pipeline": pipeline,
        "etiquetas_reales": list(etiquetas_prueba),
        "etiquetas_predichas": list(etiquetas_predichas),
    }