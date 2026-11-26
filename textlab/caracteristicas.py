"""Vectorizadores de características de texto para TextLab."""

from sklearn.feature_extraction.text import TfidfVectorizer


def crear_vectorizador():
    """Crea un vectorizador TF-IDF configurado para usar unigramas.

    El vectorizador conserva tildes y la ñ, y no elimina stopwords. Se devuelve
    sin ajustar para que el Pipeline de entrenamiento aprenda el vocabulario
    y los pesos TF-IDF solo con los textos de entrenamiento de cada partición.
    """
    return TfidfVectorizer(
        ngram_range=(1, 1),
        strip_accents=None,
        stop_words=None,
    )
