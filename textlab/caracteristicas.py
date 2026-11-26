"""Vectorizadores de características de texto para TextLab."""

from sklearn.feature_extraction.text import TfidfVectorizer


def crear_vectorizador(ngram_range=(1, 1)):
    """Crea un vectorizador TF-IDF para el rango de n-gramas indicado.

    Args:
        ngram_range: Rango inclusivo de n-gramas. El valor (1, 1) usa
            unigramas; (2, 2) usa bigramas; y (3, 3) usa trigramas.

    El vectorizador conserva tildes y la ñ, y no elimina stopwords. Se devuelve
    sin ajustar para que el Pipeline aprenda vocabulario y pesos TF-IDF solo
    con los textos de entrenamiento de cada partición.
    """
    return TfidfVectorizer(
        ngram_range=ngram_range,
        strip_accents=None,
        stop_words=None,
    )
