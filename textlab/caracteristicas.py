"""Vectorizadores de características de texto para TextLab."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def crear_vectorizador(ngram_range=(1, 1), representacion="tfidf"):
    """Crea un vectorizador de texto sin ajustar.

    Args:
        ngram_range: Rango inclusivo de n-gramas; (1, 1) usa unigramas,
            (2, 2) bigramas y (3, 3) trigramas.
        representacion: "tfidf", "frecuencia" o "binaria".

    Conserva tildes y ñ. Las stopwords no se eliminan. El vectorizador se
    devuelve sin ajustar para que el Pipeline aprenda solo del entrenamiento.
    """
    if representacion == "tfidf":
        return TfidfVectorizer(
            ngram_range=ngram_range,
            strip_accents=None,
            stop_words=None,
        )

    if representacion == "frecuencia":
        return CountVectorizer(
            ngram_range=ngram_range,
            strip_accents=None,
            stop_words=None,
        )

    if representacion == "binaria":
        return CountVectorizer(
            ngram_range=ngram_range,
            binary=True,
            strip_accents=None,
            stop_words=None,
        )

    raise ValueError(
        "Representación desconocida. Usa 'tfidf', 'frecuencia' o 'binaria'."
    )
