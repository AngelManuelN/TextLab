"""Vectorizadores de características de texto para TextLab."""

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .pos import extraer_palabras_contenido, extraer_pos, extraer_stopwords


_EXTRACTORES_TEXTO = {
    "pos": extraer_pos,
    "stopwords": extraer_stopwords,
    "palabras_contenido": extraer_palabras_contenido,
}


def crear_vectorizador(
    ngram_range=(1, 1),
    representacion="tfidf",
    modo_texto="palabras",
):
    """Crea un vectorizador sin ajustar para usar dentro de un Pipeline.

    Args:
        ngram_range: Rango inclusivo de n-gramas; por ejemplo, (1, 1) para
            unigramas y (2, 2) para bigramas.
        representacion: "tfidf", "frecuencia" o "binaria".
        modo_texto: "palabras", "pos", "stopwords" o "palabras_contenido".

    En los modos POS, stopwords y palabras_contenido, el extractor devuelve
    los tokens y scikit-learn aplica ngram_range sobre esa secuencia.
    Conserva tildes y ñ, y no elimina stopwords de forma adicional.
    """
    if modo_texto == "palabras":
        opciones_tokens = {}
    elif modo_texto in _EXTRACTORES_TEXTO:
        opciones_tokens = {
            "tokenizer": _EXTRACTORES_TEXTO[modo_texto],
            "token_pattern": None,
            "lowercase": False,
        }
    else:
        raise ValueError(
            "Modo de texto desconocido. Usa 'palabras', 'pos', 'stopwords' "
            "o 'palabras_contenido'."
        )

    opciones = {
        "ngram_range": ngram_range,
        "strip_accents": None,
        "stop_words": None,
        **opciones_tokens,
    }
    if representacion == "tfidf":
        return TfidfVectorizer(**opciones)
    if representacion == "frecuencia":
        return CountVectorizer(**opciones)
    if representacion == "binaria":
        return CountVectorizer(binary=True, **opciones)

    raise ValueError(
        "Representación desconocida. Usa 'tfidf', 'frecuencia' o 'binaria'."
    )
