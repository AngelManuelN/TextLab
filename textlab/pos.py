"""Extracción de POS y palabras de contenido en español con spaCy."""

from functools import lru_cache


_CATEGORIAS_CONTENIDO = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


@lru_cache(maxsize=1)
def _cargar_modelo():
    """Carga el modelo una vez y lo reutiliza en llamadas siguientes."""
    try:
        import spacy
    except ImportError as error:
        raise RuntimeError(
            "spaCy no está instalado. Instálalo con 'python -m pip install spacy' "
            "y luego descarga el modelo con 'python -m spacy download es_core_news_sm'."
        ) from error

    try:
        return spacy.load("es_core_news_sm")
    except OSError as error:
        raise RuntimeError(
            "No se encontró el modelo de español 'es_core_news_sm'. "
            "Instálalo con 'python -m spacy download es_core_news_sm'."
        ) from error


def extraer_pos(texto):
    """Devuelve las etiquetas POS de las palabras de una frase en español.

    Omite los espacios y los signos de puntuación. Un texto vacío o formado
    únicamente por espacios produce una lista vacía.
    """
    if not texto or texto.isspace():
        return []

    doc = _cargar_modelo()(texto)
    return [token.pos_ for token in doc if not token.is_space and not token.is_punct]


def extraer_palabras_contenido(texto):
    """Devuelve sustantivos, verbos, adjetivos y adverbios en orden original."""
    if not texto or texto.isspace():
        return []

    doc = _cargar_modelo()(texto)
    return [
        token.text
        for token in doc
        if not token.is_space
        and not token.is_punct
        and token.pos_ in _CATEGORIAS_CONTENIDO
    ]


def extraer_stopwords(texto):
    """Devuelve tokens que spaCy identifica como stopwords, en orden original."""
    if not texto or texto.isspace():
        return []

    doc = _cargar_modelo()(texto)
    return [
        token.text
        for token in doc
        if not token.is_space and not token.is_punct and token.is_stop
    ]
