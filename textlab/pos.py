"""Extracción de etiquetas gramaticales POS en español con spaCy."""

from functools import lru_cache


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
