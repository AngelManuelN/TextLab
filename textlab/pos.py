from functools import lru_cache

MODELOS = {"es": "es_core_news_sm", "en": "en_core_web_sm", "fr": "fr_core_news_sm"}
_CATEGORIAS_CONTENIDO = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


@lru_cache(maxsize=len(MODELOS))
def _cargar_modelo(idioma="es"):
    if idioma not in MODELOS:
        raise ValueError(f"No hay un analizador lingüístico configurado para {idioma!r}.")
    nombre = MODELOS[idioma]
    try:
        import spacy
        return spacy.load(nombre, exclude=["parser", "ner", "lemmatizer"])
    except (ImportError, OSError) as error:
        raise RuntimeError(f"Instala las dependencias para usar el modelo {nombre}.") from error


def _tokens(texto, idioma):
    if not texto or not texto.strip():
        return []
    return [token for token in _cargar_modelo(idioma)(texto) if not token.is_space and not token.is_punct]


def extraer_pos(texto, idioma="es"):
    return [token.pos_ for token in _tokens(texto, idioma)]


def extraer_palabras_contenido(texto, idioma="es"):
    return [token.text.lower() for token in _tokens(texto, idioma) if token.pos_ in _CATEGORIAS_CONTENIDO]


def extraer_stopwords(texto, idioma="es"):
    return [token.text.lower() for token in _tokens(texto, idioma) if token.is_stop]
