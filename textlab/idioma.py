from itertools import islice

from langdetect import DetectorFactory, LangDetectException, detect_langs


DetectorFactory.seed = 42


def detectar_idioma(textos):
    """Estima el idioma predominante de una muestra de entrenamiento."""
    muestra = " ".join(texto[:300] for texto in islice(textos, 80))
    if sum(caracter.isalpha() for caracter in muestra) < 40:
        return None
    try:
        candidatos = detect_langs(muestra)
    except LangDetectException:
        return None
    mejor = candidatos[0]
    return mejor.lang if mejor.prob >= 0.85 else None
