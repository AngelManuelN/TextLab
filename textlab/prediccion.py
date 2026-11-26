from collections.abc import Callable, Sequence
from typing import Any


def predecir_lote(
    frases: Sequence[str],
    predictor: Callable[[list[str]], Any],
) -> list[dict[str, Any]]:
    """Ejecuta predicción por lote respetando el orden de las frases."""
    lista_frases = list(frases)
    if not lista_frases:
        return []

    predicciones_crudas = predictor(lista_frases)
    etiquetas = list(predicciones_crudas)

    if len(etiquetas) != len(lista_frases):
        raise ValueError(
            f"Cantidad de etiquetas ({len(etiquetas)}) no coincide con la cantidad de frases ({len(lista_frases)})."
        )

    return [
        {"texto": frase, "etiqueta": etiqueta}
        for frase, etiqueta in zip(lista_frases, etiquetas)
    ]
