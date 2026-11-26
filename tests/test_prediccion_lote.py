import csv
from pathlib import Path


def _cargar_frases_csv(ruta_csv: Path) -> list[str]:
    with ruta_csv.open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        assert lector.fieldnames == ["texto"]
        return [fila["texto"] for fila in lector]


def predecir_lote(frases: list[str], predictor) -> list[dict[str, str]]:
    etiquetas = predictor(frases)
    assert len(etiquetas) == len(frases)
    return [
        {"texto": frase, "etiqueta": etiqueta}
        for frase, etiqueta in zip(frases, etiquetas)
    ]


def test_prediccion_por_lote_respeta_cantidad_y_orden():
    ruta_csv = Path("datos/pruebas/prediccion_sin_etiquetas.csv")
    frases = _cargar_frases_csv(ruta_csv)

    etiquetas_fijas = ["positiva", "negativa", "neutra"]

    def predictor_simulado(textos: list[str]) -> list[str]:
        assert textos == frases
        return etiquetas_fijas

    resultados = predecir_lote(frases, predictor_simulado)

    assert len(resultados) == 3
    assert resultados == [
        {"texto": "La empresa aumentó sus ganancias", "etiqueta": "positiva"},
        {"texto": "El banco reportó pérdidas", "etiqueta": "negativa"},
        {"texto": "El informe se publicará el lunes", "etiqueta": "neutra"},
    ]
    return resultados


if __name__ == "__main__":
    salida = test_prediccion_por_lote_respeta_cantidad_y_orden()
    print(f"OK: {len(salida)} predicciones en orden correcto -> {salida}")
