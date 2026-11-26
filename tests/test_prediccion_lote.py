import csv
from pathlib import Path
import sys

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
if str(RAIZ_PROYECTO) not in sys.path:
    sys.path.insert(0, str(RAIZ_PROYECTO))

from textlab.prediccion import predecir_lote


def _cargar_frases_csv(ruta_csv: Path) -> list[str]:
    with ruta_csv.open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        assert lector.fieldnames == ["texto"]
        return [fila["texto"] for fila in lector]


def test_prediccion_por_lote_respeta_cantidad_y_orden():
    ruta_csv = RAIZ_PROYECTO / "datos/pruebas/prediccion_sin_etiquetas.csv"
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


def test_cantidad_incorrecta_de_etiquetas_lanza_error_claro():
    frases = [
        "La empresa aumentó sus ganancias",
        "El banco reportó pérdidas",
        "El informe se publicará el lunes",
    ]

    def predictor_incompleto(textos: list[str]) -> list[str]:
        assert textos == frases
        return ["positiva", "negativa"]

    try:
        predecir_lote(frases, predictor_incompleto)
    except ValueError as exc:
        assert "no coincide" in str(exc)
        return str(exc)

    raise AssertionError("Se esperaba un ValueError por cantidad incorrecta de etiquetas")


if __name__ == "__main__":
    salida = test_prediccion_por_lote_respeta_cantidad_y_orden()
    error_msg = test_cantidad_incorrecta_de_etiquetas_lanza_error_claro()
    print(f"OK: {len(salida)} predicciones en orden correcto -> {salida}")
    print(f"OK: Error claro capturado -> {error_msg}")
