from sklearn.model_selection import train_test_split

from textlab.entrenamiento import entrenar_modelo

DEPORTES = [
    "el equipo ganó el partido de futbol anoche",
    "el delantero metió dos goles en la final",
    "la corredora rompió su marca en el maratón",
    "el entrenador cambió la alineación del equipo",
    "el torneo de tenis empieza este fin de semana",
    "el portero atajó un penal en el último minuto",
    "los jugadores entrenan todos los días en la cancha",
    "la selección clasificó al mundial de basquetbol",
    "el ciclista subió la montaña en tiempo récord",
    "el árbitro anuló el gol por fuera de lugar",
    "la nadadora ganó la medalla en la competencia",
    "el estadio estuvo lleno para ver el partido",
]

COCINA = [
    "mezcla la harina con huevos y un poco de leche",
    "hornea el pastel durante cuarenta minutos",
    "corta la cebolla y el ajo para el sofrito",
    "agrega sal y pimienta a la sopa de verduras",
    "la receta lleva tomate, chile y mucho cilantro",
    "fríe las tortillas hasta que queden doradas",
    "hierve el arroz con caldo y una pizca de sal",
    "bate la crema y añade el azúcar poco a poco",
    "marina el pollo con limón antes de asarlo",
    "sirve el guiso caliente con frijoles y queso",
    "amasa la masa y déjala reposar una hora",
    "licúa los jitomates para preparar la salsa",
]


def _datos():
    """24 textos (12 por clase). Cada uno lleva una palabra única (termino0x, termino1x...)."""
    textos, etiquetas = [], []
    for i, frase in enumerate(DEPORTES):
        textos.append(f"{frase} termino{i}x")
        etiquetas.append("deportes")
    for i, frase in enumerate(COCINA, start=len(DEPORTES)):
        textos.append(f"{frase} termino{i}x")
        etiquetas.append("cocina")
    return textos, etiquetas


def test_una_prediccion_por_cada_texto_de_prueba():
    textos, etiquetas = _datos()
    resultado = entrenar_modelo(textos, etiquetas, proporcion_prueba=0.20)

    assert set(resultado) == {"pipeline", "etiquetas_reales", "etiquetas_predichas"}
    assert len(resultado["etiquetas_predichas"]) == len(resultado["etiquetas_reales"])
    # 20 % de 24 textos -> 5 textos de prueba
    assert len(resultado["etiquetas_predichas"]) == 5


def test_pipeline_predice_otro_texto_despues_de_entrenar():
    textos, etiquetas = _datos()
    pipeline = entrenar_modelo(textos, etiquetas)["pipeline"]

    prediccion = pipeline.predict(["el equipo jugó un gran partido de futbol"])

    assert len(prediccion) == 1
    assert prediccion[0] in {"deportes", "cocina"}


def test_palabra_solo_de_prueba_queda_fuera_del_vocabulario():
    textos, etiquetas = _datos()
    resultado = entrenar_modelo(textos, etiquetas, proporcion_prueba=0.20)
    vocabulario = resultado["pipeline"].named_steps["vectorizador"].vocabulary_

    # Repetimos el mismo reparto para saber qué textos quedaron en prueba
    textos_ent, textos_prueba, _, _ = train_test_split(
        textos, etiquetas, test_size=0.20, stratify=etiquetas, random_state=42
    )
    palabras_ent = {p for t in textos_ent for p in t.split() if p.startswith("termino")}
    palabras_prueba = {p for t in textos_prueba for p in t.split() if p.startswith("termino")}

    assert palabras_prueba  # sí hay palabras únicas en la parte de prueba
    for palabra in palabras_prueba:
        assert palabra not in vocabulario
    for palabra in palabras_ent:
        assert palabra in vocabulario
