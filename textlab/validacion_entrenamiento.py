from collections import Counter
from sklearn.model_selection import train_test_split


FOLDS_PERMITIDOS = (None, 3, 5, 10)


def validar_configuracion_entrenamiento(
    etiquetas,
    proporcion_prueba,
    folds=None,
    semilla=42,
):
    """Valida clases, división estratificada y folds antes de entrenar."""
    avisos = []
    etiquetas = list(etiquetas)

    # Revisa que haya etiquetas disponibles
    if not etiquetas:
        avisos.append(
            "No hay etiquetas disponibles para entrenar. Agrega o selecciona datos válidos antes de continuar."
        )
        return avisos

    # Revisa que existan al menos 2 clases distintas
    conteo_total = Counter(etiquetas)
    if len(conteo_total) < 2:
        clase_unica = next(iter(conteo_total))
        avisos.append(
            f"Solo se encontró una clase ('{clase_unica}'). Se necesitan al menos 2 clases distintas para entrenar; agrega o selecciona datos con más clases."
        )
        return avisos

    # Revisa que la proporción de prueba esté entre 0 y 1 (exclusivo)
    if not isinstance(proporcion_prueba, (int, float)) or isinstance(proporcion_prueba, bool) or not (0 < proporcion_prueba < 1):
        avisos.append(
            "La proporción de prueba debe ser un número mayor que 0 y menor que 1 (por ejemplo, 0.2 para 20 %). Ajusta ese valor antes de entrenar."
        )
        return avisos

    # Revisa que folds sea uno de los valores permitidos
    if folds not in FOLDS_PERMITIDOS:
        avisos.append(
            "El número de folds no es válido. Usa None, 3, 5 o 10, o quita la validación cruzada."
        )
        return avisos

    # Realiza la división estratificada real con semilla provisional
    indices = list(range(len(etiquetas)))
    try:
        indices_entrenamiento, indices_prueba = train_test_split(
            indices,
            test_size=proporcion_prueba,
            stratify=etiquetas,
            random_state=semilla,
        )
    except ValueError:
        avisos.append(
            f"No se pudo hacer la división estratificada con una proporción de prueba de {proporcion_prueba} para {len(etiquetas)} ejemplos y {len(conteo_total)} clases (por ejemplo, hay muy pocos lugares en prueba para cubrir las clases). Ajusta el porcentaje de prueba o agrega más ejemplos."
        )
        return avisos

    # Cuenta cuántas muestras de cada clase quedaron realmente en entrenamiento
    etiquetas_entrenamiento = [etiquetas[i] for i in indices_entrenamiento]
    conteo_entrenamiento = Counter(etiquetas_entrenamiento)

    conteo_prueba = Counter(etiquetas[i] for i in indices_prueba)
    if conteo_entrenamiento.keys() != conteo_total.keys() or conteo_prueba.keys() != conteo_total.keys():
        return [
            "Alguna clase quedó fuera de entrenamiento o de prueba. "
            "Ajusta la proporción de prueba o agrega más ejemplos de esa clase."
        ]


    # Si se pidieron folds, verifica que cada clase tenga suficientes ejemplos en entrenamiento
    if folds is not None:
        for clase, cantidad_train in conteo_entrenamiento.items():
            if cantidad_train < folds:
                # Sugiere un número de folds menor que sí alcance
                sugeridos = [f for f in (10, 5, 3) if f < folds and f <= min(conteo_entrenamiento.values())]
                if sugeridos:
                    sugerencia_folds = f"usar {sugeridos[0]} folds"
                else:
                    sugerencia_folds = "quitar los folds (dejarlo en None)"
                avisos.append(
                    f"La clase '{clase}' tiene {cantidad_train} ejemplo(s) en entrenamiento, lo cual no alcanza para {folds} folds. Puedes {sugerencia_folds}, ajustar el porcentaje de prueba o agregar más ejemplos de '{clase}'."
                )

    return avisos
