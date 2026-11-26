from collections import Counter
import math


def validar_configuracion_entrenamiento(etiquetas, proporcion_prueba, folds=None):
    # Revisa si la configuración sirve antes de entrenar; [] significa que todo está bien.
    avisos = []
    lista_etiquetas = list(etiquetas)

    # Sin datos no se puede entrenar.
    if len(lista_etiquetas) == 0:
        return [
            "No hay datos utilizables para entrenar. Revisa que tu archivo tenga textos y etiquetas válidas."
        ]

    # La prueba debe ser un porcentaje válido entre 0 y 1 (ej. 0.20).
    if proporcion_prueba <= 0 or proporcion_prueba >= 1:
        return [
            f"La proporción de prueba ({proporcion_prueba}) no es válida. Usa un número entre 0 y 1, como 0.20 (20%)."
        ]

    conteo = Counter(lista_etiquetas)
    clases = list(conteo.keys())

    # Hace falta al menos 2 clases para clasificar.
    if len(clases) < 2:
        return [
            f"Solo se encontró una clase ('{clases[0]}'). Se necesitan al menos 2 clases distintas para entrenar; agrega o selecciona datos con más clases."
        ]

    # Cada clase necesita mínimo 2 ejemplos para repartir entre entrenar y probar.
    clases_con_muy_pocos = []
    for clase, total in conteo.items():
        if total < 2:
            clases_con_muy_pocos.append(f"'{clase}' (solo {total})")

    if clases_con_muy_pocos:
        return [
            "No alcanza para repartir bien los datos porque estas clases tienen menos de 2 ejemplos: "
            + ", ".join(clases_con_muy_pocos)
            + ". Consigue más ejemplos de esas clases o ajusta tus datos."
        ]

    # Calcula cuántos ejemplos de cada clase quedan en entrenamiento.
    en_entrenamiento_por_clase = {}
    for clase, total in conteo.items():
        para_prueba = int(math.floor(total * proporcion_prueba))
        if para_prueba < 1:
            para_prueba = 1
        if para_prueba >= total:
            para_prueba = total - 1

        en_entrenamiento_por_clase[clase] = total - para_prueba

    # Si hay folds activos, revisa que cada clase alcance en entrenamiento.
    if folds is not None:
        if folds < 2:
            return [
                f"El número de folds ({folds}) debe ser de al menos 2, o bien desactiva los folds."
            ]

        for clase, cantidad_train in en_entrenamiento_por_clase.items():
            if cantidad_train < folds:
                avisos.append(
                    f"La clase '{clase}' tendría más o menos {cantidad_train} ejemplo(s) en entrenamiento, "
                    f"y eso no alcanza para {folds} folds. "
                    f"Puedes usar menos folds (como 3), quitar los folds, bajar el porcentaje de prueba o agregar más ejemplos de '{clase}'."
                )

    return avisos