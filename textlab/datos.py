"""Funciones para revisar datos de archivos CSV de TextLab."""

import pandas as pd


def perfilar_dataset(df, columna_texto, columna_etiqueta):
    """Resume faltantes, duplicados y etiquetas de un DataFrame.

    Args:
        df: DataFrame de pandas con los datos del CSV.
        columna_texto: Nombre de la columna que contiene los textos.
        columna_etiqueta: Nombre de la columna que contiene las etiquetas.

    Returns:
        Diccionario con el total de filas, valores faltantes, filas duplicadas
        y conteos de las etiquetas válidas.

    Raises:
        ValueError: Si una columna no existe o si se indica la misma columna
            para texto y etiqueta.
    """
    if columna_texto == columna_etiqueta:
        raise ValueError(
            "La columna de texto y la columna de etiqueta deben ser distintas."
        )

    columnas_faltantes = [
        nombre
        for nombre in (columna_texto, columna_etiqueta)
        if nombre not in df.columns
    ]
    if columnas_faltantes:
        raise ValueError(
            "No existe la columna solicitada: "
            + ", ".join(repr(nombre) for nombre in columnas_faltantes)
            + "."
        )

    textos = df[columna_texto]
    etiquetas = df[columna_etiqueta]

    textos_vacios = textos.map(
        lambda valor: isinstance(valor, str) and not valor.strip()
    )
    etiquetas_vacias = etiquetas.map(
        lambda valor: isinstance(valor, str) and not valor.strip()
    )

    etiquetas_validas = etiquetas[etiquetas.notna() & ~etiquetas_vacias]
    distribucion = etiquetas_validas.value_counts(sort=False).to_dict()

    return {
        "total_filas": int(len(df)),
        "textos_faltantes": int((textos.isna() | textos_vacios).sum()),
        "etiquetas_faltantes": int((etiquetas.isna() | etiquetas_vacias).sum()),
        "filas_duplicadas": int(df.duplicated().sum()),
        "distribucion_clases": distribucion,
    }
