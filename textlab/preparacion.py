import pandas as pd


def _es_faltante(serie):
    """True donde la celda esta vacia, es nula o solo tiene espacios."""
    return serie.isna() | serie.astype(str).str.strip().eq("")


def preparar_dataset(df, columna_texto, columna_etiqueta):
    df = df.copy()

    # 1. Aparta filas donde falta el texto o la etiqueta
    sin_datos = _es_faltante(df[columna_texto]) | _es_faltante(df[columna_etiqueta])
    filas_sin_datos = int(sin_datos.sum())
    df = df[~sin_datos]

    # 2. Aparta todas las filas de un texto con etiquetas contradictorias
    etiquetas_distintas = df.groupby(columna_texto)[columna_etiqueta].transform("nunique")
    en_conflicto = etiquetas_distintas > 1
    filas_en_conflicto = int(en_conflicto.sum())
    df = df[~en_conflicto]

    # 3. De los pares identicos (texto, etiqueta) conserva una sola copia
    duplicadas = df.duplicated(subset=[columna_texto, columna_etiqueta], keep="first")
    filas_duplicadas = int(duplicadas.sum())
    df = df[~duplicadas]

    df = df.reset_index(drop=True)
    conteos = {
        "filas_sin_datos": filas_sin_datos,
        "filas_en_conflicto": filas_en_conflicto,
        "filas_duplicadas": filas_duplicadas,
        "filas_utilizables": len(df),
    }
    return df, conteos
