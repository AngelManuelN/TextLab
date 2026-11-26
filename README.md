# TextLab · manuel_solutions

Aplicación web para entrenar, comparar y usar clasificadores supervisados de textos desde archivos CSV. Las clases se obtienen del archivo que se carga.

[Abrir TextLab](https://textlab-manuel-solutions.streamlit.app/) · [Tablero de seguimiento](https://github.com/users/AngelManuelN/projects/1)

## Ejecutar en local

Se recomienda Python 3.12. Las dependencias incluyen Streamlit, Pandas, Plotly, scikit-learn, langdetect y los modelos de spaCy para español, inglés y francés.

```sh
python -m venv .venv
```

Activa el entorno con `.\.venv\Scripts\Activate.ps1` en PowerShell o `source .venv/bin/activate` en macOS y Linux. Luego ejecuta:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abre [TextLab en local](http://localhost:8501). Para publicar en Streamlit Community Cloud, usa la rama `entrega/textlab`, el archivo `app.py` y Python 3.12.

## Guía de configuración

### Primer experimento

1. Sube el archivo original y elige la columna que contiene los textos y la que contiene las etiquetas. Los nombres pueden variar. Si ves una sola columna o la vista previa se ve mal, abre **Formato de importación** para ajustar separador, encabezados y comillas. También se admiten TSV y TXT delimitados. Las codificaciones admitidas son UTF-8, UTF-16 y UTF-32 con BOM, y Windows-1252.
2. Revisa **Explorador de datos** para ver las clases, los datos faltantes y los duplicados. La barra lateral indica cuántas filas se usarán y por qué se excluyeron las demás.
3. Pulsa **Configuración rápida** para usar SVM lineal, Palabras, unigramas, TF-IDF, 20 % de prueba y validación cruzada desactivada. Luego pulsa **Entrenar** y espera el mensaje **Modelo entrenado**.
4. Abre **Resultados** para ver las métricas y la matriz de confusión. En **Análisis de errores** puedes leer qué textos se clasificaron mal. **Experimentos** conserva la configuración y métricas de cada ejecución durante la sesión.
5. En **Predicción**, elige un modelo entrenado. Puedes escribir un texto o subir un CSV con textos nuevos y descargar las clases predichas.

### Qué cambia cada ajuste

| Ajuste | Cómo usarlo |
| --- | --- |
| Clasificador | SVM lineal y regresión logística usan `C`. Un `C` menor aplica más regularización. Random Forest permite elegir cuántos árboles entrenar. Naive Bayes usa `alpha` para suavizar los conteos. |
| Características | **Palabras** usa términos del texto. **POS** usa categorías gramaticales. **Stopwords** conserva las palabras frecuentes detectadas. **Palabras de contenido** conserva categorías como sustantivos, verbos y adjetivos. |
| Idioma | Los modos lingüísticos detectan español, inglés o francés y permiten corregir la selección. Para datos en otros idiomas puedes probar **Palabras**. |
| N-gramas | Unigramas son elementos individuales, bigramas son pares y trigramas son grupos de tres. Por ejemplo, «muy buen servicio» es un trigrama de palabras. |
| Representación | **Frecuencia** cuenta apariciones, **Binaria** marca presencia y **TF-IDF** reduce el peso de términos comunes en muchos textos. |
| División de prueba | Con 20 %, el modelo se entrena con 80 % de las filas utilizables y se evalúa con el 20 % reservado. |
| Validación cruzada | Usa 3, 5 o 10 folds para comparar configuraciones dentro del conjunto de entrenamiento. Empieza por 3 si quieres medir estabilidad. Cada fold ajusta su propio vocabulario y sus pesos, usando solo su partición de entrenamiento. |

Si cambias un ajuste, vuelve a pulsar **Entrenar** para aplicarlo. El modelo anterior sigue disponible en la sesión. Si un fold necesita más ejemplos por clase, reduce el número de folds o usa más datos. Si el vocabulario queda vacío, prueba unigramas o el modo **Palabras**.

## Datos de prueba

- `datos/pruebas/carga_valida.csv` contiene frases etiquetadas de sentimiento.
- `datos/pruebas/spam.csv` contiene mensajes de spam y legítimos con columnas personalizadas.
- `datos/pruebas/carga_con_errores.csv` sirve para revisar los avisos de calidad de datos.
- `datos/pruebas/prediccion_sin_etiquetas.csv` contiene tres textos para predicción por lote.
- El [dataset de desarrollo](https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment) se usa filtrando `language = es`, con `sentence` como texto y `label` como etiqueta.
