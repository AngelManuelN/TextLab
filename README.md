# TextLab · manuel_solutions

Aplicación web para entrenar, comparar y usar clasificadores supervisados de textos desde archivos CSV. Las clases se obtienen del archivo que se carga.

## Ejecutar en local

Se recomienda Python 3.12. Las dependencias incluyen Streamlit, Pandas, Plotly, scikit-learn y los modelos de spaCy para español, inglés y francés.

```sh
python -m venv .venv
```

Activa el entorno con `.\.venv\Scripts\Activate.ps1` en PowerShell o `source .venv/bin/activate` en macOS y Linux. Luego ejecuta:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abre [TextLab en local](http://localhost:8501). Para publicar en Streamlit Community Cloud, usa la rama `entrega/textlab`, el archivo `app.py` y Python 3.12.

## Usar TextLab

1. Carga un CSV y selecciona las columnas de texto y etiqueta. Revisa la distribución de clases y las filas excluidas.
2. Elige palabras, POS, stopwords o content words. Combina unigramas, bigramas, trigramas o unigramas y bigramas juntos con frecuencia, binaria o TF-IDF. Selecciona el idioma del CSV para POS, stopwords o palabras de contenido. El modo Palabras no requiere un analizador de idioma.
3. Selecciona regresión logística, SVM, Random Forest o Naive Bayes. Ajusta la proporción de prueba, los parámetros del clasificador y la validación cruzada estratificada. Se admiten 3, 5 y 10 folds o desactivarla. Cada partición ajusta las características con sus datos de entrenamiento.
4. Entrena y revisa accuracy, precisión, recall y F1, la matriz de confusión, las gráficas por clase y los resultados de validación cruzada. Los experimentos se conservan durante la sesión.
5. Selecciona un modelo entrenado y escribe un texto o sube un CSV sin etiquetas. Puedes descargar las predicciones por lote.

Si la validación pide más ejemplos por clase, reduce los folds o carga más datos. Si el vocabulario está vacío, elige un tamaño de n-grama menor o un modo que conserve más palabras.

Cada sesión conserva sus propios CSV, experimentos y modelos entrenados. En una instalación compartida, todas las personas ven la misma aplicación. Para recuperar resultados fuera de tu sesión, puedes descargar las predicciones por lote.

Los CSV admiten UTF-8, UTF-16 o UTF-32 con marca de codificación y Windows-1252.

## Datos de prueba

- `datos/pruebas/carga_valida.csv` contiene frases etiquetadas de sentimiento.
- `datos/pruebas/spam.csv` contiene mensajes de spam y legítimos con columnas personalizadas.
- `datos/pruebas/carga_con_errores.csv` sirve para revisar los avisos de calidad de datos.
- `datos/pruebas/prediccion_sin_etiquetas.csv` contiene tres textos para predicción por lote.
- El [dataset de desarrollo](https://huggingface.co/datasets/Kenpache/multilingual-financial-sentiment) se usa filtrando `language = es`, con `sentence` como texto y `label` como etiqueta.
