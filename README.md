# Taller Week 6: Encoder-Decoder con Atención para CNN/DailyMail

Este proyecto adapta el notebook de referencia:

`../applied-ml-course/Week6/Code/5-Encoder_Decoder_+_Atención_+_Pre_trained_Embeddings.ipynb`

El notebook original entrena una arquitectura Encoder-Decoder con atención y embeddings preentrenados para traducción inglés-español. En este taller se reutiliza esa idea para una tarea distinta de NLP: resumen automático de noticias usando la base de datos `cnn_dailymail` de Hugging Face.

## Objetivo

Entrenar una arquitectura propia Encoder-Decoder + Atención + pretrained embeddings que reciba un artículo de noticia como entrada y genere un resumen breve como salida.

## Dataset: `cnn_dailymail`

`cnn_dailymail` es una base de datos de Hugging Face construida con artículos periodísticos de CNN y Daily Mail. Cada ejemplo contiene, de forma general:

- `article`: texto completo de la noticia.
- `highlights`: resumen o puntos principales del artículo.
- `id`: identificador del ejemplo.

Un modelo entrenado con esta base de datos permite resolver un problema de **resumen automático de texto**. En particular, aprende a transformar documentos largos en versiones cortas que conserven las ideas principales. Esto es útil para resumir noticias, reportes, documentos extensos o contenido informativo que un usuario necesita revisar rapidamente.

## Relación con el notebook Week 6

El notebook de referencia usa:

- Tokenización de textos de entrada y salida.
- Secuencias con longitud máxima.
- Embeddings preentrenados.
- Encoder LSTM.
- Decoder LSTM.
- Mecanismo de atención.
- Inferencia autoregresiva para generar texto.

Para este taller, el cambio principal es el dominio de la tarea:

- Antes: entrada en inglés y salida en español.
- Ahora: entrada `article` y salida `highlights`.

La arquitectura general se mantiene, pero se deben ajustar el preprocesamiento, los tamaños máximos de secuencia, los vocabularios, los tokens especiales del decoder y las métricas de evaluación.

## Carga del dataset

La base de datos se puede cargar desde Hugging Face con `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset["train"]
validation_data = dataset["validation"]
test_data = dataset["test"]
```

Para entrenar en un computador local, se recomienda comenzar con una muestra reducida:

```python
train_sample = train_data.select(range(10000))
validation_sample = validation_data.select(range(1000))
test_sample = test_data.select(range(1000))
```

## Ejecución en Google Colab

1. Sube `taller_week6_cnn_dailymail.ipynb` a Google Colab.
2. En Colab, activa GPU desde `Runtime` -> `Change runtime type` -> `T4 GPU`.
3. Ejecuta la celda `0. Colab setup` antes de las demás celdas. Esa celda instala `datasets`, `evaluate` y `rouge_score`, descarga GloVe en `data/glove/` y muestra la GPU disponible.
4. Si quieres conservar el modelo después de cerrar Colab, cambia `USE_GOOGLE_DRIVE = True` en la celda de setup. El notebook guardará el modelo en `/content/drive/MyDrive/ml-nlp-project/model/`.
5. Ejecuta el resto del notebook en orden.

Con la configuración actual (`N_TRAIN = 10_000`, `N_VAL = 1_000`, `epochs = 10`), el entrenamiento puede tardar varios minutos en GPU y bastante más en CPU. Si necesitas una prueba rápida, baja temporalmente `N_TRAIN`, `N_VAL` o `epochs`.

## Flujo esperado del notebook

1. Instalar e importar dependencias: `datasets`, `tensorflow`, `numpy`, `pandas`, `spacy` o una fuente equivalente de embeddings.
2. Cargar `cnn_dailymail` desde Hugging Face.
3. Explorar ejemplos de `article` y `highlights`.
4. Limpiar y normalizar textos.
5. Agregar tokens especiales al resumen, por ejemplo `<start>` y `<end>`.
6. Tokenizar artículos y resúmenes.
7. Definir longitudes máximas razonables para entrada y salida.
8. Construir matrices de embeddings preentrenados.
9. Implementar Encoder-Decoder con atención.
10. Entrenar el modelo.
11. Generar resúmenes sobre ejemplos de validación o prueba.
12. Evaluar resultados con ejemplos cualitativos y, si es posible, ROUGE.

## Resultado esperado

El entregable debe mostrar:

- Una breve consulta o descripción de la base de datos `cnn_dailymail`.
- El dataset cargado directamente desde Hugging Face.
- El modelo Encoder-Decoder + Atención + pretrained embeddings entrenado para resumen automático.
- Ejemplos de artículos, resumen real y resumen generado por el modelo.
- Una conclusión corta sobre la calidad de los resultados y las limitaciones observadas.
