# TODO - Taller Week 6 CNN/DailyMail

Referencia base:

`../applied-ml-course/Week6/Code/5-Encoder_Decoder_+_Atención_+_Pre_trained_Embeddings.ipynb`

## 1. Consulta sobre `cnn_dailymail`

- [ ] Investigar la base de datos `cnn_dailymail` en Hugging Face.
- [ ] Identificar sus columnas principales: `article`, `highlights`, `id`.
- [ ] Explicar que el problema a resolver es resumen automático de texto.
- [ ] Escribir una respuesta corta en el notebook sobre aplicaciones posibles del modelo.

## 2. Carga del dataset

- [ ] Instalar `datasets` si no esta disponible.
- [ ] Cargar el dataset con:

```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
```

- [ ] Revisar los splits disponibles: `train`, `validation`, `test`.
- [ ] Mostrar al menos un ejemplo de `article` y `highlights`.
- [ ] Crear una muestra pequena para entrenamiento inicial si el equipo no soporta el dataset completo.

## 3. Adaptación del preprocesamiento

- [ ] Reutilizar la estructura del notebook Week 6.
- [ ] Cambiar el par entrada-salida:
  - entrada: `article`
  - salida: `highlights`
- [ ] Limpiar textos: minusculas, espacios repetidos, caracteres innecesarios.
- [ ] Agregar tokens especiales al target: `<start>` y `<end>`.
- [ ] Definir `max_len_article` y `max_len_summary`.
- [ ] Tokenizar artículos y resúmenes.
- [ ] Crear secuencias con padding.

## 4. Embeddings preentrenados

- [ ] Descargar o cargar embeddings preentrenados.
- [ ] Construir la matriz de embeddings para el vocabulario de entrada.
- [ ] Construir la matriz de embeddings para el vocabulario de salida, si aplica.
- [ ] Validar que las dimensiones coincidan con las capas `Embedding`.

## 5. Modelo Encoder-Decoder + Atención

- [ ] Adaptar el Encoder LSTM del notebook de traducción.
- [ ] Adaptar el Decoder LSTM para generar resúmenes.
- [ ] Conectar el mecanismo de atención.
- [ ] Configurar la salida con `Dense(vocab_size_summary, activation="softmax")`.
- [ ] Compilar el modelo con una perdida apropiada para clasificación por token.

## 6. Entrenamiento

- [ ] Preparar entradas del decoder desplazadas:
  - decoder input: resumen con `<start>`
  - decoder target: resumen desplazado hasta `<end>`
- [ ] Entrenar primero con una muestra reducida.
- [ ] Guardar historial de entrenamiento.
- [ ] Guardar modelo y tokenizadores.
- [ ] Registrar perdida de entrenamiento y validación.

## 7. Inferencia y evaluación

- [ ] Implementar función para generar resúmenes.
- [ ] Probar con ejemplos del conjunto de prueba.
- [ ] Mostrar:
  - fragmento del artículo
  - resumen real
  - resumen generado
- [ ] Evaluar cualitativamente si el resumen conserva la idea principal.
- [ ] Opcional: calcular ROUGE con la librería `evaluate`.

## 8. Entrega

- [ ] Notebook ejecutable y ordenado.
- [ ] Respuestas escritas a los tres puntos del taller.
- [ ] Capturas o salidas visibles del entrenamiento.
- [ ] Ejemplos finales de funcionamiento.
- [ ] Conclusión corta sobre desempeño, limitaciones y posibles mejoras.

