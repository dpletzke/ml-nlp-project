# TODO - Taller Week 6 CNN/DailyMail

Referencia base:

`../applied-ml-course/Week6/Code/5-Encoder_Decoder_+_Atención_+_Pre_trained_Embeddings.ipynb`

## 1. Consulta sobre `cnn_dailymail`

- [x] Investigar la base de datos `cnn_dailymail` en Hugging Face.
- [x] Identificar sus columnas principales: `article`, `highlights`, `id`.
- [x] Explicar que el problema a resolver es resumen automático de texto.
- [x] Escribir una respuesta corta en el notebook sobre aplicaciones posibles del modelo.

## 2. Carga del dataset

- [x] Instalar `datasets` si no esta disponible.
- [x] Cargar el dataset con:

```python
from datasets import load_dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
```

- [x] Revisar los splits disponibles: `train`, `validation`, `test`.
- [x] Mostrar al menos un ejemplo de `article` y `highlights`.
- [x] Crear una muestra pequena para entrenamiento inicial si el equipo no soporta el dataset completo.

## 3. Adaptación del preprocesamiento

- [x] Reutilizar la estructura del notebook Week 6.
- [x] Cambiar el par entrada-salida:
  - entrada: `article`
  - salida: `highlights`
- [x] Limpiar textos: minusculas, espacios repetidos, caracteres innecesarios.
- [x] Agregar tokens especiales al target: `<start>` y `<end>`.
- [x] Definir `max_len_article` y `max_len_summary`.
- [x] Tokenizar artículos y resúmenes.
- [x] Crear secuencias con padding.

## 4. Embeddings preentrenados

- [x] Descargar o cargar embeddings preentrenados.
- [x] Construir la matriz de embeddings para el vocabulario de entrada.
- [x] Construir la matriz de embeddings para el vocabulario de salida, si aplica.
- [x] Validar que las dimensiones coincidan con las capas `Embedding`.

## 5. Modelo Encoder-Decoder + Atención

- [x] Adaptar el Encoder LSTM del notebook de traducción.
- [x] Adaptar el Decoder LSTM para generar resúmenes.
- [x] Conectar el mecanismo de atención.
- [x] Configurar la salida con `Dense(vocab_size_summary, activation="softmax")`.
- [x] Compilar el modelo con una perdida apropiada para clasificación por token.

## 6. Entrenamiento

- [x] Preparar entradas del decoder desplazadas:
  - decoder input: resumen con `<start>`
  - decoder target: resumen desplazado hasta `<end>`
- [ ] Entrenar primero con una muestra reducida. *(iniciado — se detuvo en epoch 1/10)*
- [ ] Guardar historial de entrenamiento.
- [ ] Guardar modelo y tokenizadores.
- [ ] Registrar perdida de entrenamiento y validación.

## 7. Inferencia y evaluación

- [x] Implementar función para generar resúmenes. *(código escrito, celda no ejecutada)*
- [ ] Probar con ejemplos del conjunto de prueba.
- [ ] Mostrar:
  - fragmento del artículo
  - resumen real
  - resumen generado
- [ ] Evaluar cualitativamente si el resumen conserva la idea principal.
- [ ] Opcional: calcular ROUGE con la librería `evaluate`. *(código escrito, celda no ejecutada)*

## 8. Entrega

- [x] Notebook ejecutable y ordenado.
- [x] Respuestas escritas a los tres puntos del taller.
- [ ] Capturas o salidas visibles del entrenamiento. *(solo se ve el inicio del epoch 1)*
- [ ] Ejemplos finales de funcionamiento.
- [ ] Conclusión corta sobre desempeño, limitaciones y posibles mejoras.

