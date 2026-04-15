# Laboratorio de NLP: Modelado de Secuencias con RNNs y LSTMs

## Descripción general

En este laboratorio se introduce el **modelado de secuencias aplicado al procesamiento de lenguaje natural (NLP)**. Se propone un flujo de trabajo progresivo que inicia con el procesamiento de datos y culmina con la implementación de modelos neuronales capaces de capturar dependencias temporales en texto.

El objetivo es que el estudiante comprenda cómo transformar texto en representaciones numéricas y cómo distintos enfoques modelan la información secuencial.

---

## Estructura del laboratorio

El laboratorio está compuesto por los siguientes notebooks:

### `01_datos_vocabulario.ipynb` — Datos y Vocabulario

En este notebook se trabaja:

- Carga y exploración del dataset  
- Limpieza y preprocesamiento del texto  
- Construcción del vocabulario  
- Conversión de texto a secuencias numéricas  

Este paso es fundamental, ya que define la representación de entrada para los modelos.

---

### `02_baseline.ipynb` — Modelo Base

En este notebook se implementa un modelo base para la tarea, típicamente utilizando enfoques simples como:

- Bag of Words (BoW) o representaciones agregadas  
- Modelos lineales o redes feedforward  
- Pipeline básico de entrenamiento y evaluación  

El objetivo es establecer un **punto de referencia (baseline)** contra el cual comparar modelos más avanzados.

---

### `03_modelos_secuenciales.ipynb` — **Ejercicio del laboratorio**

**Este es el notebook que debe ser desarrollado por el estudiante.**

En este ejercicio se debe implementar modelos de aprendizaje profundo para secuencias, específicamente:

- Redes Neuronales Recurrentes (**RNNs**)  
- Redes de memoria a largo plazo (**LSTMs**)  

---

## Objetivo del ejercicio

El objetivo es diseñar e implementar modelos que:

- Capturen dependencias temporales en los datos de texto  
- Mejoren el desempeño respecto al baseline  
- Permitan analizar el impacto de la modelación secuencial  

---

## Requisitos del ejercicio

El notebook `03_modelos_secuenciales.ipynb` debe incluir como mínimo:

### Modelos
- Implementación de una **RNN**
- Implementación de una **LSTM**

### Entrenamiento
- Definición clara del loop de entrenamiento (o uso de framework)
- Manejo de batches y padding (si aplica)

### Evaluación
- Métricas de desempeño (accuracy, loss, u otras relevantes)
- Comparación explícita con el baseline

### Análisis
- Discusión breve de resultados:
  - ¿Mejora frente al baseline?
  - ¿Qué modelo funciona mejor (RNN vs LSTM)?
  - Posibles razones

---

## Preguntas guía

Para orientar el desarrollo del ejercicio:

- ¿Qué limitaciones tiene el modelo baseline frente a datos secuenciales?
- ¿Cómo cambia la representación interna en una RNN/LSTM?
- ¿Qué tipo de dependencias logra capturar cada modelo?
- ¿Se observa overfitting? ¿Cómo mitigarlo?

---

## Recomendaciones

- Utilizar embeddings (propios o preentrenados si se desea)
- Monitorear el entrenamiento (curvas de pérdida)
- Mantener el código modular y claro
- Documentar decisiones relevantes

---

## Entregable

El estudiante debe entregar:

- Notebook `03_modelos_secuenciales.ipynb` completamente funcional
- Código ejecutable y reproducible
- Resultados y análisis incluidos en el notebook
