# Taller 1A – Fundamentos de Deep Learning

## Objetivo

Aplicar los conceptos fundamentales relacionados con regresión, clasificación, funciones de costo, gradiente, descenso del gradiente, funciones de activación, backpropagation y entrenamiento de redes neuronales.

Justifique sus respuestas cuando se solicite.

---

## 1. Regresión lineal y función de costo

Se está entrenando un modelo de regresión lineal:

\[
\hat{y}=wx+b
\]

La siguiente figura muestra los datos y la recta obtenida por el modelo.

**[FIGURA 1: nube de puntos + recta de regresión. Incluir un punto (P) claramente por encima de la recta.]**

Para el punto (P):

**a.** ¿La predicción del modelo (\hat y) es mayor o menor que el valor real (y)?

**b.** Si definimos el residuo como:

[
e=y-\hat y
]

¿el residuo correspondiente a (P) es positivo o negativo?

**c.** Si usamos el error cuadrático:

[
L=(y-\hat y)^2
]

¿podemos obtener un valor negativo de la función de costo? Justifique.

---

## 2. Regresión logística y frontera de decisión

La siguiente figura representa un problema de clasificación binaria. La línea corresponde a la frontera de decisión obtenida mediante regresión logística.

**[FIGURA 2: nube de puntos de dos clases + frontera lineal. Marcar P1 correctamente clasificado y P2 incorrectamente clasificado.]**

El modelo utiliza:

[
z=w_1x_1+w_2x_2+b
]

[
\hat y=\sigma(z)
]

y clasifica como clase 1 cuando:

[
\hat y\geq0.5
]

Responda:

**a.** ¿Cuál de los puntos señalados está incorrectamente clasificado?

**b.** ¿Qué representa geométricamente la línea mostrada?

**c.** Sobre la frontera de decisión, ¿qué valor tiene aproximadamente (\hat y)?

**d.** ¿Qué valor tiene (z) sobre la frontera?

---

## 3. Gradiente y función de costo

La siguiente figura representa la función de costo (J(w)) de un modelo.

**[FIGURA 3: parábola convexa con mínimo aproximadamente en w=4. Marcar A a la izquierda del mínimo, B a la derecha y C exactamente en el mínimo.]**

### Punto A

**a.** ¿El gradiente

[
\frac{\partial J}{\partial w}
]

es positivo, negativo o aproximadamente cero?

**b.** Si queremos minimizar (J), ¿deberíamos aumentar o disminuir (w)?

### Punto B

**c.** ¿El gradiente es positivo, negativo o aproximadamente cero?

**d.** ¿Deberíamos aumentar o disminuir (w)?

### Punto C

**e.** ¿Qué valor aproximado debería tener el gradiente?

**f.** ¿Por qué este punto es importante durante el entrenamiento?

---

## 4. Actualización de parámetros

Un modelo tiene actualmente el parámetro:

[
w=3
]

Durante backpropagation se obtiene:

[
\frac{\partial J}{\partial w}=-4
]

y se utiliza un learning rate:

[
\eta=0.1
]

La regla de actualización es:

[
w_{\text{nuevo}}
================

w-\eta\frac{\partial J}{\partial w}
]

**a.** Calcule el nuevo valor de (w).

**b.** ¿El parámetro aumentó o disminuyó?

**c.** Explique por qué la dirección del cambio tiene sentido teniendo en cuenta el signo del gradiente.

**d.** ¿Qué ocurriría con el tamaño de la actualización si usamos (\eta=0.001)?

No es necesario volver a realizar todo el entrenamiento; razone a partir de la ecuación.

---

## 5. Learning rate

Tres modelos idénticos fueron entrenados usando diferentes valores de learning rate.

**[FIGURA 4: tres gráficas de Loss vs Epoch.]**

* **Modelo A:** loss disminuye extremadamente lento.
* **Modelo B:** loss disminuye rápidamente y converge suavemente.
* **Modelo C:** loss presenta grandes oscilaciones y no converge.

Responda:

**a.** ¿Cuál modelo probablemente tiene un learning rate demasiado pequeño?

**b.** ¿Cuál parece utilizar un learning rate adecuado?

**c.** ¿Cuál probablemente utiliza un learning rate demasiado grande?

**d.** Explique por qué un learning rate muy grande puede impedir que el modelo alcance un mínimo de la función de costo.

---

## 6. ¿Por qué necesitamos funciones de activación?

Considere las siguientes dos redes.

### Red A

[
x
\rightarrow
Linear_1
\rightarrow
Linear_2
\rightarrow
Linear_3
\rightarrow
\hat y
]

### Red B

[
x
\rightarrow
Linear_1
\rightarrow
ReLU
\rightarrow
Linear_2
\rightarrow
ReLU
\rightarrow
Linear_3
\rightarrow
\hat y
]

Dos transformaciones lineales consecutivas pueden escribirse como:

[
z_1=W_1x+b_1
]

[
z_2=W_2z_1+b_2
]

**a.** Sustituya (z_1) dentro de la segunda ecuación.

**b.** A partir del resultado anterior, explique por qué varias capas lineales consecutivas sin funciones de activación pueden representarse mediante una única transformación lineal.

**c.** ¿Qué aportan las funciones ReLU de la Red B?

**d.** ¿Cuál de las dos redes puede aprender fronteras de decisión no lineales?

---

## 7. Backpropagation

Considere el siguiente flujo simplificado:

[
x
\xrightarrow{w}
z
\xrightarrow{f}
\hat y
\xrightarrow{}
L
]

donde:

[
z=wx
]

[
\hat y=f(z)
]

[
L=L(\hat y,y)
]

Queremos determinar cómo afecta (w) a la función de costo.

**a.** Complete la expresión utilizando la regla de la cadena:

[
\frac{\partial L}{\partial w}
=============================

\frac{\partial L}{\partial \hat y}
\cdot
\underline{\hspace{2cm}}
\cdot
\underline{\hspace{2cm}}
]

**b.** ¿En qué dirección viaja la información durante el **forward pass**?

**c.** ¿En qué dirección se calculan los gradientes durante **backpropagation**?

**d.** Explique en una frase qué información proporciona finalmente:

[
\frac{\partial L}{\partial w}
]

---

## 8. Curvas de entrenamiento

Se entrenó una red neuronal durante 100 épocas y se obtuvo la siguiente gráfica.

**[FIGURA 5: Training Loss y Validation Loss. Ambas bajan inicialmente; aproximadamente después de epoch 40, training continúa bajando mientras validation empieza a subir.]**

Responda:

**a.** ¿Está aprendiendo el modelo durante las primeras épocas? ¿Qué evidencia observa?

**b.** ¿Qué ocurre aproximadamente después de la época 40?

**c.** ¿Continuar entrenando necesariamente mejora el modelo?

**d.** ¿En qué región de la gráfica consideraría razonable seleccionar el modelo final?

**e.** ¿Cómo se denomina el fenómeno observado después de la época 40?

---

## 9. Full Batch, SGD y Mini-Batch

Tenemos un dataset con:

[
N=10,000
]

observaciones.

Se consideran tres estrategias:

### Estrategia A

Se utilizan las 10 000 observaciones para calcular cada actualización de los parámetros.

### Estrategia B

Se utiliza una sola observación para cada actualización.

### Estrategia C

Se utilizan grupos de 32 observaciones para cada actualización.

**a.** Identifique cuál corresponde a:

* Full Batch Gradient Descent
* Stochastic Gradient Descent (SGD)
* Mini-Batch Gradient Descent

**b.** ¿Cuál realiza aproximadamente más actualizaciones de parámetros durante una época?

**c.** ¿En cuál espera que la estimación del gradiente tenga mayor variabilidad entre actualizaciones?

**d.** ¿Cuál de las tres estrategias es la más utilizada normalmente para entrenar redes neuronales modernas?

**e.** Explique una ventaja de utilizar mini-batches.

---

## 10. Diagnóstico de un entrenamiento

Se está entrenando una red neuronal para clasificación multiclase.

Durante las primeras épocas se obtiene:

| Epoch | Training Loss | Training Accuracy | Validation Accuracy |
| ----: | ------------: | ----------------: | ------------------: |
|     1 |          1.82 |               34% |                 33% |
|     5 |          1.21 |               55% |                 52% |
|    10 |          0.79 |               72% |                 68% |
|    20 |          0.43 |               88% |                 79% |
|    40 |          0.18 |               97% |                 72% |

Responda:

**a.** ¿Hay evidencia de que la red está aprendiendo? Justifique usando los datos.

**b.** ¿En qué intervalo comienza a aparecer un problema de generalización?

**c.** ¿Por qué una reducción continua del `Training Loss` no implica necesariamente que el modelo esté mejorando?

**d.** Si únicamente pudiera observar `Training Accuracy`, ¿sería fácil detectar el problema? Explique.

**e.** Un estudiante propone:

> "Debemos entrenar durante más épocas porque el Training Accuracy todavía puede llegar al 100%."

¿Está de acuerdo? Justifique.

---

## Pregunta adicional – Integración

Considere el proceso:

[
X
\rightarrow
\text{Linear}
\rightarrow
\text{ReLU}
\rightarrow
\text{Linear}
\rightarrow
\text{Softmax}
\rightarrow
\hat y
\rightarrow
L
]

Explique brevemente cuál es la función de cada uno de los siguientes elementos durante el entrenamiento:

1. Transformación lineal.
2. Función de activación.
3. Softmax.
4. Función de costo.
5. Gradiente.
6. Backpropagation.
7. Learning rate.

Finalmente, explique cómo estos elementos trabajan conjuntamente para que la red neuronal aprenda.
