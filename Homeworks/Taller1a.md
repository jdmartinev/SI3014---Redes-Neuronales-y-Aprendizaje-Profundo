# Taller 1 — Fundamentos de Deep Learning

## Objetivo

Aplicar los conceptos fundamentales relacionados con regresión, clasificación, funciones de costo, gradiente, descenso del gradiente, funciones de activación, *backpropagation* y entrenamiento de redes neuronales.

> **Instrucciones:** Justifique sus respuestas cuando se solicite. El objetivo del taller no es únicamente realizar cálculos, sino interpretar el comportamiento de los modelos y de su proceso de entrenamiento.

---

## 1. Regresión lineal y función de costo

Se está entrenando un modelo de regresión lineal:

$$
\hat{y} = wx + b
$$

La siguiente figura muestra los datos y la recta obtenida por el modelo.

![Regresión lineal](figures/taller1/regresion_lineal.png)

En la figura se ha señalado un punto $P$.

### Preguntas

1. ¿La predicción del modelo $\hat{y}$ para el punto $P$ es mayor o menor que el valor real $y$?

2. Si definimos el residuo como:

$$
e = y - \hat{y}
$$

¿el residuo correspondiente al punto $P$ es positivo o negativo?

3. Si utilizamos el error cuadrático:

$$
L = (y - \hat{y})^2
$$

¿puede la función de costo tomar un valor negativo? Justifique.

---

## 2. Regresión logística y frontera de decisión

La siguiente figura representa un problema de clasificación binaria. La línea corresponde a la frontera de decisión obtenida mediante regresión logística.

![Clasificación binaria](figures/taller1/clasificacion_binaria.png)

El modelo calcula inicialmente:

$$
z = w_1x_1 + w_2x_2 + b
$$

y posteriormente:

$$
\hat{y} = \sigma(z)
$$

El modelo clasifica una observación como clase 1 cuando:

$$
\hat{y} \geq 0.5
$$

### Preguntas

1. ¿Cuál de los puntos señalados en la figura está incorrectamente clasificado?

2. ¿Qué representa geométricamente la línea mostrada en la figura?

3. Sobre la frontera de decisión, ¿qué valor tiene aproximadamente $\hat{y}$?

4. ¿Qué valor tiene $z$ sobre la frontera de decisión?

5. Suponga que un punto tiene $z = 3$. Sin calcular exactamente la función sigmoide, ¿esperaría que fuera clasificado como clase 0 o clase 1? Justifique.

---

## 3. Gradiente y función de costo

La siguiente figura representa la función de costo $J(w)$ de un modelo para diferentes valores del parámetro $w$.

![Función de costo](figures/taller1/funcion_costo.png)

En la gráfica se muestran tres posiciones diferentes: **A**, **B** y **C**.

### Punto A

1. ¿El gradiente

$$
\frac{\partial J}{\partial w}
$$

es positivo, negativo o aproximadamente cero?

2. Si queremos minimizar $J$, ¿deberíamos aumentar o disminuir $w$?

### Punto B

3. ¿El gradiente es positivo, negativo o aproximadamente cero?

4. ¿Deberíamos aumentar o disminuir $w$ para acercarnos al mínimo?

### Punto C

5. ¿Qué valor aproximado debería tener el gradiente?

6. ¿Por qué este punto es importante durante el entrenamiento?

---

## 4. Actualización de parámetros

Un modelo tiene actualmente el parámetro:

$$
w = 3
$$

Durante *backpropagation* se obtiene:

$$
\frac{\partial J}{\partial w} = -4
$$

y se utiliza un *learning rate*:

$$
\eta = 0.1
$$

La regla de actualización es:

$$
w_{\text{nuevo}}
=
w - \eta \frac{\partial J}{\partial w}
$$

### Preguntas

1. Calcule el nuevo valor de $w$.

2. ¿El parámetro aumentó o disminuyó?

3. Explique por qué la dirección del cambio tiene sentido teniendo en cuenta el signo del gradiente.

4. Suponga ahora que utilizamos:

$$
\eta = 0.001
$$

¿Qué efecto tendría sobre el tamaño de la actualización?

5. ¿Un *learning rate* más pequeño garantiza un mejor entrenamiento? Justifique.

---

## 5. Efecto del Learning Rate

Tres modelos idénticos fueron entrenados utilizando diferentes valores de *learning rate*.

![Learning rate](figures/taller1/learning_rate.png)

Las tres curvas muestran la evolución de la función de costo durante el entrenamiento.

### Preguntas

1. ¿Cuál modelo probablemente utiliza un *learning rate* demasiado pequeño?

2. ¿Cuál parece utilizar un *learning rate* adecuado?

3. ¿Cuál probablemente utiliza un *learning rate* demasiado grande?

4. Explique por qué un *learning rate* demasiado grande puede impedir que el modelo alcance un mínimo de la función de costo.

5. Si el *loss* disminuye de forma estable, pero después de muchas épocas todavía se encuentra lejos de su valor mínimo, ¿qué modificación del *learning rate* consideraría probar?

---

## 6. ¿Por qué necesitamos funciones de activación?

Considere las siguientes redes neuronales.

### Red A

`Entrada → Linear₁ → Linear₂ → Linear₃ → Salida`

### Red B

`Entrada → Linear₁ → ReLU → Linear₂ → ReLU → Linear₃ → Salida`

Considere inicialmente dos transformaciones lineales consecutivas:

$$
z_1 = W_1x + b_1
$$

$$
z_2 = W_2z_1 + b_2
$$

### Preguntas

1. Sustituya $z_1$ en la expresión de $z_2$.

2. Reorganice la expresión resultante de manera que tenga la forma:

$$
z_2 = Wx + b
$$

3. A partir del resultado anterior, explique por qué varias capas lineales consecutivas sin funciones de activación pueden representarse mediante una única transformación lineal.

4. ¿Qué aportan las funciones ReLU de la **Red B**?

5. ¿Cuál de las dos redes puede aprender relaciones no lineales más complejas? Justifique.

---

## 7. Backpropagation y regla de la cadena

Considere una red extremadamente sencilla:

`x → z → ŷ → L`

donde:

$$
z = wx
$$

$$
\hat{y} = f(z)
$$

$$
L = L(\hat{y}, y)
$$

Queremos determinar cómo afecta el parámetro $w$ al valor final de la función de costo.

### Preguntas

1. Complete la siguiente expresión utilizando la regla de la cadena:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial \hat{y}}
\cdot
\text{[ completar ]}
\cdot
\text{[ completar ]}
$$

2. ¿En qué dirección viaja la información durante el **forward pass**?

3. ¿En qué dirección se calculan los gradientes durante **backpropagation**?

4. ¿Qué información nos proporciona finalmente el valor:

$$
\frac{\partial L}{\partial w}
$$

5. ¿Por qué necesitamos calcular este gradiente antes de actualizar $w$?

---

## 8. Curvas de entrenamiento

Se entrenó una red neuronal durante 100 épocas y se obtuvo el siguiente comportamiento:

![Training y Validation Loss](figures/taller1/training_validation.png)

La figura muestra el *Training Loss* y el *Validation Loss*.

### Preguntas

1. ¿Está aprendiendo el modelo durante las primeras épocas? ¿Qué evidencia observa?

2. ¿Qué ocurre aproximadamente después de la época indicada en la figura?

3. ¿Continuar disminuyendo el *Training Loss* significa necesariamente que el modelo está mejorando?

4. ¿En qué región de la gráfica consideraría razonable seleccionar el modelo final?

5. ¿Cómo se denomina el fenómeno en el cual el modelo continúa mejorando sobre los datos de entrenamiento pero empeora sobre los datos de validación?

6. ¿Qué diferencia conceptual existe entre **aprender los datos de entrenamiento** y **generalizar**?

---

## 9. Full Batch, SGD y Mini-Batch

Tenemos un dataset con:

$$
N = 10\,000
$$

observaciones.

Se consideran tres estrategias de entrenamiento.

### Estrategia A

Se utilizan las **10 000 observaciones** para calcular el gradiente antes de realizar cada actualización de los parámetros.

### Estrategia B

Se utiliza **una sola observación** para calcular el gradiente antes de cada actualización.

### Estrategia C

Se utilizan grupos de **32 observaciones** para calcular el gradiente antes de cada actualización.

### Preguntas

1. Identifique cuál estrategia corresponde a:

   - Full Batch Gradient Descent
   - Stochastic Gradient Descent (SGD)
   - Mini-Batch Gradient Descent

2. ¿Cuántas actualizaciones de parámetros realiza aproximadamente la estrategia A durante una época?

3. ¿Cuántas realiza aproximadamente la estrategia B?

4. Para la estrategia C, estime el número de actualizaciones realizadas durante una época.

5. ¿En cuál estrategia espera que la estimación del gradiente presente mayor variabilidad entre actualizaciones?

6. ¿Cuál estrategia es la más utilizada normalmente para entrenar redes neuronales?

7. Explique una ventaja de utilizar *mini-batches*.

---

## 10. Diagnóstico de un entrenamiento

Se está entrenando una red neuronal para un problema de clasificación multiclase.

Durante el entrenamiento se obtuvieron los siguientes resultados:

| Epoch | Training Loss | Training Accuracy | Validation Accuracy |
|------:|--------------:|------------------:|--------------------:|
| 1     | 1.82 | 34% | 33% |
| 5     | 1.21 | 55% | 52% |
| 10    | 0.79 | 72% | 68% |
| 20    | 0.43 | 88% | 79% |
| 40    | 0.18 | 97% | 72% |

### Preguntas

1. ¿Hay evidencia de que la red está aprendiendo? Justifique utilizando los datos de la tabla.

2. ¿En qué intervalo comienza a aparecer un problema de generalización?

3. Entre las épocas 20 y 40:

   - ¿Qué ocurre con el *Training Loss*?
   - ¿Qué ocurre con el *Training Accuracy*?
   - ¿Qué ocurre con el *Validation Accuracy*?

4. ¿Por qué una reducción continua del *Training Loss* no implica necesariamente que el modelo esté mejorando?

5. Si únicamente observáramos el *Training Accuracy*, ¿sería fácil detectar el problema? Explique.

6. Un estudiante propone:

   > "Debemos entrenar durante más épocas porque el Training Accuracy todavía puede llegar al 100%."

   ¿Está de acuerdo? Justifique.

7. Si tuviera que elegir entre el modelo de la época 20 y el modelo de la época 40 para utilizarlo con datos nuevos, ¿cuál elegiría y por qué?

---

## Para pensar

Considere el proceso general de entrenamiento de una red neuronal:

`Datos → Forward Pass → Predicción → Loss → Backpropagation → Gradientes → Actualización de parámetros`

Explique con sus propias palabras cómo este ciclo permite que una red neuronal aprenda a partir de los datos.
