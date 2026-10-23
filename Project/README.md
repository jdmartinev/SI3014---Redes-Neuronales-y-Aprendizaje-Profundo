# Deep Learning — Lineamientos del proyecto final

Nos acercamos a la recta final del curso de Deep Learning y, con ello, a las sustentaciones de los proyectos finales. Por esta razón, quiero compartir con ustedes los lineamientos académicos y logísticos que deberán tener en cuenta para el desarrollo de esta última actividad.

Les solicito leer cuidadosamente la siguiente información, especialmente lo relacionado con las fechas de entrega, los contenidos mínimos de la presentación y los criterios de evaluación.

## 1. Fechas de sustentación y entrega

| Actividad | Fecha o período | Medio de entrega |
| --- | --- | --- |
| Entrega preliminar: título y objetivos | Jueves 29 de octubre de 2026 | Buzón interactiva |
| Entrega de la presentación y del informe final | Miércoles 11 de noviembre de 2026 | Buzón interactiva |
| Sustentaciones de los proyectos finales | Semanas 17 y 18 | Según la programación de sustentaciones |

La fecha límite para entregar **tanto la presentación como el informe final** será el **miércoles 11 de noviembre de 2026**, independientemente de la semana en la que tengan programada su sustentación. Ambos documentos deberán ser enviados a mi correo electrónico a más tardar en esa fecha.

### Condiciones de entrega

- La sustentación será evaluada exclusivamente con base en los documentos entregados dentro del plazo establecido.
- No se aceptarán versiones actualizadas, correcciones ni modificaciones posteriores a la fecha límite.
- La ausencia de entrega de los documentos dentro del plazo establecido se interpretará como la no presentación del proyecto final y, en consecuencia, la calificación correspondiente será de **0.0**.

Les recomiendo organizar sus actividades con suficiente anticipación y evitar dejar la preparación de los documentos para los últimos días.

## 2. Distribución de la calificación

El proyecto final representa el **20 % de la calificación definitiva del curso**, distribuido de la siguiente manera:

| Componente | Porcentaje de la calificación definitiva del curso |
| --- | ---: |
| Sustentación oral | 15 % |
| Informe final | 5 % |
| **Total** | **20 %** |

Quiero recordarles que el propósito de este proyecto no es únicamente desarrollar un modelo que produzca buenos resultados. El verdadero objetivo es demostrar que comprenden el problema, que son capaces de diseñar e implementar una solución fundamentada y, sobre todo, que pueden interpretar críticamente los resultados obtenidos.

En Deep Learning, obtener una buena métrica de desempeño no es suficiente si no se comprende cómo se alcanzó, qué factores influyeron en ella y cuáles son las limitaciones del modelo.

## 3. Entrega preliminar: título y objetivos del proyecto

Con el propósito de conocer y revisar oportunamente el alcance de los proyectos finales, cada grupo deberá enviar, a más tardar el **viernes 30 de octubre de 2026**, un correo electrónico que contenga la siguiente información:

1. **Título del proyecto:** nombre que identifique claramente el problema o la aplicación que se pretende desarrollar.
2. **Objetivo general:** declaración precisa del propósito principal del proyecto, indicando qué se busca alcanzar mediante la implementación de técnicas de Deep Learning.
3. **Objetivos específicos:** entre dos (2) y tres (3) objetivos que describan las etapas o actividades fundamentales necesarias para alcanzar el objetivo general.

Los objetivos deberán formularse de manera clara, coherente y verificable, procurando que exista una relación directa entre el problema planteado, la metodología propuesta y los resultados que se esperan obtener.

## 4. Lineamientos para la sustentación oral

Cada grupo dispondrá de un máximo de **15 minutos de presentación**, seguidos de **5 minutos destinados a preguntas**.

### Condiciones de participación

- Todos los integrantes del grupo deberán participar activamente en la exposición.
- Cualquier integrante deberá estar en capacidad de responder preguntas relacionadas con cualquier componente del proyecto, independientemente de la sección que haya presentado.
- Se evaluará no solo la calidad de los resultados obtenidos, sino también la comprensión conceptual, el rigor metodológico y la capacidad para justificar las decisiones tomadas durante el desarrollo del proyecto.

La presentación deberá seguir una estructura lógica que permita comprender el problema abordado, la solución propuesta, su implementación y los resultados alcanzados.

### Contenido mínimo de la presentación

#### 4.1. Planteamiento del problema y objetivos

La presentación deberá comenzar con una descripción clara del problema que se pretende resolver, su contexto y la justificación de su relevancia. Posteriormente, deberán formular explícitamente el objetivo general y los objetivos específicos del proyecto.

#### 4.2. Arquitectura del modelo

Deberán explicar el modelo de Deep Learning seleccionado, su arquitectura y los fundamentos que justifican su elección.

Si utilizaron una arquitectura previamente reportada en la literatura, deberán presentar la referencia correspondiente e identificar claramente las modificaciones realizadas respecto al modelo original.

No será suficiente indicar qué arquitectura utilizaron; deberán demostrar que comprenden su funcionamiento y justificar por qué resulta apropiada para el problema planteado.

#### 4.3. Base de datos y preprocesamiento

Deberán describir la base de datos utilizada, su procedencia, sus principales características y las estrategias implementadas para su preparación.

Esto incluye, cuando corresponda:

- La división de los conjuntos de entrenamiento, validación y prueba.
- Las técnicas de normalización.
- El aumento de datos (*data augmentation*).
- Cualquier otro procedimiento de preprocesamiento.

Cada decisión metodológica deberá estar debidamente justificada.

#### 4.4. Entrenamiento y configuración experimental

Deberán presentar las condiciones bajo las cuales se entrenó el modelo, incluyendo:

- Curvas de aprendizaje correspondientes al entrenamiento y la validación (*loss* y métricas de desempeño).
- Hiperparámetros utilizados y justificación de su selección.
- Función de pérdida, optimizador y estrategia de entrenamiento.
- Recursos computacionales utilizados: CPU, GPU, memoria y entorno de ejecución, según corresponda.
- Análisis del comportamiento del modelo durante el entrenamiento, identificando posibles problemas de sobreajuste, subajuste o dificultades de convergencia.

**Presentar una gráfica no equivale a analizarla.** Se espera una interpretación crítica de los resultados y una explicación de las decisiones tomadas a partir de ellos.

#### 4.5. Resultados y evaluación del modelo

Deberán presentar los resultados obtenidos mediante métricas cuantitativas y visualizaciones pertinentes.

Además, será necesario demostrar el funcionamiento del modelo con muestras externas o datos no utilizados durante el entrenamiento, analizando su capacidad de generalización.

Se valorará especialmente la capacidad para identificar las fortalezas, limitaciones y posibles fallas del modelo, así como para establecer si los resultados obtenidos permiten responder satisfactoriamente al problema inicialmente planteado.

#### 4.6. Conclusiones

Las conclusiones deberán estar directamente relacionadas con los objetivos formulados y respaldadas por los resultados experimentales.

Eviten conclusiones genéricas o afirmaciones que no puedan sustentarse con la evidencia presentada. Se espera una reflexión crítica sobre los alcances del trabajo, sus limitaciones y las posibles oportunidades de mejora.

## 5. Lineamientos para el informe final

El informe final deberá presentarse en **formato PDF**, con una **extensión máxima de tres (3) páginas**. Este documento deberá constituir una síntesis clara, precisa y técnicamente fundamentada del trabajo desarrollado.

### Contenido mínimo del informe

1. **Objetivos del proyecto:** descripción del problema abordado, objetivo general y objetivos específicos que orientaron el desarrollo del trabajo.
2. **Base de datos:** descripción de los datos utilizados, su procedencia, características principales, estrategias de preprocesamiento y distribución en los conjuntos de entrenamiento, validación y prueba.
3. **Modelo implementado:** presentación de la arquitectura seleccionada, justificación de su elección y descripción de las principales decisiones metodológicas relacionadas con su implementación y entrenamiento.
4. **Resultados:** presentación y análisis de los resultados más relevantes, incluyendo las métricas de desempeño, las visualizaciones pertinentes y la evaluación del modelo sobre datos no utilizados durante el entrenamiento.
5. **Conclusiones:** síntesis de los principales hallazgos, cumplimiento de los objetivos planteados y limitaciones identificadas durante el desarrollo del proyecto.

Dada la extensión limitada del documento, se espera que los estudiantes demuestren capacidad de síntesis, seleccionando cuidadosamente la información, las tablas y las figuras que mejor permitan comunicar los resultados de su investigación.

## 6. Evaluación de la sustentación

La calificación de la sustentación considerará tanto la evaluación del profesor como la evaluación realizada por los estudiantes.

La nota final se calculará mediante un algoritmo que promedia ambas calificaciones y aplica una penalización cuando existe una diferencia significativa entre ellas.

Este mecanismo busca promover una evaluación responsable, crítica y coherente entre la valoración del profesor y la realizada por los estudiantes. Por esta razón, es fundamental que las calificaciones asignadas por los estudiantes respondan a criterios académicos y no a consideraciones personales.

---

Les deseo muchos éxitos en la preparación de sus proyectos y quedo atento a las inquietudes que puedan surgir.

Saludos.
