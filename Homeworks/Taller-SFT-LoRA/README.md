# SI3014 — Taller: de modelo de lenguaje a asistente mediante ajuste supervisado

En este taller se ajusta el modelo base **Qwen2.5-0.5B** con pares instrucción-respuesta en español (Dolly traducido) usando **SFT** con `SFTTrainer` de TRL y adaptadores **LoRA** de `peft`. Se compara el modelo antes y después del ajuste: pérdida de validación, respuestas generadas y capacidad de detenerse.

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jdmartinev/SI3014---Redes-Neuronales-y-Aprendizaje-Profundo/blob/main/Homeworks/Taller-SFT-LoRA/Taller_SFT_LoRA_pregrado.ipynb)

**Notebook:** [Taller_SFT_LoRA_pregrado.ipynb](Taller_SFT_LoRA_pregrado.ipynb)

## Requisitos

- GPU (T4 en Colab o Kaggle es suficiente) y conexión a internet.
- Ejecuta el **Paso 0** una vez: instala `trl` y `peft` en versiones fijas. Si instala algo, reinicia la sesión y continúa desde el Paso 1.
- Sin GPU el notebook corre con muy pocos ejemplos; sirve para recorrer el flujo, pero el ajuste es solo demostrativo.

## Entrega

El notebook con los 7 TODO completados, las salidas visibles y las respuestas breves en las celdas de texto.
