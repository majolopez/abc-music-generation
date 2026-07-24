# 🎵 Generación Simbólica de Música en Notación ABC: Recurrencia (LSTM) vs. Autoatención Causal (Transformer)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Music21](https://img.shields.io/badge/Music21-9.0+-008080)](https://web.mit.edu/music21/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-NVIDIA%20H200-f9ab00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

Un estudio comparativo sobre la capacidad de los modelos de Deep Learning para aprender, retener y reproducir gramática formal rígida y coherencia estética de largo alcance en secuencias musicales simbólicas.

---

## 📖 Resumen / Abstract
La generación de música simbólica a nivel de carácter impone un desafío complejo: el modelo debe aprender simultáneamente las estructuras sintácticas de la notación musical (**formato ABC**) y las estructuras melódicas/armónicas a largo plazo. 

Este repositorio contrasta dos paradigmas fundamentales del procesamiento de secuencias:
1. **Modelo Recurrente (LSTM Baseline):** Basado en memoria oculta paso a paso.
2. **Modelo de Atención (Transformer Causal - GPT style):** Basado en máscaras causales triangulares y procesamiento paralelo de ventanas de contexto.

---

## 📊 Comparativa de Rendimiento Empírico

El experimento se evaluó sobre un conjunto de pruebas de **100 melodías independientes** generadas mediante muestreo multinomial con temperatura controlada ($T=0.8$). La gramaticalidad se verificó de forma automatizada y silenciosa utilizando el motor de análisis de `music21`.

| Modelo | # Parámetros | Train Loss | Val Loss | Perplejidad (PPL) | Tiempo/Época (H200) | Tasa de Validez Sintáctica |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM Baseline** | *[Ver Notebook]* | *[Ver Notebook]* | *[Ver Notebook]* | *[Ver Notebook]* | **~226.5 s** | **[Ver Notebook]%** |
| **Transformer Causal** | *[Ver Notebook]* | *[Ver Notebook]* | *[Ver Notebook]* | *[Ver Notebook]* | **~55.4 s** | **[Ver Notebook]%** |

> *Nota: El Transformer demuestra una velocidad de entrenamiento significativamente superior (~4x más rápido por época) gracias a la paralelización matricial en los Tensor Cores de la GPU NVIDIA H200, superando el cuello de botella secuencial de la memoria LSTM.*

---

## 🛠️ Decisiones Metodológicas y de Ingeniería

* **Tokenización a Nivel de Carácter:** El modelo aprende la sintaxis directamente de los caracteres (corchetes de repetición `|:`, duraciones `/2`, alteraciones `^f`) sobre las 4,443 canciones del dataset.
* **Inyección de Cabeceras Sintéticas (`Synthetic Header`):** El dataset original carece de metadatos ABC (`X:`, `T:`, `M:`, `K:`). Para poder evaluar la legalidad matemática del compás y la duración de las notas generadas en el Paso de Validación, se diseñó un middleware que inyecta un encabezado sintético normalizado antes del parseo por `music21`.
* **Workaround de Compatibilidad H200:** En servidores con conflictos de versión entre librerías cuDNN (v91000 vs v92000), se implementó el bypass `torch.backends.cudnn.enabled = False` para garantizar una estabilidad absoluta en hardware de última generación.

---

## 📂 Estructura del Repositorio

```text
├── data/
│   └── piano-musics-abc-notation.txt   # Dataset original de tunes en formato ABC
├── models/
│   ├── lstm.py                         # Arquitectura LSTMModel
│   └── transformer.py                  # Arquitectura TransformerModel (Causal/Decoder-only)
├── utils/
│   ├── dataset.py                      # Tokenizador a nivel de carácter y ABCDataset
│   └── evaluation.py                   # Motor de validación music21 y conteo de parámetros
├── notebooks/
│   └── Proyecto_Deep_Learning.ipynb    # Notebook completo con outputs, gráficas y resultados
├── requirements.txt                    # Dependencias del proyecto
└── README.md                           # Documentación principal