# Práctica 2 – PLN2

## 📖 Visión general
Este repositorio contiene la implementación de la **Práctica 2** del curso *Procesamiento del Lenguaje Natural*. El notebook `notebook.ipynb` muestra una pipeline completa que parte de la **Práctica 1** y añade:

1. **Recuperación densa** – un índice k‑NN sobre embeddings de transformadores.
2. **Clasificador k‑NN** – clasificación mediante voto mayoritario entre los vecinos recuperados.
3. **Clasificador híbrido (RAG)** – una combinación sencilla de las probabilidades del transformador (soft‑max) con las probabilidades obtenidas de los vecinos (ponderación α).
4. **Utilidades de evaluación** – precision@k para la recuperación, accuracy y macro‑F1 para la clasificación.

El código está preparado para ejecutarse en una máquina Windows estándar con un entorno virtual de Python.

---

## 🛠️ Requisitos
```bash
# Crear y activar un entorno virtual (recomendado)
python -m venv .venv
.venv\\Scripts\\activate

# Instalar las librerías necesarias
pip install pandas numpy scikit-learn torch transformers tqdm jupyterlab
```
> **Nota**: El proyecto se desarrolló con **Python 3.13** y `torch` seleccionará automáticamente CPU o CUDA si está disponible.

---

## 📂 Datos
El notebook espera el archivo CSV usado en la Práctica 1:
```
../../practica1/pln2_practica1/data/en_song_lyrics_clear.csv
```
Si la ruta relativa no existe (por ejemplo, al copiar el repositorio), el notebook recurre a un pequeño conjunto de datos ficticio para que el script pueda ejecutarse con fines de demostración.

---

## ▶️ Cómo ejecutar el notebook
1. Abra una terminal en la raíz del proyecto (`pln2_practica2`).
2. Active el entorno virtual (ver requisitos).
3. Inicie Jupyter Lab/Notebook:
   ```bash
   jupyter lab notebook.ipynb   # o `jupyter notebook notebook.ipynb`
   ```
4. Ejecute las celdas de forma secuencial. Las secciones principales son:
   - **Entorno y carga de datos** – importaciones, gestión de rutas, división train/val/test.
   - **Módulo de recuperación densa** – `DenseRetriever` construye un índice a partir del conjunto de entrenamiento.
   - **Evaluación de recuperación** – calcula *Precision@k* sobre una muestra aleatoria del conjunto de prueba.
   - **Clasificador k‑NN** – predice etiquetas usando voto mayoritario de los vecinos y muestra *Accuracy* y *Macro‑F1*.
   - **Clasificador híbrido (RAG)** – funciones placeholder que combinan probabilidades del transformador (`P_T`) con las del k‑NN (`P_K`). La inferencia real del transformador queda como TODO para el informe final.
   - **Experimentos adicionales** – secciones para compresión, resumen y destilación de conocimiento están descritas en celdas markdown del notebook.

---

## 📄 Estructura del notebook (resumen rápido)
| Celda | Propósito |
|------|-----------|
| 1‑4  | Importaciones y utilidades auxiliares |
| 5‑14 | Carga del CSV, fallback a datos ficticios, división train/val/test |
| 15‑22| Inicialización de `DenseRetriever` (RoBERTa‑base) y construcción del índice |
| 23‑31| Función `evaluate_retrieval` → *Precision@k* en una muestra de prueba |
| 32‑44| Clase `KNNClassifier` → predicción por voto mayoritario |
| 45‑53| Evaluación del k‑NN en la muestra de prueba (accuracy y macro‑F1) |
| 54‑70| Esqueleto del clasificador híbrido RAG (fusión de probabilidades) |
| 71‑… | Celdas markdown que describen extensiones opcionales (compresión de modelo, resumen automático, análisis ético, etc.) |

---

## 📦 Organización del proyecto
```
pln2_practica2/
├─ README.md          # ← ¡estás leyendo este archivo!
├─ notebook.ipynb     # notebook principal de experimentos
├─ src/               # código fuente (modelos, utilidades, etc.) – reutilizado de la Práctica 1
└─ .venv/            # entorno virtual (no versionado en git)
```

---

## 🎓 Qué hace el notebook
1. **Carga el conjunto de datos** y crea divisiones estratificadas.
2. **Genera embeddings densos** para cada frase de entrenamiento usando el mismo modelo transformador afinado en la Práctica 1.
3. **Construye un índice k‑NN** (`sklearn.NearestNeighbors`) sobre esos embeddings.
4. **Evalúa la calidad de la recuperación** mediante *Precision@5* en una muestra aleatoria del conjunto de prueba.
5. **Implementa un clasificador k‑NN sencillo** que predice la etiqueta de una consulta mediante voto mayoritario entre sus vecinos.
6. **Informa métricas de clasificación** (accuracy, macro‑F1).
7. **Esboza un clasificador híbrido RAG** que combinaría la salida soft‑max del transformador con las probabilidades basadas en vecinos (interpolación α). La inferencia del transformador se deja como placeholder (`get_transformer_probs`).
8. **Proporciona una hoja de ruta** para extensiones opcionales como destilación de conocimiento, resumen automático por clase y análisis ético.

---

## 📚 Lecturas complementarias
- Repositorio de la Práctica 1 para el modelo base del transformador y utilidades de preprocesado.
- El *enunciado.pdf* del curso para la especificación completa de las tareas.

---

*¡Feliz experimentación!* 🚀

## 📖 Overview
This repository contains the implementation for **Practica 2** of the *Procesamiento del Lenguaje Natural* course.  The notebook `notebook.ipynb` walks through a complete pipeline that builds on **Practica 1** and adds:

1. **Dense Retrieval** – a k‑NN index over transformer embeddings.
2. **k‑NN Classifier** – classification by majority vote on the retrieved neighbours.
3. **Hybrid (RAG) Classifier** – a simple combination of the transformer’s soft‑max probabilities with the k‑NN probabilities (alpha‑weighted).
4. **Evaluation utilities** – precision@k for retrieval, accuracy & macro‑F1 for classification.

The code is written to be runnable on a standard Windows machine with a Python virtual environment.

---

## 🛠️ Requirements
```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate

# Install the required libraries
pip install pandas numpy scikit-learn torch transformers tqdm jupyterlab
```
> **Note**: The project was developed with **Python 3.13** and `torch` will automatically select CPU or CUDA if available.

---

## 📂 Data
The notebook expects the CSV file used in Practica 1:
```
../../practica1/pln2_practica1/data/en_song_lyrics_clear.csv
```
If the relative path does not exist (e.g., when the repository is copied), the notebook falls back to a small dummy dataset so that the script can still be executed for demonstration purposes.

---
