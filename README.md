# Práctica 2 – Procesamiento del Lenguaje Natural 2
## Instrucciones de ejecución ▶️
- Clonar el repositorio de GitHub:
    ```bash
    git clone https://github.com/Di3g0E/pln2_practica2.git
    ```
- Navegar al directorio del proyecto:
    ```bash
    cd pln2_practica2
    ```
- Instalar las dependencias necesarias:
    - Usando `pip` 📦:
        ```bash
        pip install -r requirements.txt
        ```
    - Usando `conda` 🐍:
        ```bash
        conda env create -f environment.yml
        conda activate pln2_practica2
        ```
    - Usando `uv` 🚀:
        ```bash
        uv sync
        source .venv/bin/activate
        ```
- Abrir `notebook.ipynb` en Jupyter Notebook o JupyterLab:
    ```bash
    jupyter notebook notebook.ipynb
    ```
    o
    ```bash
    jupyter lab notebook.ipynb
    ```

## Descripción del proyecto 📚
### Datos utilizados 📊
- Este proyecto utiliza el conjunto de datos generado en la práctica anterior, que consiste en una **armonización** entre los siguientes conjuntos de datos de Kaggle:
    - [Genius Song Lyrics](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information)
    - [Multi-Lingual Lyrics for Genre Classification](https://www.kaggle.com/datasets/mateibejan/multilingual-lyrics-for-genre-classification)

- Esta armonización de conjuntos de datos da lugar a los siguientes *splits*:
    - **Train**: 222,308 ejemplos.
    - **Eval**: 47,638 ejemplos.
    - **Test**: 47,638 ejemplos.

### Modelos implementados 🛠
- En este proyecto se han implementado los siguientes modelos de clasificación:
    - **k-NN Classifier**: Clasificador basado en *retrieval* utilizando el modelo Transformer entrenado en la práctica anterior (**teacher**) para generar embeddings y la métrica de similitud coseno. Las predicciones se realizan mediante voto mayoritario de las etiquetas de los k vecinos más cercanos.
    - **Hybrid Classifier**: Clasificador que combina las predicciones del modelo Transformer y del k-NN Classifier mediante una combinación lineal ponderada. Para cada ejemplo, obtiene:
        1. Las probabilidades del k-NN mediante voto mayoritario de sus k vecinos más cercanos.
        2. Las probabilidades del Transformer mediante softmax de sus logits.
        3. Una combinación lineal con parámetro $\alpha$: $p_\text{comb}(y|x) = \alpha \cdot p_T(y|x) + (1 - \alpha) \cdot p_K(y|x)$.
        Donde $\alpha \in [0, 1]$ permite ajustar el balance entre ambos modelos ($\alpha=0$ equivale al Transformer puro, $\alpha=1$ al k-NN puro).
    - **Modelo student**: Modelo Transformer (`distilroberta-base`) al que se le aplica fine-tuning.

### Experimentos realizados en el notebook 🧪
- En el notebook, a parte entrenarse los diferentes modelos mencionados, se realizan las comparativas en cuanto a:
    - **Métricas de calidad** (entre todos los modelos implementados):
        - Exactitud (*accuracy*).
        - Macro F1.
        - Per-class F1.
    - **Tiempos de inferencia** de los modelos **teacher** y **student**.
    - **Espacio en disco ocupado** por el modelo **teacher** y el modelo **student**.

### Hardware utilizado y tiempo de ejecución ⏱
- El entrenamiento y evaluación de los modelos se ha realizado en un entorno con las siguientes características:
    - **Procesador**: AMD Ryzen 7 7800X3D.
    - **Memoria RAM**: 32 GB.
    - **Tarjeta gráfica**: NVIDIA GeForce RTX 5070 V2 (12 GB VRAM).
- El tiempo total de ejecución del notebook es de $\approx 3$ horas contando con que los siguientes elementos están guardados localmente:
    - Conjunto de datos armonizado (splits train, eval y test) compatible con transformer y con baseline.
    - Vectores densos de embeddings extraídos con el modelo Transformer teacher para el conjunto de datos.
    - Modelo Transformer teacher entrenado en la práctica anterior.
    - Modelo student.
    - Modelo baseline.
