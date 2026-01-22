import time
from pathlib import Path
import pickle
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from src.models.model import Model


class BaselineModel(Model):
    """
    Clase que representa al modelo que se utiliza como *baseline*, que emplea
    regresión logística con SGD.
    """

    def __init__(
            self,
            epochs: int = 15,
            batch_size: int = 128
    ):
        """
        Constructor de la clase BaselineModel.

        :param epochs: Número de épocas para el entrenamiento.
        :type epochs: int
        :param batch_size: Tamaño del mini-batch para el entrenamiento.
        :type batch_size: int
        """
        self.epochs = epochs
        self.batch_size = batch_size

        self.clf = SGDClassifier(
            loss='log_loss',
            penalty='l2',
            max_iter=1,
            learning_rate='optimal',
            n_jobs=-1,
            warm_start=True,
            early_stopping=False
        )

        self.history = {"train_acc": [], "val_acc": [],
                        'train_loss': [], 'val_loss': []}

    def fit(self, X: csr_matrix, y: np.ndarray, X_eval: csr_matrix | None = None, y_eval: np.ndarray | None = None):
        """
        Entrena el modelo utilizando mini-batch SGD con validación opcional.

        :param X: Matriz de características de entrenamiento.
        :type X: csr_matrix
        :param y: Vector de etiquetas de entrenamiento.
        :type y: np.ndarray
        :param X_eval: Matriz de características de validación.
        :type X_eval: csr_matrix | None
        :param y_eval: Vector de etiquetas de validación.
        :type y_eval: np.ndarray | None
        """
        print(f"- Iniciando entrenamiento de {self.epochs} epochs...")

        classes = np.unique(y)

        start_time = time.time()

        for epoch in range(self.epochs):
            batch_acc = []
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y[i:i+self.batch_size]

                self.clf.partial_fit(
                    X=X_batch, y=y_batch, classes=classes)

                batch_acc.append(self.clf.score(X_batch, y_batch))

            train_acc = np.mean(batch_acc)

            # Evaluación (sin entrenar, solo predecir)
            if X_eval is not None and y_eval is not None:
                val_acc = self.clf.score(X_eval, y_eval)
                self.history['val_acc'].append(val_acc)
            self.history['train_acc'].append(train_acc)

            print(
                f"Época {epoch+1}/{self.epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        print(
            f"Entrenamiento finalizado en {time.time() - start_time:.2f} segundos.")

    def predict(self, X: csr_matrix) -> np.ndarray:
        """
        Realiza predicciones utilizando el modelo entrenado.

        :param X: Matriz de características sobre las que predecir.
        :type X: csr_matrix
        :return: Predicciones del modelo.
        :rtype: np.ndarray
        """

        return self.clf.predict(X)

    def plot_training_curves(self) -> plt.Figure:
        """
        Visualiza la evolución del accuracy de entrenamiento y validación.

        :return: Figura de matplotlib con las curvas de aprendizaje.
        :rtype: plt.Figure
        """
        epochs = range(1, len(self.history['train_acc']) + 1)

        fig = plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_acc'],
                 'b-o', label='Train Accuracy')
        plt.plot(epochs, self.history['val_acc'],
                 'r-o', label='Validation Accuracy')

        plt.title('Curvas de Aprendizaje: Regresión Logística (SGD)')
        plt.xlabel('Épocas')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()
        return fig

    def evaluate(self, X: csr_matrix, y: np.ndarray, y_pred: np.ndarray | None = None) -> dict:
        """
        Evalúa el modelo con métricas de accuracy, F1 y matriz de confusión.

        :param X: Matriz de características.
        :type X: csr_matrix
        :param y: Vector de etiquetas verdaderas.
        :type y: np.ndarray
        :param y_pred: Predicciones precomputadas (opcional).
        :type y_pred: np.ndarray | None
        :return: Diccionario con las métricas de evaluación.
        :rtype: dict
        """
        if y_pred is None:
            preds = self.predict(X)
        else:
            preds = y_pred

        acc = accuracy_score(y, preds)
        f1_macro = f1_score(y, preds, average="macro")
        f1_per_class = f1_score(y, preds, average=None)
        cm = confusion_matrix(y, preds)
        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_per_class": f1_per_class,
            "confusion_matrix": cm
        }

    def plot_confusion_matrix(self, X: csr_matrix, y: np.ndarray, y_pred: np.ndarray | None = None) -> plt.Figure:
        """
        Grafica la matriz de confusión del modelo.

        :param X: Matriz de características.
        :type X: csr_matrix
        :param y: Vector de etiquetas verdaderas.
        :type y: np.ndarray
        :param y_pred: Predicciones precomputadas (opcional).
        :type y_pred: np.ndarray | None
        :return: Figura de matplotlib con la matriz de confusión.
        :rtype: plt.Figure
        """
        if y_pred is None:
            preds = self.predict(X)
        else:
            preds = y_pred

        cm = confusion_matrix(y, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax)
        plt.title('Matriz de Confusión')
        plt.show()
        return fig

    def save_model(self, path: str | Path):
        """
        Guarda el modelo en la ruta especificada.

        :param path: Ruta donde guardar el modelo.
        :type path: str | Path
        """
        print(f"Guardando modelo en {path}...")
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)

    @staticmethod
    def load_model(path: str | Path) -> "BaselineModel":
        """
        Carga el modelo desde la ruta especificada.

        :param path: Ruta desde donde cargar el modelo.
        :type path: str | Path
        :return: Instancia del modelo cargado.
        :rtype: BaselineModel
        """
        path = Path(path)
        print(f"- Cargando modelo desde {path}...")
        with open(path, "rb") as f:
            clf = pickle.load(f)
        model = BaselineModel()
        model.clf = clf
        return model
