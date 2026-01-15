from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import matplotlib.pyplot as plt

from src.models.model import Model


class TransformerModel(Model):
    """
    Clase que representa el modelo basado en arquitectura Transformer
    que utiliza RoBERTa por defecto.
    """

    def __init__(
            self,
            output_dir: str | Path,
            label2id: dict[str: int],
            id2label: dict[int: str],
            model_name: str = 'roberta-base',
            num_labels: int = 4,
            epochs: int = 3,
            batch_train: int = 32,
            batch_eval: int = 64
    ):
        """
        Constructor de la clase TransformerModel.
        Args:
            output_dir (str | Path): Directorio donde se guardará el modelo.
            label2id (dict[str: int]): Mapeo de etiquetas a IDs.
            id2label (dict[int: str]): Mapeo de IDs a etiquetas.
            model_name (str): Nombre del modelo preentrenado de Hugging Face.
            num_labels (int): Número de etiquetas del modelo.
            epochs (int): Número de épocas para el entrenamiento.
            batch_train (int): Tamaño del batch para el entrenamiento.
            batch_eval (int): Tamaño del batch para la evaluación.
        """
        self.output_dir = output_dir
        self.label2id = label2id
        self.id2label = id2label
        self.model_name = model_name
        self.num_labels = num_labels
        self.epochs = epochs
        self.batch_train = batch_train
        self.batch_eval = batch_eval

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=self.num_labels)

        self.dataset = None

    def compute_metrics(self, eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        """
        Calcula las métricas de evaluación.
        Args:
            eval_pred: Predicciones del modelo.
        Output:
            dict[str, float]
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return {
            "accuracy": acc,
            "f1_macro": f1
        }

    def fit(self, train_dataset: Dataset, eval_dataset: Dataset):
        """
        Entrena el modelo con el conjunto de datos de entrenamiento y evalúa en el conjunto de validación.
        Args:
            train_dataset (Dataset): Conjunto de datos de entrenamiento.
            eval_dataset (Dataset): Conjunto de datos de validación.
        """
        print(f"- Iniciando entrenamiento del modelo {self.model_name}...")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_train,
            per_device_eval_batch_size=self.batch_eval,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            report_to="none"
        )

        self.trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()
        print("Entrenamiento finalizado.")

    def predict(self, X: Dataset) -> np.ndarray:
        """
        Realiza predicciones sobre el conjunto de datos dado.
        Args:
            X (Dataset): Conjunto de datos sobre el cual se realizarán las predicciones.
        Output:
            np.ndarray
        """
        print("- Realizando predicciones...")
        predictions = self.trainer.predict(X)
        preds = np.argmax(predictions.predictions, axis=-1)
        return preds

    def plot_training_curves(self) -> plt.Figure:
        """
        Visualiza la evolución del accuracy durante el entrenamiento.
        Output:
            plt.Figure
        """
        log_history = self.trainer.state.log_history

        eval_epochs = []
        eval_acc = []

        for log in log_history:
            if "epoch" in log:
                if "eval_accuracy" in log:
                    eval_epochs.append(log["epoch"])
                    eval_acc.append(log["eval_accuracy"])

        fig = plt.figure(figsize=(10, 6))

        plt.plot(eval_epochs, eval_acc, 'r-o',
                 label='Validation Accuracy')

        plt.xlabel('Épocas')
        plt.ylabel('Accuracy')
        plt.title('Curvas de Aprendizaje: Transformer (RoBERTa)')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return fig

    def evaluate(self, dataset: Dataset, y_pred: np.ndarray | None = None) -> dict:
        """
        Evalúa el modelo.
        Args:
            X (Dataset): Conjunto de datos para evaluar.
            y (np.ndarray): Etiquetas verdaderas.
        Output:
            dict
        """
        print("- Evaluando el modelo...")
        if y_pred is None:
            preds = self.trainer.predict(dataset)
        else:
            preds = y_pred

        acc = accuracy_score(dataset["label"], preds)
        f1_macro = f1_score(dataset["label"], preds, average='macro')
        f1_per_class = f1_score(dataset["label"], preds, average=None)
        cm = confusion_matrix(dataset["label"], preds)

        return {
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_per_class": f1_per_class,
            "confusion_matrix": cm
        }

    def plot_confusion_matrix(self, dataset: Dataset, y_pred: np.ndarray | None = None) -> plt.Figure:
        """
        Grafica la matriz de confusión del modelo.
        Args:
            dataset (Dataset): Conjunto de datos para evaluar.
        Output:
            plt.Figure
        """
        print("- Graficando matriz de confusión...")
        if y_pred is None:
            preds = self.predict(dataset)
        else:
            preds = y_pred

        cm = confusion_matrix(dataset["label"], preds)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=list(self.label2id.keys()))

        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
        plt.title('Matriz de Confusión')
        plt.show()

        return fig

    def save_model(self, path: str | Path):
        """
        Guarda el modelo en la ruta especificada.
        Args:
            path (str | Path): Ruta donde se guardará el modelo.
        """
        print(f"- Guardando modelo en {path}...")
        self.trainer.save_model(path)

    @staticmethod
    def load_model(path: str | Path, label2id: dict | None = None, id2label: dict | None = None) -> "TransformerModel":
        """
        Carga el modelo desde la ruta especificada.
        Args:
            path: Ruta del modelo guardado
            label2id (dict): Mapeo de etiquetas a IDs.
            id2label (dict): Mapeo de IDs a etiquetas.
        Returns:
            TransformerModel
        """
        print(f"- Cargando modelo desde {path}...")
        model = AutoModelForSequenceClassification.from_pretrained(path)
        instance = TransformerModel(
            output_dir=path,
            label2id=label2id or {},
            id2label=id2label or {},
            num_labels=model.config.num_labels
        )
        instance._model = model
        instance.trainer = Trainer(model=model)
        return instance
