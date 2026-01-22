import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from datasets import Dataset

from src.models.model import Model
from src.models.transformer_model import TransformerModel


class KNNClassifier(Model):
    """
    Clase que representa un clasificador k-NN que utiliza embeddings obtenidos
    de un modelo Transformer para realizar clasificación basada en similitud.
    """

    def __init__(
        self,
        transformer_model: TransformerModel,
        device: torch.device,
        k: int
    ):
        """
        Constructor de la clase KNNClassifier.

        :param transformer_model: Transformer para obtener embeddings.
        :type transformer_model: TransformerModel
        :param device: Dispositivo para PyTorch.
        :type device: torch.device
        :param k: Número de vecinos a considerar.
        :type k: int
        """
        self.transformer_model = transformer_model
        self.device = device
        self.k = k
        self.nearest_neighbors = None

    def fit(self, train_dataset: dict | Dataset, train_embeddings: np.ndarray | None = None):
        """
        Entrena el clasificador k-NN construyendo el índice de vecinos más cercanos.

        :param train_dataset: Conjunto de datos de entrenamiento.
        :type train_dataset: dict | Dataset
        :param train_embeddings: Embeddings precomputados para el conjunto de entrenamiento.
        :type train_embeddings: np.ndarray | None
        """
        self.train_dataset = train_dataset
        self.train_embeddings = train_embeddings if train_embeddings is not None else self._embed_fn(
            train_dataset)
        self.nearest_neighbors = NearestNeighbors(n_neighbors=self.k)
        self.nearest_neighbors.fit(self.train_embeddings)

    def predict(self, texts: dict | Dataset | None = None, embeddings: np.ndarray | None = None):
        """
        Predice las etiquetas utilizando el clasificador k-NN.

        :param texts: Conjunto de datos para la predicción.
        :type texts: dict | Dataset | None
        :param embeddings: Embeddings precomputados para la predicción.
        :type embeddings: np.ndarray | None
        :return: Índices de los vecinos y etiquetas predichas.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        if texts is None and embeddings is None:
            raise ValueError(
                "Se debe proporcionar un conjunto de datos o embeddings para la predicción.")
        embs = embeddings if embeddings is not None else self._embed_fn(
            texts)
        _, indices, _, _ = self._search(embs)
        predictions = []
        for neighbor_indices in indices:
            neighbor_labels = [self.train_dataset[i]["label"]
                               for i in neighbor_indices]
            pred_label = max(set(neighbor_labels), key=neighbor_labels.count)
            predictions.append(pred_label)
        return indices, np.array(predictions)

    def _embed_fn(self, texts: Dataset | dict) -> np.ndarray:
        """
        Obtiene los embeddings del Transformer para un conjunto de datos.
        :param texts: Conjunto de datos para obtener embeddings.
        :type texts: Dataset | dict
        :return: Matriz de embeddings.
        :rtype: np.ndarray
        """
        embeddings = []
        for text_id in tqdm.tqdm(range(len(texts)), desc="Obteniendo embeddings"):
            input_ids = torch.tensor(
                texts[text_id]["input_ids"]).to(self.device)
            attention_mask = torch.tensor(
                texts[text_id]["attention_mask"]).to(self.device)

            inputs = {"input_ids": input_ids.unsqueeze(
                0), "attention_mask": attention_mask.unsqueeze(0)}

            with torch.no_grad():
                outputs = self.transformer_model._model(
                    **inputs, output_hidden_states=True)
                e = outputs.hidden_states[-1][:, 0, :]

            embeddings.append(e.cpu().numpy())

        return np.vstack(embeddings)

    def _search(self, query_embeddings: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, list[list[str]], list[list[int]]]:
        """
        Realiza la búsqueda de los k vecinos más cercanos para los embeddings dados.

        :param query_embeddings: Embeddings de consulta.
        :type query_embeddings: np.ndarray | None
        :return: Distancias e índices de los vecinos más cercanos.
        :rtype: tuple[np.ndarray, np.ndarray, list[list[str]], list[list[int]]]
        """
        distances, indices = self.nearest_neighbors.kneighbors(
            query_embeddings, n_neighbors=self.k)

        texts = []
        labels = []
        for row_indices in indices:
            texts.append([self.train_dataset[i]["text"] for i in row_indices])
            labels.append([self.train_dataset[i]["label"]
                          for i in row_indices])

        return distances, indices, texts, labels
