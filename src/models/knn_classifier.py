import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from datasets import Dataset

from src.models.model import Model
from src.models.transformer_model import TransformerModel


class KNNClassifier(Model):
    def __init__(
        self,
        transformer_model: TransformerModel,
        device: torch.device,
        k: int
    ):
        """
        Constructor de la clase KNNClassifier

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
        Entrena el clasificador k-NN con el conjunto de entrenamiento dado.
        :param train_dataset: Conjunto de entrenamiento.
        :type train_dataset: dict | Dataset
        """
        self.train_dataset = train_dataset
        self.train_embeddings = train_embeddings if train_embeddings is not None else self._embed_fn(
            train_dataset)
        self.nearest_neighbors = NearestNeighbors(n_neighbors=self.k)
        self.nearest_neighbors.fit(self.train_embeddings)

    def predict(self, texts: dict | Dataset | None = None, embeddings: np.ndarray | None = None):
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

    def _embed_fn(self, texts: Dataset | dict):
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

    def _search(self, query_embeddings: np.ndarray | None = None):

        distances, indices = self.nearest_neighbors.kneighbors(
            query_embeddings, n_neighbors=self.k)

        texts = []
        labels = []
        for row_indices in indices:
            texts.append([self.train_dataset[i]["text"] for i in row_indices])
            labels.append([self.train_dataset[i]["label"]
                          for i in row_indices])

        return distances, indices, texts, labels
