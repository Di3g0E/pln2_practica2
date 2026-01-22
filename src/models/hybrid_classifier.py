import numpy as np
import torch
from datasets import Dataset
import tqdm

from src.models.transformer_model import TransformerModel
from src.models.knn_classifier import KNNClassifier


class HybridClassifier(KNNClassifier):
    """
    Clase que representa un clasificador híbrido que combina predicciones del modelo
    Transformer con voto mayoritario del clasificador k-NN, utilizando un parámetro
    alpha para ponderar ambas contribuciones.
    """

    def __init__(
        self,
        transformer_model: TransformerModel,
        device: torch.device,
        k: int,
        alpha: float
    ):
        """
        Constructor de la clase HybridClassifier.
        :param transformer_model: Transformer para obtener embeddings y predicciones.
        :type transformer_model: TransformerModel
        :param device: Dispositivo para PyTorch.
        :type device: torch.device
        :param k: Número de vecinos a considerar.
        :type k: int
        :param alpha: Peso para combinar predicciones k-NN y Transformer (0 <= alpha <= 1).
        :type alpha: float
        """
        super().__init__(transformer_model, device, k)
        self.alpha = alpha

    def predict(
        self,
        texts: dict | Dataset,
        embeddings: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Realiza la predicción combinando k-NN y el modelo Transformer.

        :param texts: Conjunto de datos para la predicción.
        :type texts: dict | Dataset
        :param embeddings: Embeddings precomputados para la predicción.
        :type embeddings: np.ndarray | None
        :return: Índices de los vecinos y etiquetas predichas.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        _, knn_indices, _, _ = self._search(
            embeddings if embeddings is not None else self._embed_fn(texts))
        transformer_probs = self.__transformer_inference(
            texts=texts, embeddings=embeddings)
        predictions = []
        for i in range(len(texts) if texts is not None else len(embeddings)):
            # Calcular probabilidades del k-NN (voto mayoritario)
            probs_knn = np.zeros(len(self.transformer_model.label2id))
            for neighbor_idx in knn_indices[i]:
                neighbor_label = self.train_dataset[neighbor_idx]["label"]
                probs_knn[neighbor_label] += 1
            probs_knn /= self.k

            # Obtener probabilidades del Transformer (ya están como array numpy)
            probs_transformer = transformer_probs[i]

            # Combinar probabilidades con alpha
            combined_probs = self.alpha * probs_knn + \
                (1 - self.alpha) * probs_transformer
            pred_label = np.argmax(combined_probs)
            predictions.append(pred_label)

        return knn_indices, np.array(predictions)

    def __transformer_inference(self, texts: dict | Dataset | None = None, embeddings: np.ndarray | None = None):
        """
        Realiza la inferencia con el modelo Transformer para los textos dados.
        Devuelve las probabilidades normalizadas (softmax) para clasificación multi-clase.

        :param texts: Textos para la inferencia.
        :type texts: dict | Dataset | None
        :param embeddings: Embeddings de los textos. Si se proporciona pero no textos, retorna distribución uniforme.
        :type embeddings: np.ndarray | None
        :return: Array (N, num_clases) con probabilidades para cada muestra y clase.
        :rtype: np.ndarray
        """
        # Si tenemos textos, hacer la inferencia del transformer
        if texts is not None:
            probs_list = []
            for text_id in tqdm.tqdm(range(len(texts)), desc="Inferencia con Transformer"):
                input_ids = torch.tensor(
                    texts[text_id]["input_ids"]).to(self.device)
                attention_mask = torch.tensor(
                    texts[text_id]["attention_mask"]).to(self.device)

                inputs = {"input_ids": input_ids.unsqueeze(
                    0), "attention_mask": attention_mask.unsqueeze(0)}

                with torch.no_grad():
                    outputs = self.transformer_model._model(**inputs)
                    logits = outputs.logits.squeeze(0)
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()

                probs_list.append(probs)

            return np.array(probs_list)

        # Si solo tenemos embeddings, retornar distribución uniforme
        elif embeddings is not None:
            num_classes = len(self.transformer_model.label2id)
            uniform_probs = np.ones(
                (len(embeddings), num_classes)) / num_classes
            return uniform_probs

        else:
            raise ValueError(
                "Se debe proporcionar textos o embeddings para la inferencia.")
