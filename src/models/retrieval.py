
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm
from typing import List, Dict, Tuple

class DenseRetriever:
    def __init__(self, model_name_or_path: str, device: str = 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        print(f"Loading model from {model_name_or_path}...")
        try:
            # Try loading as a base model first
            self.model = AutoModel.from_pretrained(model_name_or_path).to(device)
        except:
            print("Could not load as AutoModel, trying AutoModelForSequenceClassification...")
            # If it fails (e.g. checkpoint has classification head keys), load as Classif and extrat base
            full_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
            if hasattr(full_model, 'roberta'):
                self.model = full_model.roberta.to(device)
            elif hasattr(full_model, 'bert'):
                self.model = full_model.bert.to(device)
            elif hasattr(full_model, 'distilbert'):
                self.model = full_model.distilbert.to(device)
            else:
                self.model = full_model.base_model.to(device)
        
        self.model.eval()
        self.index = None
        self.corpus_texts = None
        self.corpus_labels = None
        self.corpus_embeddings = None

    def embed_fn(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
                batch_texts = texts[i:i+batch_size]
                # Ensure texts are strings
                batch_texts = [str(t) for t in batch_texts]
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                
                # Use CLS token (index 0)
                # outputs.last_hidden_state shape: [batch, seq_len, hidden]
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        if embeddings:
            return np.vstack(embeddings)
        return np.array([])

    def build_index(self, texts: List[str], labels: List[any], n_neighbors: int = 5):
        print(f"Building index with {len(texts)} documents...")
        self.corpus_texts = np.array(texts, dtype=object)
        self.corpus_labels = np.array(labels)
        self.corpus_embeddings = self.embed_fn(texts)
        
        # Metric='cosine' computes distance = 1 - cosine_similarity
        self.index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
        self.index.fit(self.corpus_embeddings)
        print("Index building complete.")

    def search(self, query_texts: List[str], k: int = 5) -> Tuple[List[List[Dict]], np.ndarray, np.ndarray]:
        """
        Returns:
            results: List of List of Dicts (details)
            distances: numpy array of distances
            indices: numpy array of indices
        """
        query_embeddings = self.embed_fn(query_texts)
        if self.index is None:
            raise ValueError("Index not built! Call build_index first.")
            
        distances, indices = self.index.kneighbors(query_embeddings, n_neighbors=k)
        
        results = []
        for i in range(len(query_texts)):
            res = []
            for j in range(k):
                idx = indices[i][j]
                res.append({
                    'text': self.corpus_texts[idx],
                    'label': self.corpus_labels[idx],
                    'distance': distances[i][j]
                })
            results.append(res)
            
        return results, distances, indices
