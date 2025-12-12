
import numpy as np
from collections import Counter
from typing import List, Optional

class HybridClassifier:
    def __init__(self, transformer_model, retriever, class_labels: List[any]):
        """
        Args:
            transformer_model: Object with predict_proba(texts) method.
            retriever: Object with search(texts, k) method.
            class_labels: List of all possible labels (in the order expected by transformer probs).
        """
        self.transformer_model = transformer_model
        self.retriever = retriever
        self.class_labels = class_labels
        self.label_to_idx = {l: i for i, l in enumerate(class_labels)}

    def predict_hybrid(self, texts: List[str], alpha: float = 0.5, k: int = 5):
        """
        Returns:
            preds_labels: List of predicted labels
            p_comb: Combined probabilities
            p_t: Transformer probabilities
            p_k: k-NN probabilities
        """
        # 1. P_T (Transformer)
        p_t = self.transformer_model.predict_proba(texts)
        
        # 2. P_K (k-NN)
        results, _, _ = self.retriever.search(texts, k=k)
        p_k = []
        
        for res in results:
            counts = Counter([item['label'] for item in res])
            probs = np.zeros(len(self.class_labels))
            for label, count in counts.items():
                if label in self.label_to_idx:
                    probs[self.label_to_idx[label]] = count / k
            p_k.append(probs)
        p_k = np.array(p_k)
        
        # 3. Combine
        # p_comb(y|x) = alpha * P_T(y|x) + (1 - alpha) * P_K(y|x)
        p_comb = alpha * p_t + (1 - alpha) * p_k
        
        preds_idx = np.argmax(p_comb, axis=1)
        preds_labels = [self.class_labels[i] for i in preds_idx]
        
        return preds_labels, p_comb, p_t, p_k
