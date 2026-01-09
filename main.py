"""
Práctica 2 - PLN2
=================

1. Módulo de Retrieval Denso
2. Clasificador k-NN sobre Embeddings
3. Clasificador Híbrido (RAG para Clasificación)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter
import torch

# Importar módulos propios
from src.models.retrieval import DenseRetriever


# =============================================================================
# Preparación del Entorno y Carga de Datos
# =============================================================================

def load_data():
    """Carga y divide los datos en train/val/test."""
    # Configurar el path a los datos (ajustar si es necesario)
    data_path = Path("../../practica1/pln2_practica1/data/en_song_lyrics_clear.csv")
    if not data_path.exists():
        # Fallback to local if copied
        data_path = Path("data/en_song_lyrics_clear.csv")

    print(f"Cargando datos desde {data_path}...")
    try:
        df = pd.read_csv(data_path)
        # Sample for quick dev if needed (uncomment to test speed)
        # df = df.groupby('label').sample(n=100, random_state=42)
    except FileNotFoundError:
        print("ERROR: No se encontró el fichero de datos. Verifica la ruta.")
        # Create dummy data for structure if loading fails
        df = pd.DataFrame({'text': ['sample text'] * 100, 'label': ['Pop'] * 100})

    # Split estratificado como en P1
    X = df['text']
    y = df['label']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# 1. Módulo de Retrieval Denso
# =============================================================================

def build_retriever(X_train, y_train, model_name='roberta-base', device='cpu', n_neighbors=5):
    """Construye el retriever denso con el índice k-NN."""
    retriever = DenseRetriever(model_name_or_path=model_name, device=device)
    # Construir índice con Train
    retriever.build_index(X_train.tolist(), y_train.tolist(), n_neighbors=n_neighbors)
    return retriever


def evaluate_retrieval(retriever, texts, labels, k=5):
    """Evaluación Básica (Precision@k, Recall@k)."""
    results, _, _ = retriever.search(texts, k=k)
    precisions = []
    # Recall es confuso en k-NN puro si no sabemos cuantos relevantes totales hay, 
    # aquí asumimos que 'relevantes' son los de la misma clase.
    
    for i, res in enumerate(results):
        target_label = labels[i]
        relevant_matches = sum(1 for item in res if item['label'] == target_label)
        precisions.append(relevant_matches / k)
    return np.mean(precisions)


# =============================================================================
# 2. Clasificador k-NN sobre Embeddings
# =============================================================================

class KNNClassifier:
    """Clasificador k-NN basado en embeddings."""
    
    def __init__(self, retriever, k=5):
        self.retriever = retriever
        self.k = k
        
    def predict(self, texts):
        results, _, _ = self.retriever.search(texts, k=self.k)
        preds = []
        for res in results:
            labels = [item['label'] for item in res]
            # Voto mayoritario
            most_common = Counter(labels).most_common(1)[0][0]
            preds.append(most_common)
        return preds


def evaluate_knn_classifier(knn_clf, X_test, y_test):
    """Evalúa el clasificador k-NN."""
    print("Evaluando Clasificador k-NN...")
    y_pred_knn = knn_clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred_knn)
    macro_f1 = f1_score(y_test, y_pred_knn, average='macro')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    return y_pred_knn, accuracy, macro_f1


# =============================================================================
# 3. Clasificador Híbrido (RAG)
# =============================================================================

def get_knn_probs(retriever, texts, k, all_labels):
    """Obtiene las probabilidades del k-NN como distribución de clases."""
    results, _, _ = retriever.search(texts, k=k)
    probs_list = []
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    
    for res in results:
        counts = Counter([item['label'] for item in res])
        probs = np.zeros(len(all_labels))
        for label, count in counts.items():
            if label in label_to_idx:
                probs[label_to_idx[label]] = count / k
        probs_list.append(probs)
    return np.array(probs_list)


def get_transformer_probs(model, tokenizer, texts, device):
    """
    Obtiene las probabilidades del Transformer.
    
    TODO: Implementar inferencia real con el modelo de clasificación.
    Esto requiere que el modelo cargado sea AutoModelForSequenceClassification.
    Si retriever.model es solo base, necesitamos cargar la cabecera o el modelo completo aparte.
    """
    # Placeholder - en la práctica real usar: model(inputs).logits.softmax(dim=-1)
    return np.random.rand(len(texts), 6)


def hybrid_predict(p_transformer, p_knn, alpha, all_labels):
    """
    Predicción híbrida combinando Transformer y k-NN.
    
    Fórmula: p_comb(y|x) = α * p_T(y|x) + (1-α) * p_K(y|x)
    """
    p_combined = alpha * p_transformer + (1 - alpha) * p_knn
    pred_indices = np.argmax(p_combined, axis=1)
    return [all_labels[i] for i in pred_indices]


def evaluate_hybrid(retriever, X_test, y_test, all_labels, alphas=[0.0, 0.25, 0.5, 0.75, 1.0], k=5):
    """Evalúa el clasificador híbrido para varios valores de alpha."""
    print("\nEvaluando Clasificador Híbrido para distintos valores de α...")
    
    # Obtener probabilidades
    p_knn = get_knn_probs(retriever, X_test, k, all_labels)
    # TODO: Reemplazar con probabilidades reales del Transformer
    p_transformer = get_transformer_probs(None, None, X_test, None)
    
    results = []
    for alpha in alphas:
        y_pred = hybrid_predict(p_transformer, p_knn, alpha, all_labels)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"α={alpha:.2f} - Accuracy: {accuracy:.4f}, Macro F1: {macro_f1:.4f}")
        results.append({'alpha': alpha, 'accuracy': accuracy, 'macro_f1': macro_f1})
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    """Función principal que ejecuta todos los experimentos."""
    # Configuración
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'roberta-base'  # Usar el mismo base que P1
    k = 5
    
    print("=" * 60)
    print("Práctica 2 - PLN2")
    print("=" * 60)
    
    # 1. Cargar datos
    print("\n[1/4] Cargando datos...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # 2. Construir retriever
    print(f"\n[2/4] Construyendo retriever denso (modelo: {model_name})...")
    retriever = build_retriever(X_train, y_train, model_name, device, n_neighbors=k)
    
    # 3. Evaluar retrieval (Precision@k)
    print(f"\n[3/4] Evaluando Precision@{k}...")
    # Usamos una muestra para rapidez
    test_subset_size = min(200, len(X_test))
    indices_test = np.random.choice(len(X_test), test_subset_size, replace=False)
    X_test_sample = X_test.iloc[indices_test].tolist()
    y_test_sample = y_test.iloc[indices_test].tolist()
    
    p_at_k = evaluate_retrieval(retriever, X_test_sample, y_test_sample, k=k)
    print(f"Precision@{k} (en muestra de {test_subset_size}): {p_at_k:.4f}")
    
    # 4. Clasificador k-NN
    print(f"\n[4/4] Evaluando clasificador k-NN (k={k})...")
    knn_clf = KNNClassifier(retriever, k=k)
    y_pred_knn, acc_knn, f1_knn = evaluate_knn_classifier(knn_clf, X_test_sample, y_test_sample)
    
    # 5. Clasificador Híbrido
    unique_labels = sorted(list(set(y_train)))
    print(f"\nEtiquetas encontradas: {unique_labels}")
    
    results_hybrid = evaluate_hybrid(
        retriever, X_test_sample, y_test_sample, unique_labels,
        alphas=[0.0, 0.25, 0.5, 0.75, 1.0], k=k
    )
    
    print("\n" + "=" * 60)
    print("Experimentos completados.")
    print("=" * 60)


if __name__ == "__main__":
    main()
