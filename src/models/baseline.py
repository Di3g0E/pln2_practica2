import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import time

class BaselineTrainer:
    def __init__(self, df, seed=42):
        self.df = df
        self.seed = seed
        self.vectorizer = None
        self.clf = None
        self.history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
        
        # Datos procesados
        self.X_train_vec = None
        self.X_val_vec = None
        self.X_test_vec = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X_test_raw = None

    def prepare_data(self):
        """
        Punto 8: Partición Train (70%) / Val (15%) / Test (15%) con estratificación.
        """
        print("Separando datos en Train, Val y Test...")
        X = self.df['text']
        y = self.df['label']

        # 1. Primero separamos Train (70%) y Temp (30%)
        X_train_raw, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=self.seed, stratify=y
        )
        
        # 2. Separamos Temp en Val (50% de temp -> 15% total) y Test (50% de temp -> 15% total)
        X_val_raw, X_test_raw, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.seed, stratify=y_temp
        )
        
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.X_test_raw = X_test_raw

        print(f"Tamaños -> Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")

        # 3. Vectorización (Punto 7: TF-IDF)
        # Es crucial hacer fit SOLO en train para evitar data leakage
        print("Vectorizando textos (TF-IDF)...")
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        )
        
        # Transformamos todo a matrices numéricas
        self.X_train_vec = self.vectorizer.fit_transform(X_train_raw)
        self.X_val_vec = self.vectorizer.transform(X_val_raw)
        self.X_test_vec = self.vectorizer.transform(X_test_raw)
        print("Vectorización completada")

    def train_iterative(self, epochs=15, batch_size=512):
        """
        Entrena el modelo iterativamente para registrar métricas en cada época.
        Usamos SGDClassifier con loss='log_loss' (Equivalente a Regresión Logística).
        """
        print(f"\nIniciando entrenamiento por {epochs} épocas...")
        
        # Configuración del modelo iterativo
        self.clf = SGDClassifier(
            loss='log_loss',       # Log_loss = Logistic Regression
            penalty='l2',          # Regularización estándar
            max_iter=1,            # Importante: 1 iteración por llamada a partial_fit
            learning_rate='optimal',
            random_state=self.seed,
            n_jobs=-1,
            warm_start=True        # Mantiene los pesos anteriores
        )
        
        classes = np.unique(self.y_train) # Necesario para la primera llamada a partial_fit
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Entrenamos una época completa
            # Nota: SGD procesa por lotes automáticamente, pero partial_fit acepta todo el bloque
            # y lo gestiona internamente.
            self.clf.partial_fit(self.X_train_vec, self.y_train, classes=classes)
            
            # Evaluación (Sin entrenar, solo predecir)
            train_acc = self.clf.score(self.X_train_vec, self.y_train)
            val_acc = self.clf.score(self.X_val_vec, self.y_val)
            
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            print(f"Época {epoch+1}/{epochs} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

        print(f"Entrenamiento finalizado en {time.time() - start_time:.2f} segundos.")

    def plot_training_curves(self):
        """
        Visualiza la evolución del Accuracy durante el entrenamiento.
        """
        epochs = range(1, len(self.history['train_acc']) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.history['train_acc'], 'b-o', label='Train Accuracy')
        plt.plot(epochs, self.history['val_acc'], 'r-o', label='Validation Accuracy')
        
        plt.title('Curvas de Aprendizaje: Regresión Logística (SGD)')
        plt.xlabel('Épocas')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def evaluate_final(self, id2label=None):
        """
        Evaluación final sobre el conjunto de TEST (nunca visto durante el bucle).
        """
        print("\n--- Evaluación Final en TEST ---")
        y_pred = self.clf.predict(self.X_test_vec)
        
        print(classification_report(self.y_test, y_pred, target_names=id2label.values() if id2label else None))
        
        # Matriz de confusión
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        labels = list(id2label.values()) if id2label else None
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Matriz de Confusión (Test Set)')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        plt.show()
