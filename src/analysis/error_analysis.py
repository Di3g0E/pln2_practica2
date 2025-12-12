import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class ErrorAnalyzer:
    def __init__(self, model, vectorizer, X_test, y_test, id2label, df_test_raw=None):
        """
        Clase para analizar errores y explicabilidad del modelo.
        
        Args:
            model: Modelo entrenado (debe tener coef_ o feature_importances_)
            vectorizer: Vectorizador ajustado (TfidfVectorizer)
            X_test: Matriz de características de test (vectorizada)
            y_test: Etiquetas reales de test
            id2label: Diccionario {id: etiqueta_texto}
            df_test_raw: DataFrame original de test (opcional, para ver textos completos)
        """
        self.model = model
        self.vectorizer = vectorizer
        self.X_test = X_test
        self.y_test = y_test
        self.id2label = id2label
        self.df_test_raw = df_test_raw
        
        # Predecir una vez para usar en todos los métodos
        self.y_pred = self.model.predict(self.X_test)
        
        # Obtener nombres de características una vez
        self.feature_names = np.array(self.vectorizer.get_feature_names_out())

    def plot_confusion_matrix(self):
        """Muestra la matriz de confusión visual."""
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(10, 8))
        labels = [self.id2label[i] for i in sorted(self.id2label.keys())]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Matriz de Confusión')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        plt.show()

    def get_misclassified_examples(self, n=10):
        """
        Devuelve un DataFrame con n ejemplos mal clasificados.
        """
        if self.df_test_raw is None:
            print("Error: Se necesita df_test_raw para mostrar ejemplos.")
            return None
            
        # Índices donde la predicción fue incorrecta
        # Nota: X_test y y_test deben estar alineados con df_test_raw
        # Si df_test_raw es el split de test original, los índices deberían coincidir si no se resetearon
        
        # Creamos un DF temporal con real, pred y texto
        # Asumimos que df_test_raw es una Serie o DF con 'text'
        texts = self.df_test_raw if isinstance(self.df_test_raw, pd.Series) else self.df_test_raw['text']
        
        # Aseguramos que los índices estén reseteados para alinear con y_pred (numpy array)
        texts = texts.reset_index(drop=True)
        y_test_reset = self.y_test.reset_index(drop=True) if isinstance(self.y_test, pd.Series) else pd.Series(self.y_test)
        
        errors_mask = y_test_reset != self.y_pred
        
        error_indices = np.where(errors_mask)[0]
        
        if len(error_indices) == 0:
            print("¡Increíble! No hay errores.")
            return pd.DataFrame()
            
        # Seleccionar n errores aleatorios
        selected_indices = np.random.choice(error_indices, min(n, len(error_indices)), replace=False)
        
        results = []
        for idx in selected_indices:
            real_id = y_test_reset.iloc[idx]
            pred_id = self.y_pred[idx]
            text = texts.iloc[idx]
            
            results.append({
                'text_snippet': text[:200] + "...", # Primeros 200 caracteres
                'true_label': self.id2label[real_id],
                'pred_label': self.id2label[pred_id],
                'original_index': idx
            })
            
        return pd.DataFrame(results)

    def explain_prediction(self, text_idx, top_n=10):
        """
        Explica una predicción específica mostrando las palabras que más contribuyeron
        a la clase predicha y a la clase real (si son diferentes).
        
        Args:
            text_idx: Índice del ejemplo en el conjunto de test (0 a len(test)-1)
        """
        # Obtener vector del ejemplo
        x_vec = self.X_test[text_idx]
        
        # Obtener predicción y real
        pred_id = self.y_pred[text_idx]
        
        # Manejo de y_test si es Series o array
        if isinstance(self.y_test, pd.Series):
            real_id = self.y_test.iloc[text_idx]
        else:
            real_id = self.y_test[text_idx]
            
        print(f"--- Explicación para ejemplo {text_idx} ---")
        print(f"Predicho: {self.id2label[pred_id]} | Real: {self.id2label[real_id]}")
        
        # Para modelos lineales (SGD, LogisticRegression, LinearSVC)
        if hasattr(self.model, 'coef_'):
            # coef_ shape: (n_classes, n_features) para multiclase
            # Ojo: Si es binario es (1, n_features)
            
            # Contribución = valor_feature * coeficiente
            # x_vec es sparse, lo convertimos a denso para multiplicar o usamos indices
            feature_indices = x_vec.indices
            feature_values = x_vec.data
            
            # Analizar clase predicha
            self._print_top_features(pred_id, feature_indices, feature_values, f"Top palabras para clase PREDICHA ({self.id2label[pred_id]})", top_n)
            
            # Si es error, analizar clase real
            if pred_id != real_id:
                self._print_top_features(real_id, feature_indices, feature_values, f"Top palabras para clase REAL ({self.id2label[real_id]})", top_n)
        else:
            print("Este modelo no soporta explicabilidad basada en coeficientes lineales directos.")

    def _print_top_features(self, class_id, feature_indices, feature_values, title, top_n):
        """Helper para imprimir features importantes"""
        # Coeficientes para esta clase
        class_coefs = self.model.coef_[class_id]
        
        # Calcular contribución: peso * valor tfidf
        contributions = class_coefs[feature_indices] * feature_values
        
        # Ordenar por contribución absoluta o positiva? 
        # Queremos ver qué empujó a esta clase -> contribución positiva más alta
        sorted_idx = np.argsort(contributions)[::-1] # Descendente
        
        print(f"\n{title}:")
        print(f"{'Palabra':<20} {'Contribución':<10} {'TF-IDF':<10} {'Coef':<10}")
        print("-" * 55)
        
        count = 0
        for i in sorted_idx:
            if count >= top_n: break
            
            # Solo mostrar contribuciones positivas (que apoyan la clase)
            if contributions[i] <= 0: continue
            
            feat_idx = feature_indices[i]
            word = self.feature_names[feat_idx]
            contrib = contributions[i]
            tfidf = feature_values[i]
            coef = class_coefs[feat_idx]
            
            print(f"{word:<20} {contrib:.4f}       {tfidf:.4f}       {coef:.4f}")
            count += 1

    def analyze_subgroups(self):
        """
        Analiza el rendimiento en subgrupos basados en características del texto.
        Ejemplo: Longitud del texto.
        """
        if self.df_test_raw is None: return
        
        texts = self.df_test_raw if isinstance(self.df_test_raw, pd.Series) else self.df_test_raw['text']
        texts = texts.reset_index(drop=True)
        y_test_reset = self.y_test.reset_index(drop=True) if isinstance(self.y_test, pd.Series) else pd.Series(self.y_test)
        
        # Calcular longitud
        lengths = texts.str.len()
        
        # Crear bins
        df_analysis = pd.DataFrame({
            'length': lengths,
            'correct': (y_test_reset == self.y_pred)
        })
        
        # Definir rangos
        bins = [0, 500, 1000, 2000, 5000, 100000]
        labels = ['Muy Corto (<500)', 'Corto (500-1k)', 'Medio (1k-2k)', 'Largo (2k-5k)', 'Muy Largo (>5k)']
        
        df_analysis['length_group'] = pd.cut(df_analysis['length'], bins=bins, labels=labels)
        
        print("\n--- Análisis de Error por Longitud de Texto ---")
        grouped = df_analysis.groupby('length_group', observed=True)['correct'].agg(['count', 'mean'])
        grouped.columns = ['Total Muestras', 'Accuracy']
        print(grouped)
