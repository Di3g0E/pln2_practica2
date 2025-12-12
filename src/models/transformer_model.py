import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict

class TransformerTrainer:
    def __init__(self, model_name='roberta-base', num_labels=6, seed=42):
        self.model_name = model_name
        self.num_labels = num_labels
        self.seed = seed
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dataset = None
        
    def prepare_data(self, df):
        """
        Prepara los datos convirtiendo el DataFrame a HuggingFace Dataset,
        tokenizando y dividiendo en train/val/test.
        """
        print("Preparando datos para Transformer...")
        
        # 1. Split estratificado (igual que en baseline)
        X = df['text']
        y = df['label']
        
        # Train (70%) / Temp (30%)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=self.seed, stratify=y
        )
        
        # Val (15%) / Test (15%)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.seed, stratify=y_temp
        )
        
        # Crear DataFrames intermedios
        train_df = pd.DataFrame({'text': X_train, 'label': y_train})
        val_df = pd.DataFrame({'text': X_val, 'label': y_val})
        test_df = pd.DataFrame({'text': X_test, 'label': y_test})
        
        # Convertir a HF Datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Tokenización
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
            
        print("Tokenizando datasets (esto puede tardar un poco)...")
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_val = val_dataset.map(tokenize_function, batched=True)
        tokenized_test = test_dataset.map(tokenize_function, batched=True)
        
        self.dataset = DatasetDict({
            'train': tokenized_train,
            'validation': tokenized_val,
            'test': tokenized_test
        })
        print("Datos preparados.")

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        return {
            'accuracy': acc,
            'f1_macro': f1
        }

    def train(self, output_dir='./results', epochs=3, batch_size=16):
        """
        Configura y ejecuta el entrenamiento.
        """
        print(f"Iniciando entrenamiento del modelo {self.model_name}...")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            report_to="none" # Desactivar wandb/mlflow si no se usan
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset['train'],
            eval_dataset=self.dataset['validation'],
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()
        print("Entrenamiento finalizado.")

    def save_model(self, path):
        """
        Guarda el modelo y el tokenizador en la ruta especificada.
        """
        print(f"Guardando modelo en {path}...")
        self.trainer.save_model(path)
        self.tokenizer.save_pretrained(path)
        print("Modelo guardado exitosamente.")

    def evaluate(self):
        """
        Evalúa el modelo en el conjunto de test.
        """
        print("Evaluando en conjunto de Test...")
        results = self.trainer.evaluate(self.dataset['test'])
        print(results)
        
        # Predicciones detalladas para reporte
        predictions = self.trainer.predict(self.dataset['test'])
        preds = np.argmax(predictions.predictions, axis=-1)
        
        print("\nReporte de Clasificación:")
        print(classification_report(self.dataset['test']['label'], preds))
        
        return results

    def predict_proba(self, texts, max_length=128):
        """
        Devuelve las probabilidades (Softmax) para una lista de textos.
        Útil para el modelo híbrido.
        """
        device = next(self.model.parameters()).device
        self.model.eval()
        
        # Tokenizar
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        return probs.cpu().numpy()
