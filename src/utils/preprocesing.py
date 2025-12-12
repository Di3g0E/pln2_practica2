import pandas as pd
import json
import pathlib

from src.utils.io import DatasetManager # Importamos tu clase existente

class DataProcessor:
    def __init__(self, dataset_manager=DatasetManager(), seed=42):
        self.dm = dataset_manager
        self.df = None
        self.seed = seed
        
    def load_and_unify(self, df_name="en_song_lyrics.csv"):
        """
        Target Schema: ['text', 'label']
        """

        # Asumimos que ya ejecutamos dm.download()
        try:
            df = self.dm.load(df_name) # Ajusta el nombre según la descarga real
            # Renombrar columnas al estándar
            self.df = df.rename(columns={'lyrics': 'text', 'tag': 'label'})
            print(f"Dataset cargado: {self.df.shape}")
        
        except Exception as e:
            print(f"Aviso: No se pudo cargar datos: {e}")

    def normalize_labels(self, threshold=80000):
        """
        Punto 3 revisado: Normaliza texto y convierte a valores numéricos.
        Entrada: ['rap', 'rb', 'rock', 'pop', 'misc', 'country']
        Salida: Enteros [0, 1, 2, 3, 4, 5]
        """
        if self.df is None: return

        # Eliminamos etiquetas no mapeadas (NaN)
        n_dropped = self.df['label'].isna().sum()
        if n_dropped > 0:
            print(f"Eliminando {n_dropped} filas (desconocidos).")
            self.df = self.df.dropna(subset=['label'])
        else: 
            print(f"No hay filas con valores desconocidos.")

        # 2. Conversión a Numérico (Label Encoding)
        # Ordenamos alfabéticamente para asegurar determinismo: Country=0, Hip Hop=1, etc.
        unique_labels = sorted(self.df['label'].unique())
        
        # Creamos los diccionarios de mapeo
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for i, label in enumerate(unique_labels)}
        
        print(f"Asignando IDs numéricos: {self.label2id}")
        
        # Sobrescribimos la columna 'label' con el entero
        self.df['label'] = self.df['label'].map(self.label2id)
        
        # Filtrar clases con muy pocos ejemplos (ruido)
        conteo = self.df['label'].value_counts()
        clases_validas = conteo[conteo > threshold].index # Umbral mínimo
        self.df = self.df[self.df['label'].isin(clases_validas)]
        
        print(f"Clases resultantes: {self.df['label'].unique()}")

    def clean_and_deduplicate(self, min_len=25):
        """
        Punto 4: Deduplicar textos para evitar data leakage.
        """
        print("\nLimpiando y deduplicando...")
        original_size = len(self.df)
        
        # Eliminar nulos
        self.df = self.df.dropna(subset=['text', 'label'])
        
        # Eliminar duplicados exactos en el texto
        self.df = self.df.drop_duplicates(subset=['text'], keep='first')
        
        # Limpieza básica de texto (eliminar etiquetas de metadatos comunes en lyrics)
        # Ejemplo: [Chorus], [Intro]
        self.df['text'] = self.df['text'].str.replace(r'\[.*?\]', '', regex=True)
        
        # Eliminar textos demasiado cortos tras limpieza
        self.df = self.df[self.df['text'].str.len() > min_len]
        
        final_size = len(self.df)
        print(f"--> Filas eliminadas: {original_size - final_size}")
        print(f"--> Dataset limpio: {final_size}")

    def handle_imbalance(self):
        """
        Punto 5: Controlar el desbalance (Undersampling estricto).
        Estrategia: Reducir todas las clases al tamaño de la clase minoritaria.
        """
        if self.df is None: return

        print("\nGestionando desbalance de clases...")
        conteo_inicial = self.df['label'].value_counts()
        print("Distribución original:")
        print(conteo_inicial)
        
        # 1. Identificar el tamaño de la clase más pequeña
        min_samples = conteo_inicial.min()

        # 2. Aplicar undersampling
        # Agrupamos por etiqueta y tomamos una muestra aleatoria de tamaño 'min_samples' de cada grupo
        dfs_list = []
        for label, group in self.df.groupby('label'):
            dfs_list.append(group.sample(n=min_samples, random_state=self.seed))
        
        # 3. Concatenar y mezclar (shuffle)
        self.df = pd.concat(dfs_list).sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        print("\nDistribución tras balanceo estricto:")
        print(self.df['label'].value_counts())

    def save(self, filename="dataset_harmonized.csv"):
        self.dm.save_dataset(self.df, filename)

    def get_df(self):
        return self.df

    def set_df(self, df):
        self.df = df

    def save_mappings(self, filename="label_mapping.json"):
        """
        Guarda los diccionarios id2label y label2id en un archivo JSON.
        """
        if not hasattr(self, 'id2label') or not self.id2label:
            print("Error: No hay mapeos definidos. Ejecuta normalize_labels() primero.")
            return

        # Definimos la ruta (guardamos en la misma carpeta que los datos)
        filepath = self.dm.data_path / filename
        
        data = {
            "id2label": self.id2label,
            "label2id": self.label2id
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            print(f"Mapeos guardados exitosamente en: {filepath}")
        except Exception as e:
            print(f"Error al guardar mapeos: {e}")

    def load_mappings(self, filename="label_mapping.json"):
        """
        Carga los diccionarios id2label y label2id desde un archivo JSON.
        """
        filepath = self.dm.data_path / filename
        
        if not filepath.exists():
            print(f"Error: El archivo {filepath} no existe.")
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convertir claves de id2label a enteros (JSON guarda claves como strings)
            self.id2label = {int(k): v for k, v in data["id2label"].items()}
            self.label2id = data["label2id"]
            
            print(f"Mapeos cargados exitosamente desde: {filepath}")
        except Exception as e:
            print(f"Error al cargar mapeos: {e}")
