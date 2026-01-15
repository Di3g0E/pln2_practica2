from typing import Callable
from pathlib import Path
import pickle
import re
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
from scipy.sparse import csr_matrix, save_npz, load_npz
from datasets import Dataset
from transformers import AutoTokenizer

from src.utils.lyrics_dataset_config import LyricsDatasetConfig


class LyricsDataProcessor:
    def __init__(
            self,
            output_dir: str | Path,
            datasets: list[pd.DataFrame],
            dataset_configs: list[LyricsDatasetConfig],
            genre_map: dict[str, str],
            label2id: dict[str: int],
            id2label: dict[int: str],
            clean_lyrics_fns: list[Callable[[str], str]] | None = None,
            eval_split: float | None = 0.15,
            test_split: float | None = 0.15,
            transformer_model_name: str = "roberta-base",
            device: str | torch.device = "cpu"
    ):
        """
        Constructor de la clase LyricsDataProcessor.
        Args:
            output_dir (str | Path): Directorio donde se guardarán los datos procesados.
            datasets (list[pd.DataFrame]): Lista de DataFrames con los datasets a procesar
            dataset_configs (list[LyricsDatasetConfig]): Configuraciones de los datasets.
            genre_map (dict[str, str]): Mapeo de géneros musicales.
            label2id (dict[str: int]): Mapeo de etiquetas a IDs.
            id2label (dict[int: str]): Mapeo de IDs a etiquetas.
            clean_lyrics_fns (list[Callable[[str], str]] | None): Lista de funciones para limpiar los lyrics de cada dataset.
            eval_split (float | None): Proporción del dataset para validación. Si es None, no se crea un conjunto de validación.
            test_split (float | None): Proporción del dataset para test. Si es None, no se crea un conjunto de test.
            transformer_model_name (str): Nombre del modelo transformer para tokenización.
            device (str | torch.device): Dispositivo donde se ejecutarán las operaciones (cpu o cuda).
        """
        self.output_dir = Path(output_dir)
        self.__datasets = datasets.copy()
        self.dataset_configs = dataset_configs
        self.genre_map = genre_map
        self.label2id = label2id
        self.id2label = id2label
        self.clean_lyrics_fns = clean_lyrics_fns if clean_lyrics_fns is not None else [
            None for _ in range(len(self.__datasets))]
        self.eval_split = eval_split
        self.test_split = test_split
        self.vectorizer = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words=self.dataset_configs[0].vectorizer_config["stop_words"],
            sublinear_tf=True
        )
        self.transformer_tokenizer = AutoTokenizer.from_pretrained(
            transformer_model_name)
        self.device = device

        self.transformer_tokenize_fn = (lambda examples: self.transformer_tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=128))

        self.__X_train_raw = None
        self.__X_train_baseline = None
        self.__train_dataset_transformer = None

        self.__X_eval_raw = None
        self.__X_eval_baseline = None
        self.__eval_dataset_transformer = None

        self.__X_test_raw = None
        self.__X_test_baseline = None
        self.__test_dataset_transformer = None

        self.__y_train_raw = None
        self.__y_eval_raw = None
        self.__y_test_raw = None

    def __rename_columns(self):
        """
        Renombra las columnas de los datasets según la configuración de cada dataset.
        """
        cols_maps = [config.cols_map for config in self.dataset_configs]
        for dataset, cols_map in zip(self.__datasets, cols_maps):
            dataset.rename(columns=cols_map, inplace=True)
        print("    - Renombrado de columnas realizado.")

    def __normalize_label(self):
        """
        Normaliza la columna "label" de los datasets, convirtiendo a minúsculas y eliminando espacios.
        """
        for dataset in self.__datasets:
            dataset["label"] = (
                dataset["label"]
                .astype(str)
                .str.lower()
                .str.strip()
            )

        print("    - Normalización de label realizada.")

    def __filter_language(self):
        """
        Filtra los datasets por el idioma establecido en la configuración.
        """
        filtered = []
        for i, dataset in enumerate(self.__datasets):
            config = self.dataset_configs[i]
            language_col = config.language_col
            target_language = config.target_language
            if language_col and target_language:
                samples_before = len(dataset)
                dataset = dataset[dataset[language_col]
                                  == target_language].copy()
                dataset.drop(columns=[language_col], inplace=True)
                samples_after = len(dataset)
                print(
                    f"    - Filtrado por idioma realizado. Antes --> {samples_before} muestras; Después --> {samples_after} muestras.")
            filtered.append(dataset)
        self.__datasets = filtered

    def __clean_genius(self, text):
        """
        Limpieza para el dataset 'Genius Song Lyrics'.
        """
        if not type(text) == str:
            return ""

        # Convertimos el texto a minúsculas
        text = text.lower()

        # Eliminamos las etiquetas [Chorus], [Intro], etc.
        text = re.sub(r"\[.*?\]", " ", text)

        # Convertimos los saltos de línea a '\n'
        text = text.replace("\r\n", '\n').replace('\r', '\n')

        # Eliminamos caracteres no alfanuméricos (excepto apostrofes y saltos)
        text = re.sub(r"[^a-z0-9\n\s']", " ", text)

        # Eliminación de whitespaces seguidos
        text = re.sub(r"[ ]{2,}", " ", text)

        # Eliminación de saltos de línea seguidos
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def __clean_multilingual(self, text):
        """
        Limpieza para el dataset 'Multi-Lingual Lyrics for Genre Classification'.
        """
        if not type(text) == str:
            return ""

        # Convertimos el texto a minúsculas
        text = text.lower()

        # Convertimos los saltos de línea a '\n'
        text = text.replace("\r\n", '\n').replace('\r', '\n')

        # Eliminación de whitespaces seguidos
        text = re.sub(r"[ ]{2,}", " ", text)

        # Eliminación de saltos de línea seguidos
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def __clean_lyrics(self):
        """
        Limpia los lyrics de los datasets aplicando las funciones de limpieza correspondientes.
        """
        for i, dataset in enumerate(self.__datasets):
            config = self.dataset_configs[i]
            name = config.name
            clean_lyrics_fn = self.clean_lyrics_fns[i]
            if name == "Genius Song Lyrics":
                dataset["text"] = dataset["text"].apply(
                    self.__clean_genius)
            elif name == "Multi-Lingual Lyrics for Genre Classification":
                dataset["text"].apply(
                    self.__clean_multilingual)
            else:
                if not self.clean_lyrics_fns:
                    raise RuntimeError(
                        "- No se ha especificado ninguna función para limpiar los lyrics.")
                dataset["text"] = dataset["text"].apply(
                    clean_lyrics_fn)

        print(f"    - Limpieza de lyrics realizada.")

    def __remove_na(self):
        """
        Elimina las muestras con valores nulos en los datasets.
        """
        samples_before = [len(dataset) for dataset in self.__datasets]
        self.__datasets = [dataset.dropna() for dataset in self.__datasets]
        samples_after = [len(dataset) for dataset in self.__datasets]
        for (sb, sa) in zip(samples_before, samples_after):
            print(
                f"    - Eliminación de muestras con atributos nulos realizada. Antes --> {sb} muestras; Después --> {sa} muestras.")

    def __map_genres(self):
        """
        Mapea los géneros a sus identificadores numéricos en los datasets.
        """
        for dataset in self.__datasets:
            dataset["label"] = dataset["label"].map(
                self.genre_map)
            dataset["label"] = dataset["label"].map(self.label2id)
            dataset.dropna(subset=["label"], inplace=True)
        print("    - Mapeo de géneros realizado.")

    def __unify_and_deduplicate_datasets(self):
        """
        Unifica los *datasets* y elimina muestras duplicadas.
        """
        self.__unified = pd.concat(self.__datasets)
        del self.__datasets
        self.__unified.drop_duplicates(
            subset=["text"], keep="first", inplace=True)

    def __handle_imbalance(self):
        """
        Maneja el desbalance de clases en el *dataset* unificado.
        """
        min_class_count = self.__unified["label"].value_counts().min()
        balanced_dfs = []
        for label in self.__unified["label"].unique():
            class_df = self.__unified[self.__unified["label"] == label]
            resampled = resample(class_df, n_samples=min_class_count)
            balanced_dfs.append(resampled)

        self.__unified = pd.concat(balanced_dfs, ignore_index=True)
        self.__unified = self.__unified.sample(frac=1).reset_index(drop=True)

        self.__X = self.__unified["text"]
        self.__y = self.__unified["label"].astype(int)

        del self.__unified

    def __split_dataset(self):
        """
        Divide el *dataset* en conjuntos de entrenamiento, evaluación y test según las proporciones especificadas.
        """
        if self.eval_split is not None and self.test_split is None:
            self.__X_test_raw = None
            self.__y_test_raw = None

            self.__X_train_raw, self.__X_eval_raw, self.__y_train_raw, self.__y_eval_raw = train_test_split(
                self.__X, self.__y, test_size=self.eval_split, stratify=self.__y)
            del self.__X
            del self.__y

        elif self.eval_split is not None and self.test_split is not None:
            self.__X_train_raw, X_temp, self.__y_train_raw, y_temp = train_test_split(
                self.__X, self.__y, test_size=(self.eval_split + self.test_split), stratify=self.__y)

            self.__X_eval_raw, self.__X_test_raw, self.__y_eval_raw, self.__y_test_raw = train_test_split(
                X_temp, y_temp, test_size=(self.test_split / (self.eval_split + self.test_split)), stratify=y_temp
            )
            del self.__X
            del self.__y

        elif self.eval_split is None and self.test_split is not None:
            self.__X_eval_raw = None
            self.__y_eval_raw = None

            self.__X_train_raw, self.__X_test_raw, self.__y_train_raw, self.__y_test_raw = train_test_split(
                self.__X, self.__y, test_size=self.test_split, stratify=self.__y)
            del self.__X
            del self.__y

        else:
            self.__X_eval_raw = None
            self.__y_eval_raw = None
            self.__X_test_raw = None
            self.__y_test_raw = None
            self.__X_train_raw = self.__X
            self.__y_train_raw = self.__y

    def __encode_splits(self):
        """
        Codifica los conjuntos de datos divididos utilizando el vectorizador y el tokenizador.
        """
        self.__X_train_baseline = self.vectorizer.fit_transform(
            self.__X_train_raw)

        self.__train_dataset_transformer = Dataset.from_pandas(
            pd.DataFrame({"text": self.__X_train_raw, "label": self.__y_train_raw}))
        self.__train_dataset_transformer = self.__train_dataset_transformer.map(
            self.transformer_tokenize_fn, batched=True)

        if self.__X_eval_raw is not None:
            self.__X_eval_baseline = self.vectorizer.transform(
                self.__X_eval_raw)
            self.__eval_dataset_transformer = Dataset.from_pandas(
                pd.DataFrame({"text": self.__X_eval_raw, "label": self.__y_eval_raw}))
            self.__eval_dataset_transformer = self.__eval_dataset_transformer.map(
                self.transformer_tokenize_fn, batched=True)

        if self.__X_test_raw is not None:
            self.__X_test_baseline = self.vectorizer.transform(
                self.__X_test_raw)
            self.__test_dataset_transformer = Dataset.from_pandas(pd.DataFrame(
                {"text": self.__X_test_raw, "label": self.__y_test_raw}))
            self.__test_dataset_transformer = self.__test_dataset_transformer.map(
                self.transformer_tokenize_fn, batched=True)

        del self.__X_train_raw
        del self.__X_eval_raw
        del self.__X_test_raw

    def harmonize_pipeline(self):
        """
        Ejecuta la pipeline completa de armonización de los datos.
        """
        print(
            f"- Ejecutando pipeline de armonización para {[config.name for config in self.dataset_configs]}...")

        self.__rename_columns()
        self.__normalize_label()
        self.__filter_language()
        self.__clean_lyrics()
        self.__remove_na()
        self.__map_genres()
        self.__unify_and_deduplicate_datasets()
        self.__handle_imbalance()
        self.__split_dataset()
        self.__encode_splits()

        print(f"- Armonización completa.")

    def get_train_data(self, return_format: str = "baseline") -> tuple[csr_matrix, np.ndarray] | Dataset:
        """
        Obtiene los datos de entrenamiento en el formato especificado.
        Args:
            return_format (str): Formato de retorno de los datos. Puede ser "baseline" o "transformer".
        Output:
            tuple[csr_matrix, np.ndarray] | Dataset
        """
        if return_format == "baseline":
            if self.__X_train_baseline is None:
                raise RuntimeError(
                    "- No se han vectorizado los datos de entrenamiento.")

            return self.__X_train_baseline, self.__y_train_raw.values.copy()

        elif return_format == "transformer":
            if self.__train_dataset_transformer is None:
                raise RuntimeError(
                    "- No se han tokenizado los datos de entrenamiento.")

            return self.__train_dataset_transformer

        else:
            raise RuntimeError(
                f"- El formato 'return_format={return_format}' no está soportado. Utiliza 'baseline' o 'transformer'.")

    def get_eval_data(self, return_format: str = "baseline"):
        """
        Obtiene los datos de validación en el formato especificado.

        """
        if return_format == "baseline":
            if self.__X_eval_baseline is None:
                raise RuntimeError(
                    "- No se han vectorizado los datos de validación.")

            return self.__X_eval_baseline, self.__y_eval_raw.values.copy()

        elif return_format == "transformer":
            if self.__eval_dataset_transformer is None:
                raise RuntimeError(
                    "- No se han tokenizado los datos de validación.")

            return self.__eval_dataset_transformer

        else:
            raise RuntimeError(
                f"- El formato 'return_format={return_format}' no está soportado. Utiliza 'baseline' o 'transformer'.")

    def get_test_data(self, return_format: str = "baseline"):
        if return_format == "baseline":
            if self.__X_test_baseline is None:
                raise RuntimeError(
                    "- No se han vectorizado los datos de test.")

            return self.__X_test_baseline, self.__y_test_raw.values.copy()

        elif return_format == "transformer":
            if self.__test_dataset_transformer is None:
                raise RuntimeError(
                    "- No se han tokenizado los datos de test.")

            return self.__test_dataset_transformer

        else:
            raise RuntimeError(
                f"- El formato 'return_format={return_format}' no está soportado. Utiliza 'baseline' o 'transformer'.")

    def save_baseline_data(self):
        # Guardar vectorizer
        with open(self.output_dir / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)

        save_npz(self.output_dir / "X_train_baseline.npz",
                 self.__X_train_baseline)
        if self.__X_eval_baseline is not None:
            save_npz(self.output_dir / "X_eval_baseline.npz",
                     self.__X_eval_baseline)
        if self.__X_test_baseline is not None:
            save_npz(self.output_dir / "X_test_baseline.npz",
                     self.__X_test_baseline)

        np.save(self.output_dir / "y_train.npy", self.__y_train_raw.values)
        if self.__y_eval_raw is not None:
            np.save(self.output_dir / "y_eval.npy", self.__y_eval_raw.values)
        if self.__y_test_raw is not None:
            np.save(self.output_dir / "y_test.npy", self.__y_test_raw.values)

        print(f"- Datos de baseline guardados en {self.output_dir}.")

    def save_transformer_data(self):
        self.__train_dataset_transformer.save_to_disk(
            self.output_dir / "train_dataset_transformer")
        self.__eval_dataset_transformer.save_to_disk(
            self.output_dir / "eval_dataset_transformer")
        self.__test_dataset_transformer.save_to_disk(
            self.output_dir / "test_dataset_transformer")

        print(f"- Datos de transformer guardados en {self.output_dir}.")

    def save_all(self):
        self.save_baseline_data()
        self.save_transformer_data()

    @staticmethod
    def load_data_baseline(path: str | Path):
        return load_npz(path)

    @staticmethod
    def load_label(path: str | Path):
        return np.load(path)

    @staticmethod
    def load_dataset_transformer(path: str | Path):
        return Dataset.load_from_disk(path)
