import os
import shutil # Necesitamos shutil para mover archivos y borrar directorios
import pathlib
import pandas as pd
import kagglehub

class DatasetManager:
    def __init__(self, dir_path="data"):
        self.data_path = pathlib.Path(dir_path).resolve()
        self._prepare_directory()

    def _prepare_directory(self):
        self.data_path.mkdir(parents=True, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = str(self.data_path)
        print(f"Directorio de destino deseado para CSVs: {self.data_path}")

    def download(self, df1=True, df2=True, df3=True):
        if df1:
            self._download_and_move_csv(
                dataset_handle="saurabhshahane/music-dataset-1950-to-2019",
                label="Dataset 1"
            )

        if df2:
            self._download_and_move_csv(
                dataset_handle="carlosgdcj/genius-song-lyrics-with-language-information",
                label="Dataset 2"
            )

        if df3:
            self._download_and_move_csv(
                dataset_handle="chloeliu/lyrics",
                label="Dataset 2"
            )

        self._cleanup_cache()

    def _download_and_move_csv(self, dataset_handle, label):
        print(f"\nDescargando {dataset_handle}...")

        try:
            downloaded_path = kagglehub.dataset_download(dataset_handle)
            print(f"Descarga inicial completada en: {downloaded_path}")

            csv_files = list(pathlib.Path(downloaded_path).rglob('*.csv'))

            if csv_files:
                source_csv = csv_files[0]
                destination_csv = self.data_path / source_csv.name

                print(f"Moviendo '{source_csv.name}' a '{destination_csv}'...")
                shutil.move(str(source_csv), str(destination_csv))
                print(f"CSV de {label} guardado en: {destination_csv}")
            else:
                print(f"No se encontró ningún archivo CSV en {downloaded_path} para {dataset_handle}.")

        except Exception as e:
            print(f"Error al descargar o procesar {dataset_handle}: {e}")

    def _cleanup_cache(self):
        cleanup_path = self.data_path / "datasets"
        if cleanup_path.exists() and cleanup_path.is_dir():
            print(f"\nLimpiando directorio de caché de kagglehub: {cleanup_path}")
            try:
                shutil.rmtree(cleanup_path)
                print("Limpieza completada.")
            except OSError as e:
                print(f"Error al limpiar el directorio {cleanup_path}: {e}")

    def load(self, filename="tcc_ceds_music.csv"):
        file_path = self.data_path / filename
        return pd.read_csv(file_path)
    
    def save_dataset(self, df, filename="filtered_data.csv", index=False):
        """
        Guarda un DataFrame filtrado en la carpeta de datos definida en la clase.
        
        Args:
            df (pd.DataFrame): El DataFrame que quieres guardar.
            filename (str): El nombre que tendrá el archivo (ej: 'mi_filtro.csv').
            index (bool): Si se debe guardar el índice numérico de pandas (False por defecto).
        """
        if df is None or df.empty:
            print("Advertencia: El DataFrame está vacío o es None. No se guardará nada.")
            return

        destination_path = self.data_path / filename
        
        try:
            print(f"Guardando dataset filtrado en: {destination_path}...")
            df.to_csv(destination_path, index=index)
            print(f"Archivo guardado correctamente.")
        except Exception as e:
            print(f"Error al guardar el archivo: {e}")
