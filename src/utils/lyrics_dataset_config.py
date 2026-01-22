from pathlib import Path
import json


class LyricsDatasetConfig:
    """
    Clase que representa la configuración de un dataset de lyrics.
    """

    def __init__(self, config_path: str | Path):
        """
        Constructor de la clase LyricsDatasetConfig.
        :param config_path: Ruta al fichero JSON con la configuración del dataset.
        :type config_path: str | Path
        """
        self.config_path = Path(config_path)

        try:
            with open(self.config_path, 'r') as f:
                config_object = json.load(f)
            self.name = config_object["name"]
            self.csv_name = config_object["csv_name"]
            self.kagglehub_handle = config_object["kagglehub_handle"]
            self.csv_path = config_object["csv_path"] if config_object["csv_path"] else None
            self.cols_map = config_object["cols_map"]
            self.language_col = config_object["language_col"] if config_object["language_col"] else None
            self.target_language = config_object["target_language"] if config_object["target_language"] else None
            self.vectorizer_config = config_object["vectorizer_config"]

            if self.language_col and not self.target_language:
                raise RuntimeError(
                    f"- Se ha especificado una columna de idioma pero no un idioma objetivo.")

            if not self.language_col and self.target_language:
                raise RuntimeError(
                    f"- Se ha especificado un idioma objetivo pero no la columna de idioma.")

        except Exception as e:
            print(
                f"- Error al cargar el fichero JSON con la configuración: {e}")
