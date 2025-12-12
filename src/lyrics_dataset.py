from pathlib import Path
import pandas as pd


class LyricsDataset:
    """
    Clase que representa un dataset de letras musicales
    """

    def __init__(self, csv_path: Path | str):
        """
        Constructor para instanciar la clase LyricsDataset

        Args:
            csv_path (Path | str): ruta al fichero CSV del que se cargarán los datos

        Raises:
            ValueError: si la ruta especificada no lleva a un CSV
        """
        if str(csv_path).endswith(".csv"):
            self.csv_path = csv_path
            self.__dataset = pd.read_csv(csv_path)
        else:
            raise ValueError(
                f"! La ruta {csv_path} no lleva a un fichero CSV.")

    def save(self, directory: Path | str, filename: str) -> Path | str:
        """
        Guarda el dataset en un fichero CSV.

        Args:
            directory (Path | str): directorio en que se almacenará el fichero.
            filename (str): nombre que recibirá el fichero.
        """
        f = (filename + ".csv") if not filename.endswith(".csv") else filename
        d = Path(directory) if type(directory) == str else directory
        d.mkdir(parents=True, exist_ok=True)
        path = d / f
        self.__dataset.to_csv(path)
