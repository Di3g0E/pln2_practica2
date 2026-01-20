from abc import ABC, abstractmethod


class Model(ABC):
    """
    Interfaz que sirve como base para la implementación de los modelos
    utilizados en el proyecto.
    """

    @abstractmethod
    def fit(X, y):
        pass

    @abstractmethod
    def predict(X):
        pass
