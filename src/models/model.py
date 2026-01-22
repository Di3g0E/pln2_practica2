from abc import ABC, abstractmethod


class Model(ABC):
    """
    Interfaz abstracta que sirve como base para la implementación de los modelos
    de clasificación utilizados en el proyecto. Define los métodos esenciales que
    todo modelo debe implementar.
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Entrena el modelo con los datos proporcionados.

        :param X: Características de entrenamiento.
        :type X: Cualquier tipo adecuado para las características de entrada.
        :param y: Etiquetas de entrenamiento.
        :type y: Cualquier tipo adecuado para las etiquetas de salida.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Realiza predicciones sobre nuevos datos.
        :param X: Características para las que se desean predicciones.
        :type X: Cualquier tipo adecuado para las características de entrada.
        :return: Predicciones del modelo.
        :rtype: Cualquier tipo adecuado para las predicciones de salida.
        """
        pass
