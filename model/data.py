import torch

import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

from scipy.io import wavfile
import numpy as np

"""
Implementación en Pytorch Lightning del módulo de datos, que se encarga de cargar
los datos de entrenamiento y validación, y de construir los dataloaders.
"""

class AmpEmulatorDataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size=64, 
        num_workers=4,
        input_file="data/input.wav",
        output_file="data/output.wav",
        sample_time=100e-3,
        mu_law_companding=False
    ):
        """
        Constructor de la clase.

        Args:
            batch_size: Tamaño del lote.
            num_workers : Número de hilos trabajadores para cargar los datos.
            input_file: Ruta del archivo .wav de entrada.
            output_file: Ruta del archivo .wav de salida.
            sample_time: Duración de la muestra en segundos
            mu_law_compansion: Si se aplica la compansión mu-law a los datos.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_file = input_file
        self.output_file = output_file
        self.sample_time = sample_time
        self.mu_law_companding = mu_law_companding

        # Datos de entrenamiento, validación y test
        self.data = {}
        self.__train_ds = None
        self.__valid_ds = None
        self.__test_ds = None   

    # Métodos auxiliares

    def __normalize(self, data):
        """
        Función auxiliar que normaliza los datos con la norma del máximo (caso particular
        de la norma del supremo o norma infinito).

        Args:
            data: Tensor de datos a normalizar.

        Returns:
            Tensor de datos normalizados.
        """
        return data / max(max(data), abs(min(data)))
    
    def __split(self, x):
        """
        Método auxiliar que divide el array de numpy x en tres partes: 60% para entrenamiento, 20% para validación y 20% para test.

        Args:
            x (np.array): Array de numpy a dividir.

        Returns:
            Tres arrays de numpy con los datos divididos en entrenamiento, validación y test.
        """
        train, valid, test = np.split(x, [int(len(x) * 0.6), int(len(x) * 0.8)])
        return train, valid, test
    
    def __build_dataset(self, x, y):
        """
        Función auxiliar que construye un TensorDataset a partir de los tensores x e y, que pueden ser parejas
        de tensores de entrenamiento o de validación.

        Args:
            x: Tensor de datos de entrenamiento o validación.
            y: Tensor de datos de salida de entrenamiento o validación.

        Returns:
            TensorDataset construido a partir de los tensores x e y.
        """
        return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    
    # Métodos auxiliares para la compansión mu-law

    def __compress(self, data, mu=255):
        """
        Función auxiliar que aplica la compresión mu-law a los datos, transportándolos al rango [0, mu].
        
        Args:
            data: Tensor de datos comprendidos en [-1, 1] a los que aplicar la compresión mu-law.

        Returns:
            Tensor de datos comprimidos con la compresión mu-law.
        """
        return np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(1 + mu)

    def __decompress(self, data, mu=255):
        """
        Función auxiliar que aplica la descompresión mu-law a los datos, transportándolos al rango [-1, 1].
        
        Args:
            data: Tensor de datos comprendidos en [0, mu] a los que aplicar la expansión mu-law.

        Returns:
            Tensor de datos descomprimidos con la expansión mu-law.
        """
        return np.sign(data) * (1 / mu) * (np.power(1 + mu, np.abs(data)) - 1)

    # Métodos overriden de Lightning.LightningDataModule

    def setup(self, stage=None):
        """
        Método overriden que se encarga de cargar los datos y de manipularlos para que estén en el formato
        correcto para ser utilizados.
        """
        # Cargamos los ficheros .wav, nos devuelve la tasa de muestreo y los datos en formato numpy array
        in_rate, in_data = wavfile.read(self.input_file)
        out_rate, out_data = wavfile.read(self.output_file)
        assert in_rate == 44100, "La tasa de muestreo de los datos de entrada debe ser 44.1 kHz."
        assert in_rate == out_rate, "Las tasas de muestreo de in_rate y out_rate deben ser iguales."

        # Si los datos no tienen la misma longitud, los truncamos
        if len(in_data) < len(out_data):
            print("[AVISO] El audio de entrada es más corto que el audio de salida. Truncando audio de salida.")
            out_data = out_data[:len(in_data)]
        elif len(in_data) > len(out_data):
            print("[AVISO] El audio de salida es más corto que el audio de entrada. Truncando audio de entrada.")
            in_data = in_data[:len(out_data)]

        # Nos aseguramos que el audio esté en mono
        if len(in_data.shape) > 1:
            print("[AVISO] El audio de entrada no es mono. Seleccionando primer canal.")
            in_data = in_data[:, 0]
        if len(out_data.shape) > 1:
            print("[AVISO] El audio de salida no es mono. Seleccionando primer canal.")
            out_data = out_data[:, 0]

        # Convertimos de int16 a float32 para poder usar el modelo en un plugin VST3
        if in_data.dtype == np.int16:
            in_data = in_data.astype(np.float32) / 32767
        if out_data.dtype == np.int16:
            out_data = out_data.astype(np.float32) / 32767

        # Calculamos la longitud de la muestra y de los datos
        sample_size = int(in_rate * self.sample_time)
        data_length = len(in_data) - len(in_data) % sample_size

        # Normalizamos los datos
        in_data = self.__normalize(in_data)
        out_data = self.__normalize(out_data)

        # Dividimos los datos de entrada y de salida en muestras de tamaño sample_size
        x = in_data[:data_length].reshape((-1, 1, sample_size)).astype(np.float32)
        y = out_data[:data_length].reshape((-1, 1, sample_size)).astype(np.float32)

        # Dividimos los datos en entrenamiento, validación y test; y los metemos en un diccionario
        self.data["x_train"], self.data["x_valid"], self.data["x_test"] = self.__split(x)
        self.data["y_train"], self.data["y_valid"], self.data["y_test"] = self.__split(y)
        self.data["mean"], self.data["std"] = self.data["x_train"].mean(), self.data["x_train"].std()

        # Estandarizamos los datos con media 0 y desviación estándar 1
        self.data["x_train"] = (self.data["x_train"] - self.data["mean"]) / self.data["std"]
        self.data["x_valid"] = (self.data["x_valid"] - self.data["mean"]) / self.data["std"]
        self.data["x_test"] = (self.data["x_test"] - self.data["mean"]) / self.data["std"]

    def prepare_data(self):
        """
        Método que se encarga de construir los TensorDataSet de entrenamiento, validación y test.
        """
        self.__train_ds = self.__build_dataset(self.data["x_train"], self.data["y_train"])
        self.__valid_ds = self.__build_dataset(self.data["x_valid"], self.data["y_valid"])
        self.__test_ds = self.__build_dataset(self.data["x_test"], self.data["y_test"])

    def train_dataloader(self):
        """
        Método que devuelve el DataLoader de entrenamiento.

        Returns:
            DataLoader de entrenamiento.
        """
        return DataLoader(
            dataset = self.__train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,    # Mantiene la pool de hilos trabajadores abierta entre epochs para acelerar la carga de datos
            shuffle=True)
    
    def val_dataloader(self):
        """
        Método que devuelve el DataLoader de validación.

        Returns:
            DataLoader de validación.
        """
        return DataLoader(
            dataset = self.__valid_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,    # Mantiene la pool de hilos
            shuffle=False)
    
    def test_dataloader(self):
        """
        Método que devuelve el DataLoader de test.

        Returns:
            DataLoader de test.
        """
        return DataLoader(
            dataset = self.__test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,    # Mantiene la pool de hilos
            shuffle=False)
    
    def prepare_for_inference(self, inference_type="predict"):
        """
        Método que prepara los datos para hacer inferencia con el modelo. El dataset ha tenido que haber
        sido cargado previamente (setup y prepare_data han debido ser llamados).

        Args:
            inference_type: Tipo de inferencia a realizar. Puede ser 'predict' o 'test'.
        Returns:
            pad_x: Datos preparados para hacer inferencia con el modelo.
        """

        # Comprobamos que el parámetro inference_type sea correcto
        if not isinstance(inference_type, str) and inference_type not in ['predict', 'test']:
            raise ValueError("El parámetro 'inference_type' solo puede ser 'predict' o 'test'.")

        # Cargamos los datos
        data = self.data
        mean, std = data["mean"], data["std"]

        # Determinamos los datos "x" a utilizar en función del tipo de inferencia
        x = None
        if inference_type == "predict":
            # Cargamos el archivo .wav
            in_rate, in_data = wavfile.read(self.input_file)
            assert in_rate == 44100, "La tasa de muestreo de los datos de entrada debe ser 44.1 kHz."
            sample_size = int(in_rate * self.sample_time)
            data_length = len(in_data) - len(in_data) % sample_size
            # Dividimos los datos de entrada en muestras de tamaño sample_size
            in_data = in_data[:data_length].reshape((-1, 1, sample_size)).astype(np.float32)

            # Estandarizamos los datos con media 0 y desviación estándar 1 
            in_data = (in_data - mean) / std
            x = in_data

        elif inference_type == "test":
            x = data["x_test"]

        # Concatenamos las samples entre sí y añadimos un padding de ceros
        """
        Ejemplo:

        x = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        prev_sample = np.array([[0, 0, 0],
                                [1, 2, 3],
                                [4, 5, 6]])
        pad_x = np.array([[[0, 0, 0, 1, 2, 3],
                           [0, 0, 0, 4, 5, 6],
                           [0, 0, 0, 7, 8, 9]]])                    
        """
        prev_sample = np.concatenate((np.zeros_like(x[0:1]), x[:-1]), axis=0)
        pad_x = np.concatenate((prev_sample, x), axis=2)

        return pad_x