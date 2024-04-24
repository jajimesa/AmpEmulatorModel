import torch

from scipy.io import wavfile
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from model import AmpEmulatorModel
from data import AmpEmulatorDataModule

def test():
    """
    Función principal para probar el modelo.
    """
    # Cargamos los datos
    dataset = AmpEmulatorDataModule()
    dataset.setup()
    dataset.prepare_data()
    x = dataset.prepare_for_inference("test")
    batch_size = dataset.batch_size
    sample_size = dataset.sample_size

    # Tomamos los datos originales como referencia
    x_test = dataset.data["x_test"]
    mean = dataset.data["mean"]
    std = dataset.data["std"]

    y_test = dataset.data["y_test"]

    # Hacemos la inferencia
    model = AmpEmulatorModel.load_from_checkpoint("models/model.ckpt")
    model.eval()
    y_hat = model.inference(x, batch_size, sample_size)

    # Guardamos la predicción y los datos originales para la comparación
    wavfile.write("tests/y_hat.wav", 44100, y_hat)
    wavfile.write("tests/x_test.wav", 44100, x_test * std + mean)   # Deshacemos la estandarización
    wavfile.write("tests/y_test.wav", 44100, y_test)

def plot():
    """
    Función para visualizar los resultados del test.
    """
    # Cargamos las señales
    _, actual_signal = wavfile.read("tests/y_test.wav")
    _, pred_signal = wavfile.read("tests/y_hat.wav")

    # Calculo el error-to-signal ratio
    error_signal_ratio = mean_squared_error(actual_signal, pred_signal)

    # Calculo el error absoluto
    abs_error = np.abs(pred_signal - actual_signal)

    # Plot de las señales
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(actual_signal, label="Señal Real")
    plt.plot(pred_signal, label="Señal Predicha")
    plt.legend()
    plt.title("Comparación de Señales")

    # Plot del error absoluto a lo largo del tiempo
    plt.subplot(3, 1, 2)
    plt.plot(abs_error, color="red")
    plt.title("Error Absoluto")

    # Plot del espectrograma del error entre la señal predicha y la señal real
    plt.subplot(3, 1, 3)
    _, _, _, spectrogram = plt.specgram(pred_signal - actual_signal, Fs=44100)
    plt.title("Espectrograma del Error")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test()
    plot()