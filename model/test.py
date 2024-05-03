from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
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

def error_to_signal_ratio(y, y_hat):
    """
    Método que implementa la función de pérdida ESR (Error-to-Signal Ratio) sobre arrays de numpy.

    Args:
        y (np.array): Array de numpy con la señal original.
        y_hat (np.array): Array de numpy con la señal predicha.
        
    Returns:
        ESR (float): Error-to-Signal Ratio.
    """    
    y, y_hat = pre_emphasis_filter(y), pre_emphasis_filter(y_hat)

    # ¡Añadimos un pequeño valor para evitar la división por cero!
    return np.sum(np.power(y - y_hat, 2)) / (np.sum(np.power(y, 2) + 1e-10))

def pre_emphasis_filter(x, coeff=0.95):
    """
    Método que implementa el filtro de pre-énfasis de paso alto de primer orden.

    Args:
        x (np.array): Array de numpy con la señal de audio.
        alpha (float): Coeficiente de pre-énfasis. Por defecto 0.95.

    Returns:
        out (np.array): Array de numpy con las señales de audio enfatizadas en el rango de frecuencias medias y altas.
    """
    return np.concatenate([x, np.subtract(x, np.multiply(x, coeff))])

def plot():
    """
    Función para graficar la comparativa entre las señales reales y predichas, así como el error absoluto y
    el espectrograma del error. Guarda las figuras en archivos .pdf.
    """
    # Cargamos las señales
    _, y_test = wavfile.read("tests/y_test.wav")
    _, y_hat = wavfile.read("tests/y_hat.wav")

    # Encontramos el índice del valor máximo de amplitud en y_test
    max_index = np.argmax(np.abs(y_test))

    # Calculamos el intervalo para el zoom alrededor del valor máximo
    sample_rate = 44100
    zoom_duration = 0.01  # segundos
    zoom_samples = int(sample_rate * zoom_duration)
    start_index = max(0, max_index - zoom_samples // 2)
    end_index = min(len(y_test), max_index + zoom_samples // 2)

    # Calculo el error-to-signal ratio
    error_signal_ratio = error_to_signal_ratio(y_test, y_hat)

    # Calculo el error absoluto
    abs_error = np.abs(y_hat - y_test)

    # Plot de las señales con zoom
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(np.arange(start_index, end_index) / sample_rate, y_test[start_index:end_index], label="Señal real")
    axes[0].plot(np.arange(start_index, end_index) / sample_rate, y_hat[start_index:end_index], label="Señal predicha")
    axes[0].legend()
    axes[0].set_title("Comparación de señales (Zoom)", fontsize=16, fontname='sans-serif')
    axes[0].set_ylabel("Amplitud", fontsize=12, fontname='sans-serif')
    axes[0].set_xlabel("Tiempo [s]", fontsize=12, fontname='sans-serif')

    # Plot del error absoluto a lo largo del tiempo con zoom
    axes[1].plot(np.arange(start_index, end_index) / sample_rate, abs_error[start_index:end_index], color="red")
    axes[1].set_title("Error absoluto (Zoom)", fontsize=16, fontname='sans-serif')
    axes[1].set_xlabel("Tiempo [s]", fontsize=12, fontname='sans-serif')
    axes[1].set_ylabel("Error absoluto", fontsize=12, fontname='sans-serif')

    plt.tight_layout()
    plt.savefig("tests/comparacion_señales_error_zoom.pdf")
    plt.show()

    # Plot del espectrograma del error entre la señal predicha y la señal real
    plt.figure(figsize=(8, 6))
    _, _, _, spectrogram = plt.specgram(y_hat - y_test, Fs=44100, cmap='viridis', scale='dB')
    plt.title("Espectrograma del error", fontsize=16, fontname='sans-serif')
    plt.xlabel("Tiempo [s]", fontsize=12, fontname='sans-serif')
    plt.ylabel("Frecuencia [Hz]", fontsize=12, fontname='sans-serif')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("tests/espectrograma_error.pdf")
    plt.show()

if __name__ == "__main__":
    test()
    plot()
