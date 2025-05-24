from scipy.io import wavfile
from scipy import signal
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
    model = AmpEmulatorModel.load_from_checkpoint("results/model.ckpt") # Tipo WaveNet2
    #model = AmpEmulatorModel.load_from_checkpoint("results/model.ckpt", num_channels=4, dilation_depth=10, dilation_repeat=1, kernel_size=3, lr=3e-3)
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
    axes[0].set_title("Comparación de señales (ESR = {:.4f})".format(error_signal_ratio), fontsize=16, fontname='sans-serif')
    axes[0].set_ylabel("Amplitud", fontsize=12, fontname='sans-serif')
    axes[0].set_xlabel("Tiempo [s]", fontsize=12, fontname='sans-serif')
    axes[0].grid(True, linewidth=.8)
    axes[0].spines['bottom'].set_linewidth(1.2)  # Ajusta el grosor del eje x
    axes[0].spines['left'].set_linewidth(1.2)  # Ajusta el grosor del eje y
    axes[0].spines['top'].set_linewidth(1.2)
    axes[0].spines['right'].set_linewidth(1.2)

    # Plot del error absoluto a lo largo del tiempo con zoom
    axes[1].plot(np.arange(start_index, end_index) / sample_rate, abs_error[start_index:end_index], color="red")
    axes[1].set_title("Error absoluto (zoom)", fontsize=16, fontname='sans-serif')
    axes[1].set_xlabel("Tiempo [s]", fontsize=12, fontname='sans-serif')
    axes[1].set_ylabel("Error absoluto", fontsize=12, fontname='sans-serif')
    axes[1].grid(True, linewidth=.8)
    axes[1].spines['bottom'].set_linewidth(1.2)  # Ajusta el grosor del eje x
    axes[1].spines['left'].set_linewidth(1.2)  # Ajusta el grosor del eje y
    axes[1].spines['top'].set_linewidth(1.2)
    axes[1].spines['right'].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig("tests/comparacion_señales_error_zoom.pdf")
    plt.show()

    # Plot del espectrograma del error entre la señal predicha y la señal real
    plt.figure(figsize=(8, 6))
    frequencies, times, spectrogram = signal.spectrogram(y_hat - y_test, 44100)
    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-10), shading='auto', cmap='viridis')
    cbar = plt.colorbar(label='Potencia [dB]')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.label.set_fontname('sans-serif')
    plt.title("Espectrograma del error", fontsize=16, fontname='sans-serif')
    plt.xlabel("Tiempo [s]", fontsize=12, fontname='sans-serif')
    plt.ylabel("Frecuencia [Hz]", fontsize=12, fontname='sans-serif')
    plt.tight_layout()
    plt.savefig("tests/espectrograma_error.png")
    plt.show()

    # FFT de ambas señales
    freqs = np.fft.rfftfreq(len(y_test), 1 / sample_rate)
    Y_test = np.abs(np.fft.rfft(y_test))
    Y_hat = np.abs(np.fft.rfft(y_hat))
    Y_error = np.abs(Y_test - Y_hat)

    # Espectros de magnitud
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, 20 * np.log10(Y_test + 1e-10), label="Señal real", alpha=0.7)
    plt.plot(freqs, 20 * np.log10(Y_hat + 1e-10), label="Señal predicha", alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud [dB]")
    plt.title("Comparación de espectros de magnitud")
    plt.legend()
    plt.grid()
    plt.savefig("tests/comparacion_espectros.pdf")
    plt.show()

    # Energía del error a lo largo del tiempo
    window_size = sample_rate // 10  # 100 ms
    energy_error = [
        np.sum((y_hat[i : i + window_size] - y_test[i : i + window_size]) ** 2)
        for i in range(0, len(y_test), window_size)
    ]

    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(energy_error)) * (window_size / sample_rate), energy_error, color="red")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Energía del error")
    plt.title("Evolución temporal de la energía del error")
    plt.grid()
    plt.savefig("tests/energia_error.pdf")
    plt.show()

    # Correlación cruzada entre señal real y predicha
    max_index = np.argmax(np.abs(y_test))
    sample_rate = 44100
    zoom_duration = 0.01  # segundos (10 ms)
    zoom_samples = int(sample_rate * zoom_duration)
    start_index = max(0, max_index - zoom_samples // 2)
    end_index = min(len(y_test), max_index + zoom_samples // 2)
    if start_index < end_index:
        y_test_zoom = y_test[start_index:end_index]
        y_hat_zoom = y_hat[start_index:end_index]
    else:
        raise ValueError("Los índices start_index y end_index no son válidos.")
    corr = np.correlate(y_test_zoom, y_hat_zoom, mode="full")
    lags = np.arange(-len(y_hat_zoom) + 1, len(y_test_zoom))
    corr /= np.max(np.abs(corr))
    plt.figure(figsize=(8, 6))
    plt.plot(lags / sample_rate * 1000, corr, color="darkred")  # Convertimos lags a milisegundos
    plt.xlabel("Desfase [ms]")
    plt.ylabel("Correlación cruzada normalizada")
    plt.title("Correlación cruzada (zoom)")
    plt.axvline(0, color="black", linestyle="dashed")  # Línea en desfase = 0
    plt.grid()
    plt.savefig("tests/correlacion_cruzada_zoom.pdf")
    plt.show()

if __name__ == "__main__":
    test()
    plot()