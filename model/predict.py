import torch

from scipy.io import wavfile
import numpy as np
from tqdm import tqdm

from model import AmpEmulatorModel
from data import AmpEmulatorDataModule

def predict():
    """
    Función principal para predecir con el modelo.
    """
    model = AmpEmulatorModel.load_from_checkpoint("models/model.ckpt")
    model.eval()    # Ponemos el modelo en modo de evaluación

    # Cargamos los datos
    dataset = AmpEmulatorDataModule()
    dataset.setup()
    dataset.prepare_data()
    x = dataset.prepare_for_inference("predict")

    # Hacemos la predicción
    with torch.no_grad():
        pred = []
        batches = x.shape[0] // dataset.batch_size

        for batch in tqdm(np.array_split(x, batches)):    # Mostramos una barra de progreso
            pred.append(model(torch.from_numpy(batch)).numpy())

        pred = np.concatenate(pred)
        pred = pred[:, :, -x.shape[2] :]

    # Guardamos la predicción como un .wav
    wavfile.write("models/pred.wav", 44100, pred)

if __name__ == "__main__":
    #predict()

    model = AmpEmulatorModel.load_from_checkpoint("models/model.ckpt")
    model.eval()    # Ponemos el modelo en modo de evaluación

    # Cargamos los datos
    dataset = AmpEmulatorDataModule()
    dataset.setup()
    dataset.prepare_data()

    data = dataset.data

    mean, std = data["mean"], data["std"]

    # Cargamos el archivo .wav
    in_rate, in_data = wavfile.read(dataset.input_file)
    assert in_rate == 44100, "La tasa de muestreo de los datos de entrada debe ser 44.1 kHz."
    sample_size = int(in_rate * dataset.sample_time)
    data_length = len(in_data) - len(in_data) % sample_size

    # Dividimos los datos de entrada en muestras de tamaño sample_size
    in_data = in_data[:data_length].reshape((-1, 1, sample_size)).astype(np.float32)

    # Estandarizamos los datos con media 0 y desviación estándar 1 
    in_data = (in_data - mean) / std

    # Concatenamos las samples entre sí
    prev_sample = np.concatenate((np.zeros_like(in_data[0:1]), in_data[:-1]), axis=0)
    pad_in_data = np.concatenate((prev_sample, in_data), axis=2)

    # Hacemos la predicción
    with torch.no_grad():
        pred = []
        batches = pad_in_data.shape[0] // dataset.batch_size

        for x in tqdm(np.array_split(pad_in_data, batches)):    # Mostramos una barra de progreso
            pred.append(model(torch.from_numpy(x)).numpy())

        pred = np.concatenate(pred)
        pred = pred[:, :, -in_data.shape[2] :]

    # Guardamos la predicción como un .wav
    wavfile.write("models/pred.wav", in_rate, pred)