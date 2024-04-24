import torch

from scipy.io import wavfile

from model import AmpEmulatorModel
from data import AmpEmulatorDataModule

def predict():
    """
    Función principal para predecir con el modelo.
    """
    # Cargamos los datos
    dataset = AmpEmulatorDataModule()
    dataset.setup()
    dataset.prepare_data()
    x = dataset.prepare_for_inference("predict")
    batch_size = dataset.batch_size
    sample_size = dataset.sample_size

    # Hacemos la inferencia
    model = AmpEmulatorModel.load_from_checkpoint("models/model.ckpt")
    model.eval()
    pred = model.inference(x, batch_size, sample_size)

    # Guardamos la predicción como un .wav
    wavfile.write("models/pred.wav", 44100, pred)

if __name__ == "__main__":
    predict()