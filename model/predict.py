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
    model = AmpEmulatorModel.load_from_checkpoint("results/model.ckpt") # Tipo WaveNet2
    #model = AmpEmulatorModel.load_from_checkpoint("results/model.ckpt", num_channels=4, dilation_depth=10, dilation_repeat=1, kernel_size=3, lr=3e-3)
    model.eval()
    pred = model.inference(x, batch_size, sample_size)

    # Guardamos la predicción como un .wav
    wavfile.write("results/pred.wav", 44100, pred)

if __name__ == "__main__":
    predict()