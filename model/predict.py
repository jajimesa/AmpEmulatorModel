import torch

from scipy.io import wavfile
import numpy as np
import os

from model import AmpEmulatorModel

def predict():
    """
    Función principal para predecir con el modelo.
    """
    model = AmpEmulatorModel.load_from_checkpoint("models/model.ckpt")
    model.eval()    # Ponemos el modelo en modo de evaluación

    # Cargamos los datos de entrada
    

    # Hacemos la predicción
    with torch.no_grad():
        pass

    # Guardamos la predicción como un .wav
