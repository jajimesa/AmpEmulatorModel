from model import AmpEmulatorModel

import numpy as np
import sounddevice as sd


# Parámetros de audio
in_rate = 44100  # Frecuencia de muestreo
sample_time = 100e-3 
batch_size = 256
duration = 20  # Duración de la captura de audio en segundos

def open_stream():
    """
    Método para abrir un stream de audio con la entrada '2: Analogue 1 + 2 (Focusrite USB ASIO)'.
    """
    dispositivos = sd.query_devices()
    input_idx = None
    output_idx = None
    for i, dispositivo in enumerate(dispositivos):
        if 'Analogue 1 + 2' in dispositivo['name']:
            print(f"Dispositivo de entrada: {dispositivo['name']}")
            input_idx = i
        if 'Altavoces (Focusrite USB Audio)' in dispositivo['name']:
            print(f"Dispositivo de salida: {dispositivo['name']}")
            output_idx = i
        if input_idx is not None and output_idx is not None:
            break

    if input_idx is None:
        raise Exception("No se encontró el dispositivo de entrada 'Focusrite USB ASIO'.")
    if output_idx is None:
        raise Exception("No se encontró el dispositivo de salida 'Focusrite USB ASIO'.")

    stream = sd.InputStream(
                device=(input_idx, output_idx), 
                samplerate=in_rate, 
                dtype='float32',                
                callback=callback
            )
    return stream

def callback(indata, outdata, frames, time, status):
    if status:
        print('Error en la captura de audio:', status)
    
    outdata[:] = indata

"""    
def process_audio(model, in_data):
    
    sample_size = int(in_rate * sample_time)
    length = len(in_data) - len(in_data) % sample_size

    in_data = in_data[:length].reshape((-1, 1, sample_size)).astype(np.float32)

    in_data = (in_data - in_data.mean()) / in_data.std()

    prev_sample = np.concatenate((np.zeros_like(in_data[0:1]), in_data[:-1]), axis=0)
    pad_in_data = np.concatenate((prev_sample, in_data), axis=2)

    out_data = []
    batches = pad_in_data.shape[0] // batch_size
    for x in np.array_split(pad_in_data, batches):
        out_data.append(model(torch.from_numpy(x)).numpy())

    out_data = np.concatenate(out_data)
    out_data = out_data[:, :, -in_data.shape[2] :]

    return out_data
"""

def rt_predict():
    """
    Función principal para realizar predicciones en tiempo real.
    """
    stream = open_stream()
    with stream:
        sd.sleep(int(duration * 1000))

if __name__ == '__main__':
    model = AmpEmulatorModel.load_from_checkpoint('model/models/model.ckpt')
    model.eval()  # Ponemos el modelo en modo de evaluación
    rt_predict()



