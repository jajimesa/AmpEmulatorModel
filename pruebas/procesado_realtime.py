import sounddevice as sd
import numpy as np

# Configuración de parámetros de audio
sample_rate = 44100  # Frecuencia de muestreo en Hz
duration = 20  # Duración de la grabación en segundos

def callback(indata, outdata, frames, time, status):
    if status:
        print('Error en la captura de audio:', status)
    
    outdata[:] = indata

def main():
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

    print("Capturando audio del micrófono...")
    with sd.Stream(device=(input_idx, output_idx),
                   latency='low',
                   channels=1,
                   samplerate=sample_rate,
                   callback=callback
                   ):
        sd.sleep(int(duration * 1000))  # Esperar durante la grabación

if __name__ == "__main__":
    main()