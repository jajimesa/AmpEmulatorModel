import sounddevice as sd
import numpy as np
from scipy.io import wavfile

def grabar_audio(duracion_segundos, fs, canales):
    print("Selecciona el dispositivo de audio:")
    dispositivos = sd.query_devices()
    for i, dispositivo in enumerate(dispositivos):
        print(f"{i}: {dispositivo['name']}")
    
    dispositivo_idx = int(input("Ingresa el número correspondiente al dispositivo: "))
    
    print(f"Grabando {duracion_segundos} segundos de audio desde {dispositivos[dispositivo_idx]['name']}...")
    grabacion = sd.rec(int(duracion_segundos * fs), samplerate=fs, channels=canales, device=dispositivo_idx, dtype='float32')
    sd.wait()
    print("Grabación completada.")
    return grabacion

def guardar_audio(filename, audio, fs):
    wavfile.write(filename, fs, audio)

if __name__ == "__main__":
    duracion_segundos = 5
    fs = 48000  # Frecuencia de muestreo
    canales = 2  # Número de canales (estéreo)

    grabacion = grabar_audio(duracion_segundos, fs, canales)
    
    # Guardar la grabación como archivo .wav
    nombre_archivo = input("Ingrese el nombre del archivo .wav para guardar la grabación: ")
    guardar_audio(nombre_archivo, grabacion, fs)
    print(f"Grabación guardada como '{nombre_archivo}'.")
