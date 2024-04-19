import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# Parámetros de audio
fs = 48000  # Frecuencia de muestreo
canales = 1  # Número de canales (mono)

def guardar_audio(filename, audio, fs):
    # Convertir los datos de audio a float32 antes de guardarlos
    audio = np.array(audio, dtype=np.float32)
    write(filename, fs, audio)

def callback(indata, frames, time, status):
    if status:
        print(status)
    grabacion.extend(indata.flatten())

if __name__ == "__main__":
    duracion_segundos = 10
    grabacion = []

    print("Selecciona el dispositivo de audio:")
    dispositivos = sd.query_devices()
    for i, dispositivo in enumerate(dispositivos):
        print(f"{i}: {dispositivo['name']}")
    
    dispositivo_idx = int(input("Ingresa el número correspondiente al dispositivo: "))
    
    stream = sd.InputStream(callback=callback, samplerate=fs, channels=canales, device=dispositivo_idx, dtype='float32')  # Especificamos dtype='float32' aquí
    
    print(f"Grabando {duracion_segundos} segundos de audio desde {dispositivos[dispositivo_idx]['name']}...")
    with stream:
        sd.sleep(int(duracion_segundos * 1000))
    
    print("Grabación completada.")
    
    # Guardar la grabación como archivo .wav
    nombre_archivo = input("Ingrese el nombre del archivo .wav para guardar la grabación: ")
    guardar_audio(nombre_archivo, grabacion, fs)
    print(f"Grabación guardada como '{nombre_archivo}'.")
