import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# Parámetros de audio
fs = 44100  # Frecuencia de muestreo
canales = 1  # Número de canales (mono)

def guardar_audio(filename, audio, fs):
    # Convertir los datos de audio a float32 antes de guardarlos
    audio = np.array(audio, dtype=np.float32)
    write(filename, fs, audio)

def callback(indata, frames, time, status):
    if status:
        print(status)
    grabacion.extend(indata.flatten())  # Accede a la lista definida en un ámbito superior!!

if __name__ == "__main__":
    duracion_segundos = 10
    grabacion = []

    dispositivos = sd.query_devices()
    dispositivo_idx = None
    for i, dispositivo in enumerate(dispositivos):
        if 'Focusrite USB A' in dispositivo['name']:
            dispositivo_idx = i
            break
    
    if dispositivo_idx is not None:
        print(f"Grabando {duracion_segundos} segundos de audio desde {dispositivos[dispositivo_idx]['name']}...")
        stream = sd.InputStream(callback=callback, samplerate=fs, channels=canales, device=dispositivo_idx, dtype='float32')
        with stream:
            sd.sleep(int(duracion_segundos * 1000))
        print("Grabación completada.")
        
        # Guardar la grabación como archivo .wav
        nombre_archivo = input("Ingrese el nombre del archivo .wav para guardar la grabación: ")
        guardar_audio(nombre_archivo, grabacion, fs)
        print(f"Grabación guardada como '{nombre_archivo}'.")
    else:
        print("No se encontró la entrada '2: Analogue 1 + 2 (Focusrite USB ASIO)'. Por favor, verifique la conexión y los controladores.")
