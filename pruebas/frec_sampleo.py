from scipy.io import wavfile

in_rate, in_data = wavfile.read("model/data/input.wav")
print(in_rate)
print(in_data)
