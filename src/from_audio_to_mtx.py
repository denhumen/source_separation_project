import librosa
import numpy as np
import matplotlib.pyplot as plt
# from nmf import *

audio_file = './data/sounds_mixedY.wav'
y, sr = librosa.load(audio_file)

D = librosa.stft(y)
magnitude = np.abs(D)

magnitude_db = librosa.amplitude_to_db(magnitude, ref = np.max)

plt.figure(figsize = (10, 3))
librosa.display.specshow(magnitude_db, sr = sr, x_axis = 'time', y_axis = 'linear')
# plt.colorbar(format = '%+2.0f dB')
plt.title('Spectrogram')
plt.show()
