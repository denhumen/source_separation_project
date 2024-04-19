import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from nmf import *

audio_file = './data/sounds_mixedX.wav'
y, sr = librosa.load(audio_file)

# write('output.wav', sr, y[:300000])

D = librosa.stft(y)
print(D)

magnitude = np.abs(D)


magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
print(magnitude_db)

plt.figure(figsize=(10, 6))
librosa.display.specshow(magnitude_db, sr=sr, x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')
plt.show()
print(magnitude_db.shape)
rank = 10
max_iter = int(1e5)
W, H = multiplicative_update(magnitude_db, rank, max_iter)

