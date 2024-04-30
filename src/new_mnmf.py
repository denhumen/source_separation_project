import numpy as np
import librosa
from sklearn.decomposition import NMF

def mnmf(X, n_sources, n_bases, max_iter=100):
    n_channels, n_freq, n_frames = X.shape
    V = np.abs(X).reshape(n_channels * n_freq, -1)
    
    nmf = NMF(n_components=n_sources * n_bases, max_iter=max_iter, random_state=42)
    W = nmf.fit_transform(V)
    H = nmf.components_
    W = W.reshape(n_channels, n_freq, n_sources, n_bases)
    H = H.reshape(n_sources, n_bases, -1)

    Z = np.zeros_like(X)
    for c in range(n_channels):
        for s in range(n_sources):
            WH = np.dot(W[c, :, s, :], H[s])
            Z[c] += WH[:, np.newaxis] * X[c] / np.maximum(1e-16, np.dot(WH, np.abs(X[c]).T))
    
    return Z

file1 = "./data/sounds_mixedX.wav"
file2 = "./data/sounds_mixedY.wav"

y1, sr1 = librosa.load(file1, sr=None)
y2, sr2 = librosa.load(file2, sr=None)

stft_y1 = librosa.stft(y1)
stft_y2 = librosa.stft(y2)

stft_signals = np.stack((stft_y1, stft_y2), axis=0)

n_sources = 2

n_bases = 2

Z = mnmf(stft_signals, n_sources, n_bases)

for i in range(n_sources):
    y_i = librosa.istft(Z[i % 2])
    librosa.output.write_wav(f'separated_source_{i+1}.wav', y_i, sr1)

print("Done!")