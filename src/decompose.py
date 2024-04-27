import librosa
import numpy as np
from scipy.io.wavfile import write
from sklearn.decomposition import NMF
import soundfile as sf

def decompose(audio: str):
    audio_file = audio
    y, sr = librosa.load(audio_file)
    D = librosa.stft(y)
    magnitude = np.abs(D)   
    n_components = 2
    model = NMF(n_components=n_components, init='nndsvd', random_state=0, max_iter=1000)
    W = model.fit_transform(magnitude)
    H = model.components_
    reconstructed_audio = np.dot(W, H)
    reconstructed_signal = librosa.istft(reconstructed_audio)
    separated_signals = []
    for i in range(n_components):
        separated_magnitude = np.outer(W[:, i], H[i])
        separated_signal = librosa.istft(separated_magnitude * np.exp(1j * np.angle(D)))
        separated_signals.append(separated_signal)

    for i, separated_signal in enumerate(separated_signals):
        sf.write(f'./output/separated{audio.split("/")[-1].split(".")[0]}__source_{i+1}.wav', separated_signal, sr)
    write(f'./output/{audio.split("/")[-1].split(".")[0]}_dec.wav', sr, reconstructed_signal)
    
for file in ["./data/micro1.mp3", "./data/micro2.mp3"]:
    decompose(file)
