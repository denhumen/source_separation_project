"Drawer module"

import librosa
import matplotlib.pyplot as plt

def plot_components(reconstructed_sounds, sr):
    "Plots in wave form format"
    n = len(reconstructed_sounds)
    colors = ['r', 'g', 'b']
    _, ax = plt.subplots(nrows=n, ncols=1, sharex=True, figsize=(11, 3*n))
    for i in range(n):
        librosa.display.waveshow(reconstructed_sounds[i], sr=sr,
            color=colors[i], ax=ax[i], label=f'Source {i}', axis='time')
        ax[i].set(xlabel='Time [s]')
        ax[i].legend()
