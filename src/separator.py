"Separator module"

import os
import librosa
import numpy as np
import scipy.signal as ss
from scipy.io.wavfile import write
from mnmf import mnmf

iter_dir = "./database/two-channel"

def separate(input_dir, FFT = 4096, HOP = 2048, NUM_ITER = 20):
    "Separate dir"
    all_tasks = os.listdir(input_dir)
    
    for inx, task in enumerate(all_tasks):
        curr_path = input_dir + "/" + task
        files_mixed = os.listdir(curr_path + "/mixed/")
        if not os.path.exists(curr_path + f"/results_{inx}"):
            os.mkdir(curr_path + f"/results_{inx}")
        file_lst = []
        for file in files_mixed:
            audio, sr = librosa.load(curr_path + "/mixed/" + file)
            file_lst.append(audio)
        x = np.vstack(file_lst)
        _, T = x.shape
        _, _, X = ss.stft(x, nperseg = FFT, noverlap = HOP)
        Y = mnmf(X, n_basis = 2, iteration = NUM_ITER)
        _, y = ss.istft(Y, nperseg = FFT, noverlap = HOP)
        y = y[:, :T]
        for i, res in enumerate(y):
            write(curr_path + f"/results_{inx}/file_{i}.wav", rate = sr, data = res)


separate(iter_dir)
