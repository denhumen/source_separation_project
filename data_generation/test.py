
import numpy as np
import mixer
import librosa
import soundfile as sf

source1, sample_rate1 = librosa.load("./data/sound1.wav")
source2, sample_rate2 = librosa.load("./data/sound2.wav")

myMixer = mixer.Mixer(2, [[1.0, 1.0, 1.0], [9.0, 9.0, 1.0]], [[2.0, 2.0, 1.0], [8.0, 8.0, 1.0]])
myMixer.create_room(sample_rate1, dimensions=[10.0, 10.0, 10.0], rt = 0.3)
myMixer.load_sources_from_arrays([source1, source2 * 1.5])

results = myMixer.simulate_and_return_recordings()

sf.write("./sound1_mix.mp3", results[0], sample_rate1)
sf.write("./sound2_mix.mp3", results[1], sample_rate2)
