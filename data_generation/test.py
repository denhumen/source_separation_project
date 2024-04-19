
import numpy as np
import mixer
import librosa
import soundfile as sf

source1, sample_rate1 = librosa.load("./data/one_cut.wav")
source2, sample_rate2 = librosa.load("./data/two_cut.wav")

myMixer = mixer.Mixer(2, [[4.5, 4.5, 1.0], [7.5, 7.5, 1.0]], [[6.0, 6.0, 1.0], [9.0, 9.0, 1.0]])
myMixer.create_room(sample_rate1, dimensions=[10.0, 10.0, 10.0], rt = 0.3)
myMixer.load_sources_from_arrays([source1, source2 * 1.5])

results = myMixer.simulate_and_return_recordings()

sf.write("./micro1.mp3", results[0], sample_rate1)
sf.write("./micro2.mp3", results[1], sample_rate2)
