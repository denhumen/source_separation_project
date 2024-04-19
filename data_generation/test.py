
import numpy as np
import mixer
import librosa
import soundfile as sf

source1, sample_rate1 = librosa.load("./input_data/mao.mp3")
source2, sample_rate2 = librosa.load("./input_data/erika.mp3")

myMixer = mixer.Mixer(2, [[4.5, 4.5, 1.0], [6.5, 6.5, 1.0]], [[5.0, 5.0, 5.0], [9.0, 9.0, 6.0]])
myMixer.create_room(sample_rate1, dimensions=[10.0, 10.0, 10.0], rt = 0.3)
myMixer.load_sources_from_arrays([source1, source2 * 1.5])

results = myMixer.simulate_and_return_recordings()

sf.write("./output_data/micro1.mp3", results[0], sample_rate1)
sf.write("./output_data/micro2.mp3", results[1], sample_rate2)
