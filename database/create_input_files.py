import os
import numpy as np
import librosa
import soundfile as sf
from mixer import Mixer

root_path = "./"
clips_path = os.path.join(root_path, "clips/")
output_path = os.path.join(root_path, "two-channel/")

os.makedirs(output_path, exist_ok=True)

audio_files = [os.path.join(clips_path, f) for f in os.listdir(clips_path) if f.endswith('.mp3')]

def process_files():
    for i in range(0, len(audio_files), 2):
        if i + 1 < len(audio_files):
            file1 = audio_files[i]
            file2 = audio_files[i + 1]
            
            source1, sr1 = librosa.load(file1, sr=None)
            source2, sr2 = librosa.load(file2, sr=None)
            
            mixer = Mixer(2, [[1.0, 1.0, 1.0], [9.0, 9.0, 1.0]], [[2.0, 2.0, 1.0], [8.0, 8.0, 1.0]])
            mixer.create_room(sr1, dimensions=[10.0, 10.0, 10.0], rt=0.3)
            mixer.load_sources_from_arrays([source1, source2 * 1.5])
            
            results = mixer.simulate_and_return_recordings()
            folder_path = f"test_{i // 2}"
            real_folder_path = os.path.join(output_path, folder_path, "separated")
            mixed_folder_path = os.path.join(output_path, folder_path, "mixed")
            
            os.makedirs(real_folder_path, exist_ok=True)
            os.makedirs(mixed_folder_path, exist_ok=True)
            
            output_mixed_filename1 = f"mix_{i + 1}_1.mp3"
            output_mixed_filename2 = f"mix_{i + 1}_2.mp3"

            output_separated_filename1 = f"separated_{i + 1}_1.mp3"
            output_separated_filename2 = f"separated_{i + 1}_2.mp3"
            
            sf.write(os.path.join(real_folder_path, output_separated_filename1), source1, sr1, format='MP3')
            sf.write(os.path.join(real_folder_path, output_separated_filename2), source2, sr2, format='MP3')

            sf.write(os.path.join(mixed_folder_path, output_mixed_filename1), results[0], sr1, format='MP3')
            sf.write(os.path.join(mixed_folder_path, output_mixed_filename2), results[1], sr2, format='MP3')

process_files()
