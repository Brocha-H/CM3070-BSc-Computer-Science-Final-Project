#Diarize.py

import os
import torchaudio
from simple_diarizer.diarizer import Diarizer

def diarize_audio(audio_path, segment_folder, num_speakers=2):

    print("Starting diarization...")
    # Corrected variable name
    os.makedirs(segment_folder, exist_ok=True)

    # Load the audio
    waveform, sample_rate = torchaudio.load(audio_path)
    diarizer = Diarizer(
        embed_model="xvec",
        cluster_method="sc",
    )
    # Run diarization
    raw_segments = diarizer.diarize(audio_path, num_speakers=num_speakers)

    # Merge short consecutive segments of the same speaker
    merged_segments = []
    for seg in raw_segments:
        if not merged_segments:
            merged_segments.append(seg)
        else:
            last = merged_segments[-1]
            if last['label'] == seg['label'] and seg['start'] - last['end'] < 1.0:
                last['end'] = seg['end']
            else:
                merged_segments.append(seg)


    merged_segments.sort(key=lambda seg: seg['start'])

    # Save merged segments to individual .wav files
    for i, seg in enumerate(merged_segments):
        start_sample = int(seg['start'] * sample_rate)
        end_sample = int(seg['end'] * sample_rate)
        segment_waveform = waveform[:, start_sample:end_sample]

        filename = os.path.join(segment_folder, f"segment_{i}_speaker_{seg['label']}.wav")
        torchaudio.save(filename, segment_waveform, sample_rate)
    
        seg["filename"] = filename

    label_mapping = {}
    next_speaker_id = 1

    for seg in merged_segments:
        label = seg["label"]
        if label not in label_mapping:
            label_mapping[label] = f"Speaker {next_speaker_id}"
            next_speaker_id += 1
        seg["speaker"] = label_mapping[label]


    print("Diarization complete!")

    return merged_segments