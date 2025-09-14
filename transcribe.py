import re
from faster_whisper import WhisperModel

model = WhisperModel("tiny", device="cpu", compute_type="int8")

def transcribe_segments(diarized_segments):
    print("Starting transcription...")

    transcripts = []

    for seg in diarized_segments:
        filepath = seg["filename"]
        speaker = seg["label"]

        # Transcribe the audio segment
        segments, _ = model.transcribe(filepath)
        transcript = " ".join(s.text.strip() for s in segments if s.text.strip())

        transcripts.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "transcript": transcript
        })

    print("Transcription complete!")
    return transcripts