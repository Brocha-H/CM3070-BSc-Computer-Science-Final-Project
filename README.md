# CM3070-BSc-Computer-Science-Final-Project
Code for CM3070 BSc Computer Science Final Project - Meeting Minute Maker

# Audio Diarization, Transcription, and Summarization

This repository contains a Python-based tool for **speaker diarization, transcription, and summarization** of audio files. The tool splits audio into speaker-specific segments, transcribes them, and generates concise summaries for each speaker.

---

## Features

1. **Speaker Diarization**  
   - Splits audio into segments by speaker using `simple-diarizer`.  
   - Merges consecutive segments of the same speaker for cleaner output.  
   - Saves each speaker segment as a separate `.wav` file.

2. **Transcription**  
   - Converts audio segments into text using `faster-whisper`.    

3. **Summarization**  
   - Summarizes transcribed text using a distilled BART model from `transformers`.  
   - Automatically adjusts summary length based on input length.  
   - Skips summarization for very short segments (<10 words).
