import os
import unittest
from diarize import diarize_audio
from transcribe import transcribe_segments
from summarize import summarize_segment_files

class TestSummarization(unittest.TestCase):

    def setUp(self):
        self.test_audio_path = "Untitled_design.wav"  # Ensure this file exists and has 2 speakers
        self.output_dir = "test_segments"
        self.num_speakers = 2

        if not os.path.exists(self.test_audio_path):
            raise FileNotFoundError("Test audio file 'test.wav' is missing.")

        # Clean up old segments if they exist
        if os.path.exists(self.output_dir):
            for f in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, f))
        else:
            os.makedirs(self.output_dir)

    def test_full_pipeline_summarization(self):
        # Step 1: Diarize
        segments = diarize_audio(self.test_audio_path, self.output_dir, self.num_speakers)

        # Step 2: Transcribe
        transcripts = transcribe_segments(self.output_dir)
        # Step 3: Summarize
        summaries = summarize_segment_files(transcripts)
        self.assertEqual(len(summaries), len(transcripts), "Mismatch between transcript and summary count.")
        for s in summaries:
            self.assertIn("speaker", s)
            self.assertIn("summary", s)
            self.assertTrue(len(s["summary"]) > 0, "Empty summary returned.")

if __name__ == "__main__":
    unittest.main()