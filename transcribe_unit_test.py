import os
import shutil
import unittest
from diarize import diarize_audio
from transcribe import transcribe_segments

class TestTranscription(unittest.TestCase):

    def setUp(self):
        self.test_audio_path = "test.wav"  # Ensure this file exists and contains 2 speakers
        self.test_output_dir = "test_segments"
        self.num_speakers = 2

        # Ensure test file exists
        if not os.path.exists(self.test_audio_path):
            raise FileNotFoundError("Test audio file 'test.wav' is missing.")

        # Create output directory if it doesn't exist
        os.makedirs(self.test_output_dir, exist_ok=True)

    def test_transcription_outputs(self):
        # Step 1: Diarize the audio
        segments = diarize_audio(self.test_audio_path, self.test_output_dir, self.num_speakers)
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)

        # Step 2: Transcribe the diarized segments
        results = transcribe_segments(self.test_output_dir)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        for entry in results:
            self.assertIn("speaker", entry)
            self.assertIn("transcript", entry)
            self.assertIsInstance(entry["speaker"], str)
            self.assertIsInstance(entry["transcript"], str)

    def tearDown(self):
        # Recursively delete output directory and contents
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

if __name__ == '__main__':
    unittest.main()