import unittest
import os
import shutil
from diarize import diarize_audio

class TestDiarizeAudio(unittest.TestCase):
    def setUp(self):
        self.test_audio_path = "test.wav"  # Ensure this file exists and contains 2 speakers
        self.test_output_dir = "test_segments"
        self.num_speakers = 2

        # Ensure test file exists
        if not os.path.exists(self.test_audio_path):
            raise FileNotFoundError("Test audio file 'test.wav' is missing.")

    def test_diarize_audio_outputs(self):
        # Run diarization
        segments = diarize_audio(self.test_audio_path, self.test_output_dir, num_speakers=self.num_speakers)

        # Check that the output directory was created
        self.assertTrue(os.path.exists(self.test_output_dir), "Output directory was not created.")

        # Check that at least one .wav segment was created
        output_files = [f for f in os.listdir(self.test_output_dir) if f.endswith(".wav")]
        self.assertGreater(len(output_files), 0, "No segmented audio files were created.")

        # Check that each segment has required fields
        self.assertIsInstance(segments, list)
        for seg in segments:
            self.assertIn("start", seg)
            self.assertIn("end", seg)
            self.assertIn("label", seg)
            self.assertIsInstance(seg["start"], float)
            self.assertIsInstance(seg["end"], float)

    def tearDown(self):
        # Clean up segmented audio
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

if __name__ == '__main__':
    unittest.main()