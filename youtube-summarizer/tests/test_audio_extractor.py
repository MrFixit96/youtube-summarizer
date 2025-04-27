"""
Tests for the audio extractor module.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from src.summarizer.audio_extractor import AudioExtractor

class TestAudioExtractor(unittest.TestCase):
    def setUp(self):
        self.audio_extractor = AudioExtractor(output_dir="test_temp")
        
    def tearDown(self):
        # Clean up test directory if it exists
        if os.path.exists("test_temp"):
            import shutil
            shutil.rmtree("test_temp")
    
    @patch('ffmpeg.input')
    def test_extract_audio(self, mock_input):
        # Create a mock ffmpeg chain
        mock_output = MagicMock()
        mock_input.return_value.output.return_value = mock_output
        
        # Test extracting audio
        video_path = "test_video.mp4"
        result = self.audio_extractor.extract_audio(video_path)
        
        # Assertions
        expected_audio_path = os.path.join("test_temp", "test_video.wav")
        self.assertEqual(result, expected_audio_path)
        
        # Verify ffmpeg was called correctly
        mock_input.assert_called_once_with(video_path)
        mock_input.return_value.output.assert_called_once()
        mock_output.run.assert_called_once_with(quiet=True, overwrite_output=True)
    
    @patch('ffmpeg.input')
    def test_extract_audio_with_error(self, mock_input):
        # Make ffmpeg raise an error
        mock_output = MagicMock()
        mock_output.run.side_effect = Exception("Test error")
        mock_input.return_value.output.return_value = mock_output
        
        # Test that the exception is propagated
        with self.assertRaises(Exception) as context:
            self.audio_extractor.extract_audio("test_video.mp4")
            
        # Check the exception message
        self.assertTrue("Failed to extract audio: Test error" in str(context.exception))

if __name__ == '__main__':
    unittest.main()