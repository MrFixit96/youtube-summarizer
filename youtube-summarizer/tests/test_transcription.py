"""
Tests for the transcription module.
"""

import unittest
from unittest.mock import patch, MagicMock
from src.summarizer.transcription import Transcription

class TestTranscription(unittest.TestCase):
    def setUp(self):
        self.transcriber = Transcription(model_name="base")
    
    @patch('whisperx.load_model')
    def test_transcribe_audio(self, mock_load_model):
        # Create mock objects for WhisperX
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {
            "segments": [
                {"text": "This is the first segment."},
                {"text": "This is the second segment."}
            ]
        }
        mock_load_model.return_value = mock_model
        
        # Test transcribing audio
        result = self.transcriber.transcribe_audio("test_audio.wav")
        
        # Assertions
        expected_transcript = "This is the first segment. This is the second segment."
        self.assertEqual(result, expected_transcript)
        
        # Verify the model was loaded and used correctly
        mock_load_model.assert_called_once()
        mock_model.transcribe.assert_called_once_with("test_audio.wav")
    
    @patch('whisperx.load_model')
    def test_transcribe_audio_with_error(self, mock_load_model):
        # Make the model raise an error
        mock_load_model.side_effect = Exception("Test error")
        
        # Test that the exception is propagated
        with self.assertRaises(Exception) as context:
            self.transcriber.transcribe_audio("test_audio.wav")
            
        # Check the exception message
        self.assertTrue("Failed to transcribe audio: Test error" in str(context.exception))
    
    @patch('whisperx.load_model')
    def test_transcribe_audio_empty_segments(self, mock_load_model):
        # Create mock with empty segments
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}
        mock_load_model.return_value = mock_model
        
        # Test transcribing audio with no segments
        result = self.transcriber.transcribe_audio("test_audio.wav")
        
        # Assertions
        self.assertEqual(result, "")

if __name__ == '__main__':
    unittest.main()