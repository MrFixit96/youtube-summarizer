"""
Tests for the video downloader module.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from src.summarizer.video_downloader import VideoDownloader

class TestVideoDownloader(unittest.TestCase):
    def setUp(self):
        self.downloader = VideoDownloader(output_dir="test_temp")
        
    def tearDown(self):
        # Clean up test directory if it exists
        if os.path.exists("test_temp"):
            import shutil
            shutil.rmtree("test_temp")
    
    @patch('pytube.YouTube')
    def test_download_video(self, mock_youtube):
        # Mock the YouTube object and its methods
        mock_stream = MagicMock()
        mock_stream.download.return_value = "test_temp/test_video.mp4"
        
        # Mock the streams.filter().order_by().desc().first() chain
        mock_youtube.return_value.streams.filter.return_value.order_by.return_value.desc.return_value.first.return_value = mock_stream
        
        # Call the method
        result = self.downloader.download_video("https://www.youtube.com/watch?v=test_video_id")
        
        # Assertions
        self.assertEqual(result, "test_temp/test_video.mp4")
        mock_youtube.assert_called_once_with("https://www.youtube.com/watch?v=test_video_id")
        mock_stream.download.assert_called_once_with(output_path="test_temp")

    @patch('pytube.YouTube')
    def test_download_video_with_error(self, mock_youtube):
        # Mock YouTube to raise an exception
        mock_youtube.side_effect = Exception("Test error")
        
        # Test that the exception is propagated
        with self.assertRaises(Exception) as context:
            self.downloader.download_video("https://www.youtube.com/watch?v=test_video_id")
            
        # Check the exception message
        self.assertTrue("Failed to download video: Test error" in str(context.exception))

if __name__ == '__main__':
    unittest.main()