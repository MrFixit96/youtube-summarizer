"""
Tests for the audio_processor module.
"""

import os
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.audio_processor import AudioProcessor

# Skip tests if the required models are not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Skipping tests that require GPU"
)

class TestAudioProcessor:
    """Test cases for the AudioProcessor class."""
    
    @pytest.fixture
    def sample_audio_path(self, tmp_path):
        """Create a sample audio file for testing."""
        # This is a fixture that would normally create a sample audio file
        # For testing purposes, we'll just return a path and mock the actual loading
        audio_path = tmp_path / "sample_audio.wav"
        return str(audio_path)
    
    @pytest.fixture
    def mock_audio_processor(self):
        """Create a mock AudioProcessor with mocked model components."""
        with patch('src.rag.audio_processor.Wav2Vec2Processor') as mock_processor:
            with patch('src.rag.audio_processor.Wav2Vec2Model') as mock_model:
                with patch('src.rag.audio_processor.open_clip') as mock_clip:
                    # Configure mocks
                    mock_processor.from_pretrained.return_value = MagicMock()
                    mock_model.from_pretrained.return_value = MagicMock()
                    mock_model.from_pretrained.return_value.to.return_value = MagicMock()
                    mock_clip.create_model_and_transforms.return_value = (MagicMock(), MagicMock(), MagicMock())
                    
                    # Create processor with the mocked components
                    processor = AudioProcessor(embedding_model="wav2vec2", device="cpu")
                    yield processor
    
    def test_initialization(self, mock_audio_processor):
        """Test that the AudioProcessor initializes with the correct model."""
        assert mock_audio_processor.embedding_model_type == "wav2vec2"
        assert mock_audio_processor.device == "cpu"
    
    @patch('src.rag.audio_processor.AudioSegment')
    def test_chunk_audio_fixed(self, mock_audio_segment, mock_audio_processor, sample_audio_path):
        """Test chunking audio with fixed intervals."""
        # Setup mock AudioSegment
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        mock_audio.__len__.return_value = 30000  # 30 seconds
        
        # Test fixed chunking
        chunks = mock_audio_processor.chunk_audio(
            sample_audio_path, 
            chunk_method="fixed", 
            chunk_size_ms=10000, 
            chunk_overlap_ms=2000
        )
        
        # We should get 4 chunks from a 30-second audio with 10s chunks and 2s overlap
        assert len(chunks) == 4
        assert chunks[0]["start_time"] == 0
        assert chunks[0]["end_time"] == 10000
        
    @patch('src.rag.audio_processor.AudioSegment')
    @patch('src.rag.audio_processor.split_on_silence')
    @patch('src.rag.audio_processor.detect_nonsilent')
    def test_chunk_audio_silence(self, mock_detect, mock_split, mock_audio_segment, 
                                mock_audio_processor, sample_audio_path):
        """Test chunking audio by silence detection."""
        # Setup mocks
        mock_audio = MagicMock()
        mock_audio_segment.from_file.return_value = mock_audio
        
        # Mock chunks and ranges
        mock_chunks = [MagicMock(), MagicMock()]
        mock_ranges = [(0, 5000), (7000, 15000)]
        
        mock_split.return_value = mock_chunks
        mock_detect.return_value = mock_ranges
        
        # Test silence detection chunking
        chunks = mock_audio_processor.chunk_audio(
            sample_audio_path, 
            chunk_method="silence"
        )
        
        # Should get 2 chunks based on our mocks
        assert len(chunks) == 2
        assert chunks[0]["start_time"] == 0
        assert chunks[0]["end_time"] == 5000
        assert chunks[1]["start_time"] == 7000
        assert chunks[1]["end_time"] == 15000
    
    @patch('src.rag.audio_processor.AudioProcessor._process_wav2vec2')
    def test_generate_embedding_wav2vec2(self, mock_process, mock_audio_processor):
        """Test generating embedding with Wav2Vec2."""
        # Setup mock
        mock_process.return_value = np.array([0.1, 0.2, 0.3])
        
        # Create mock chunk
        chunk_data = {
            "chunk": MagicMock(),
            "start_time": 0,
            "end_time": 5000,
            "chunk_id": 0
        }
        chunk_data["chunk"].get_array_of_samples.return_value = np.array([0.1, 0.2, 0.3])
        chunk_data["chunk"].channels = 1
        chunk_data["chunk"].frame_rate = 16000
        
        # Test embedding generation
        result = mock_audio_processor.generate_embedding(chunk_data)
        
        # Check results
        assert "embedding" in result
        assert result["embedding_model"] == "wav2vec2"
        assert result["embedding_dim"] == 3
    
    @patch('src.rag.audio_processor.AudioProcessor.chunk_audio')
    @patch('src.rag.audio_processor.AudioProcessor.generate_embedding')
    def test_process_audio_file(self, mock_generate, mock_chunk, mock_audio_processor, sample_audio_path):
        """Test processing a complete audio file."""
        # Setup mocks
        mock_chunks = [
            {"chunk": MagicMock(), "start_time": 0, "end_time": 5000, "chunk_id": 0},
            {"chunk": MagicMock(), "start_time": 5000, "end_time": 10000, "chunk_id": 1}
        ]
        mock_chunk.return_value = mock_chunks
        
        mock_generate.side_effect = lambda chunk: {
            **chunk,
            "embedding": np.array([0.1, 0.2, 0.3]),
            "embedding_model": "wav2vec2",
            "embedding_dim": 3
        }
        
        # Test processing
        results = mock_audio_processor.process_audio_file(sample_audio_path)
        
        # Check results
        assert len(results) == 2
        assert "embedding" in results[0]
        assert "file_id" in results[0]
        assert results[0]["embedding_model"] == "wav2vec2"
        assert results[1]["chunk_id"] == 1