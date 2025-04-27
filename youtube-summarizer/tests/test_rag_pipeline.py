"""
Tests for the rag_pipeline module.
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from src.rag.rag_pipeline import RagPipeline

class TestRagPipeline:
    """Test cases for the RagPipeline class."""
    
    @pytest.fixture
    def mock_components(self):
        """Mock the components used by RagPipeline."""
        with patch('src.rag.rag_pipeline.AudioProcessor') as mock_audio_processor:
            with patch('src.rag.rag_pipeline.VectorStore') as mock_vector_store:
                with patch('src.rag.rag_pipeline.KnowledgeGraph') as mock_knowledge_graph:
                    # Configure mocks
                    instance_audio = MagicMock()
                    instance_vector = MagicMock()
                    instance_graph = MagicMock()
                    
                    mock_audio_processor.return_value = instance_audio
                    mock_vector_store.return_value = instance_vector
                    mock_knowledge_graph.return_value = instance_graph
                    
                    yield {
                        'audio_processor': mock_audio_processor,
                        'audio_instance': instance_audio,
                        'vector_store': mock_vector_store,
                        'vector_instance': instance_vector,
                        'knowledge_graph': mock_knowledge_graph,
                        'graph_instance': instance_graph
                    }
    
    @pytest.fixture
    def pipeline(self, mock_components, tmp_path):
        """Create a RagPipeline instance with mocked components."""
        data_dir = str(tmp_path / "rag_data")
        pipeline = RagPipeline(data_dir=data_dir)
        return pipeline
    
    def test_initialization(self, pipeline, mock_components):
        """Test that the RagPipeline initializes with the correct components."""
        assert pipeline.audio_processor == mock_components['audio_instance']
        assert pipeline.vector_store == mock_components['vector_instance']
        assert pipeline.knowledge_graph == mock_components['graph_instance']
    
    def test_process_new_video(self, pipeline, mock_components):
        """Test processing a new video through the RAG pipeline."""
        # Configure mocks
        audio_instance = mock_components['audio_instance']
        vector_instance = mock_components['vector_instance']
        graph_instance = mock_components['graph_instance']
        
        # Mock the audio processor to return fake chunks
        processed_chunks = [
            {
                "chunk_id": 0,
                "start_time": 0,
                "end_time": 5000,
                "embedding": np.array([0.1, 0.2, 0.3])
            },
            {
                "chunk_id": 1,
                "start_time": 5000,
                "end_time": 10000,
                "embedding": np.array([0.4, 0.5, 0.6])
            }
        ]
        audio_instance.process_audio_file.return_value = processed_chunks
        
        # Mock the vector store to return chunk IDs
        vector_instance.add_processed_chunks.return_value = ["chunk0", "chunk1"]
        
        # Mock the knowledge graph
        graph_instance.add_video_node.return_value = "video:test_video"
        graph_instance.extract_concepts_from_transcript.return_value = [
            {"name": "python"},
            {"name": "testing"}
        ]
        
        # Test processing
        result = pipeline.process_new_video(
            video_id="test_video",
            audio_path="test.wav",
            transcript="This is a test transcript",
            metadata={"title": "Test Video"}
        )
        
        # Check results
        assert result["status"] == "complete"
        assert result["chunks_count"] == 2
        assert result["stored_chunks_count"] == 2
        assert result["extracted_concepts_count"] == 2
        
        # Verify component calls
        audio_instance.process_audio_file.assert_called_once()
        vector_instance.add_processed_chunks.assert_called_once()
        graph_instance.add_video_node.assert_called_once()
        graph_instance.extract_concepts_from_transcript.assert_called_once()
        graph_instance.add_segment_node.assert_called()
        graph_instance.save_graph.assert_called_once()
    
    def test_retrieve_context_text_query(self, pipeline, mock_components):
        """Test retrieving context with a text query."""
        # Configure mocks
        vector_instance = mock_components['vector_instance']
        
        # Mock vector results
        vector_instance.query.return_value = {
            "ids": [["chunk1", "chunk2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"file_id": "test_video"}, {"file_id": "test_video"}]],
            "distances": [[0.1, 0.2]]
        }
        
        # Mock graph context
        graph_instance = mock_components['graph_instance']
        graph_instance.get_context_for_summary.return_value = {
            "video_id": "test_video",
            "related_concepts": [{"name": "python"}],
            "related_videos": [{"id": "video:related", "data": {"title": "Related"}}]
        }
        
        # Test retrieving with text query
        result = pipeline.retrieve_context(query_text="test query", video_id="test_video")
        
        # Check results
        assert result["status"] == "complete"
        assert result["query_type"] == "text"
        assert "vector_results" in result
        assert "graph_context" in result
        
        # Verify component calls
        vector_instance.query.assert_called_once()
        graph_instance.get_context_for_summary.assert_called_once()
    
    def test_retrieve_context_audio_query(self, pipeline, mock_components):
        """Test retrieving context with an audio query."""
        # Configure mocks
        audio_instance = mock_components['audio_instance']
        vector_instance = mock_components['vector_instance']
        
        # Mock audio processing
        audio_instance.process_audio_file.return_value = [
            {
                "embedding": np.array([0.1, 0.2, 0.3]),
                "start_time": 0,
                "end_time": 5000
            }
        ]
        
        # Mock vector results
        vector_instance.query.return_value = {
            "ids": [["chunk1", "chunk2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"file_id": "test_video"}, {"file_id": "test_video"}]],
            "distances": [[0.1, 0.2]]
        }
        
        # Test retrieving with audio query
        result = pipeline.retrieve_context(query_audio="query.wav")
        
        # Check results
        assert result["status"] == "complete"
        assert result["query_type"] == "audio"
        assert "vector_results" in result
        assert "graph_context" in result
        
        # Verify component calls
        audio_instance.process_audio_file.assert_called_once()
        vector_instance.query.assert_called_once()
    
    def test_enhance_summary(self, pipeline, mock_components):
        """Test enhancing a summary with RAG context."""
        # Configure mocks to simulate context retrieval
        with patch.object(pipeline, 'retrieve_context') as mock_retrieve:
            mock_retrieve.return_value = {
                "status": "complete",
                "vector_results": {
                    "ids": [["chunk1", "chunk2"]],
                    "documents": [["doc1", "doc2"]],
                    "metadatas": [[{"file_id": "test_video"}, {"file_id": "test_video"}]]
                },
                "graph_context": {
                    "related_concepts": [
                        {"name": "python"},
                        {"name": "testing"}
                    ],
                    "related_videos": [
                        {"id": "video:related", "data": {"title": "Related Video"}}
                    ]
                }
            }
            
            # Test enhancing summary
            original_summary = "This is a summary of the video."
            enhanced = pipeline.enhance_summary(
                original_summary=original_summary,
                video_id="test_video"
            )
            
            # Verify the enhancement
            assert enhanced != original_summary
            assert "Related topics" in enhanced
            assert "python" in enhanced
            assert "Related videos" in enhanced
            assert "Related Video" in enhanced
            
            # Verify context retrieval was called
            mock_retrieve.assert_called_once_with(video_id="test_video")