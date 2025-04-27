"""
Tests for the vector_store module.
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock

from src.rag.vector_store import VectorStore

class TestVectorStore:
    """Test cases for the VectorStore class."""
    
    @pytest.fixture
    def mock_chromadb(self):
        """Mock ChromaDB dependency."""
        with patch('src.rag.vector_store.chromadb') as mock_chroma:
            # Configure mock
            mock_collection = MagicMock()
            mock_collection.count.return_value = 10
            
            mock_client = MagicMock()
            mock_client.get_collection.return_value = mock_collection
            mock_client.create_collection.return_value = mock_collection
            
            mock_chroma.PersistentClient.return_value = mock_client
            mock_chroma.Client.return_value = mock_client
            
            yield mock_chroma
    
    @pytest.fixture
    def vector_store(self, mock_chromadb, tmp_path):
        """Create a VectorStore instance with mocked dependencies."""
        persist_dir = str(tmp_path / "vector_db")
        return VectorStore(persist_directory=persist_dir)
    
    def test_initialization_with_persistence(self, mock_chromadb, tmp_path):
        """Test VectorStore initialization with persistence directory."""
        persist_dir = str(tmp_path / "vector_db")
        vector_store = VectorStore(persist_directory=persist_dir)
        
        # Verify PersistentClient was used
        mock_chromadb.PersistentClient.assert_called_once_with(persist_dir)
        
        # Verify get_collection was attempted
        client = mock_chromadb.PersistentClient.return_value
        client.get_collection.assert_called_once()
    
    def test_initialization_in_memory(self, mock_chromadb):
        """Test VectorStore initialization without persistence (in-memory)."""
        vector_store = VectorStore(persist_directory=None)
        
        # Verify in-memory Client was used
        mock_chromadb.Client.assert_called_once()
        
        # Verify get_collection was attempted
        client = mock_chromadb.Client.return_value
        client.get_collection.assert_called_once()
    
    def test_prepare_metadata(self, vector_store):
        """Test metadata preparation for ChromaDB."""
        # Create a test chunk
        test_chunk = {
            "chunk_id": 1,
            "start_time": 1000,
            "end_time": 5000,
            "duration": 4000,
            "file_id": "test_video",
            "embedding": np.array([0.1, 0.2, 0.3]),
            "chunk": MagicMock()
        }
        
        # Prepare metadata
        metadata = vector_store._prepare_metadata(test_chunk)
        
        # Check metadata
        assert "embedding" not in metadata
        assert "chunk" not in metadata
        assert metadata["chunk_id"] == "1"  # Converted to string
        assert metadata["start_time"] == "1000"
        assert metadata["duration"] == "4000"
        assert metadata["file_id"] == "test_video"
        assert "stored_at" in metadata
    
    def test_add_processed_chunks(self, vector_store, mock_chromadb):
        """Test adding processed chunks to the vector store."""
        # Configure mock
        collection = mock_chromadb.PersistentClient.return_value.get_collection.return_value
        
        # Create test chunks
        test_chunks = [
            {
                "chunk_id": i,
                "start_time": i * 5000,
                "end_time": (i + 1) * 5000,
                "duration": 5000,
                "file_id": "test_video",
                "embedding": np.array([0.1, 0.2, 0.3]),
                "chunk": MagicMock()
            }
            for i in range(3)
        ]
        
        # Add chunks
        ids = vector_store.add_processed_chunks(test_chunks)
        
        # Check results
        assert len(ids) == 3
        assert collection.add.called
        
        # The first call args should contain ids, embeddings, metadatas, and documents
        call_args = collection.add.call_args[1]
        assert len(call_args["ids"]) == 3
        assert len(call_args["embeddings"]) == 3
        assert len(call_args["metadatas"]) == 3
        assert len(call_args["documents"]) == 3
    
    def test_add_processed_chunks_with_transcript(self, vector_store, mock_chromadb):
        """Test adding processed chunks with transcript segmentation."""
        # Configure mock
        collection = mock_chromadb.PersistentClient.return_value.get_collection.return_value
        
        # Create test chunks
        test_chunks = [
            {
                "chunk_id": i,
                "start_time": i * 5000,
                "end_time": (i + 1) * 5000,
                "duration": 5000,
                "file_id": "test_video",
                "embedding": np.array([0.1, 0.2, 0.3]),
                "chunk": MagicMock()
            }
            for i in range(3)
        ]
        
        # Create a test transcript
        transcript = "This is a test transcript. It has multiple sentences. We want to segment it properly."
        
        # Add chunks with transcript
        ids = vector_store.add_processed_chunks(test_chunks, transcript=transcript)
        
        # Check results
        assert len(ids) == 3
        assert collection.add.called
        
        # Documents should contain transcript segments
        call_args = collection.add.call_args[1]
        documents = call_args["documents"]
        assert len(documents) == 3
        assert "This is a test" in documents[0] or "This is a test" in documents[1] or "This is a test" in documents[2]
    
    def test_query_with_embedding(self, vector_store, mock_chromadb):
        """Test querying with embedding vector."""
        # Configure mock
        collection = mock_chromadb.PersistentClient.return_value.get_collection.return_value
        collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"key": "value"}, {"key": "value2"}]],
            "distances": [[0.1, 0.2]]
        }
        
        # Create test embedding
        query_embedding = np.array([0.1, 0.2, 0.3])
        
        # Query
        results = vector_store.query(query_embedding=query_embedding, n_results=2)
        
        # Check results
        assert results["ids"][0] == ["id1", "id2"]
        assert results["documents"][0] == ["doc1", "doc2"]
        assert len(results["metadatas"][0]) == 2
        
        # Verify the query args
        call_args = collection.query.call_args[1]
        assert call_args["query_embeddings"] == [query_embedding.tolist()]
        assert call_args["n_results"] == 2
    
    def test_query_with_text(self, vector_store, mock_chromadb):
        """Test querying with text."""
        # Configure mock
        collection = mock_chromadb.PersistentClient.return_value.get_collection.return_value
        collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"key": "value"}, {"key": "value2"}]],
            "distances": [[0.1, 0.2]]
        }
        
        # Query
        results = vector_store.query(query_text="test query", n_results=2)
        
        # Check results
        assert results["ids"][0] == ["id1", "id2"]
        
        # Verify the query args
        call_args = collection.query.call_args[1]
        assert call_args["query_texts"] == ["test query"]
        assert call_args["n_results"] == 2
    
    def test_get_by_source(self, vector_store, mock_chromadb):
        """Test getting all segments for a source."""
        # Configure mock
        collection = mock_chromadb.PersistentClient.return_value.get_collection.return_value
        collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"file_id": "test_video"}, {"file_id": "test_video"}]]
        }
        
        # Get by source
        results = vector_store.get_by_source("test_video")
        
        # Check results
        assert len(results["ids"][0]) == 2
        
        # Verify the query args
        call_args = collection.query.call_args[1]
        assert call_args["where"] == {"file_id": "test_video"}
        assert call_args["n_results"] == 10000  # Large number to get all
    
    def test_delete_by_source(self, vector_store, mock_chromadb):
        """Test deleting segments by source."""
        # Configure mock
        collection = mock_chromadb.PersistentClient.return_value.get_collection.return_value
        collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "metadatas": [[{"file_id": "test_video"}, {"file_id": "test_video"}]]
        }
        
        # Delete by source
        count = vector_store.delete_by_source("test_video")
        
        # Check results
        assert count == 2
        
        # Verify the query and delete calls
        query_args = collection.query.call_args[1]
        assert query_args["where"] == {"file_id": "test_video"}
        
        delete_args = collection.delete.call_args[1]
        assert delete_args["ids"] == ["id1", "id2"]