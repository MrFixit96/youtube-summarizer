"""
Tests for the knowledge_graph module.
"""

import os
import pytest
import networkx as nx
import tempfile
from unittest.mock import MagicMock, patch
from datetime import datetime

from src.rag.knowledge_graph import KnowledgeGraph

class TestKnowledgeGraph:
    """Test cases for the KnowledgeGraph class."""
    
    @pytest.fixture
    def empty_graph(self):
        """Create an empty knowledge graph."""
        return KnowledgeGraph()
    
    @pytest.fixture
    def temp_graph_path(self):
        """Create a temporary file path for graph storage."""
        fd, path = tempfile.mkstemp(suffix='.graphml')
        os.close(fd)
        yield path
        # Clean up
        if os.path.exists(path):
            os.remove(path)
    
    @pytest.fixture
    def populated_graph(self):
        """Create a knowledge graph with some test data."""
        graph = KnowledgeGraph()
        
        # Add a video node
        graph.add_video_node("test_video", {
            "title": "Test Video",
            "author": "Test Author",
            "duration": 300
        })
        
        # Add concept nodes
        graph.add_concept_node("python", "test_video")
        graph.add_concept_node("testing", "test_video")
        
        # Add segment nodes
        graph.add_segment_node("segment1", "test_video", {
            "start_time": 0,
            "end_time": 10000
        })
        graph.add_segment_node("segment2", "test_video", {
            "start_time": 10000,
            "end_time": 20000
        })
        
        return graph
    
    def test_initialization(self, empty_graph):
        """Test that the KnowledgeGraph initializes correctly."""
        assert isinstance(empty_graph.graph, nx.DiGraph)
        assert empty_graph.graph.number_of_nodes() == 0
        assert empty_graph.graph_path is None
    
    def test_initialization_with_path(self, temp_graph_path):
        """Test initialization with a graph path."""
        # First create and save a graph
        graph1 = KnowledgeGraph(graph_path=temp_graph_path)
        graph1.add_video_node("test_video", {"title": "Test"})
        graph1.save_graph()
        
        # Now load it again
        graph2 = KnowledgeGraph(graph_path=temp_graph_path)
        
        # Verify it loaded
        assert graph2.graph.number_of_nodes() == 1
        assert "video:test_video" in graph2.graph.nodes
    
    def test_add_video_node(self, empty_graph):
        """Test adding a video node."""
        node_id = empty_graph.add_video_node("test_video", {
            "title": "Test Video",
            "author": "Test Author"
        })
        
        assert node_id == "video:test_video"
        assert empty_graph.graph.number_of_nodes() == 1
        assert empty_graph.graph.nodes[node_id]["type"] == "video"
        assert empty_graph.graph.nodes[node_id]["title"] == "Test Video"
    
    def test_add_concept_node(self, empty_graph):
        """Test adding a concept node."""
        # First add a video (for relationship)
        empty_graph.add_video_node("test_video", {"title": "Test"})
        
        # Add concept
        node_id = empty_graph.add_concept_node("python", "test_video")
        
        assert node_id == "concept:python"
        assert empty_graph.graph.number_of_nodes() == 2
        assert empty_graph.graph.nodes[node_id]["type"] == "concept"
        
        # Check relationship
        assert empty_graph.graph.has_edge("video:test_video", "concept:python")
        assert empty_graph.graph.edges["video:test_video", "concept:python"]["type"] == "mentions"
    
    def test_add_segment_node(self, empty_graph):
        """Test adding a segment node."""
        # First add a video (for relationship)
        empty_graph.add_video_node("test_video", {"title": "Test"})
        
        # Add segment
        node_id = empty_graph.add_segment_node("segment1", "test_video", {
            "start_time": 0,
            "end_time": 10000
        })
        
        assert node_id == "segment:segment1"
        assert empty_graph.graph.number_of_nodes() == 2
        assert empty_graph.graph.nodes[node_id]["type"] == "segment"
        assert empty_graph.graph.nodes[node_id]["start_time"] == 0
        
        # Check relationship
        assert empty_graph.graph.has_edge("video:test_video", "segment:segment1")
        assert empty_graph.graph.edges["video:test_video", "segment:segment1"]["type"] == "contains"
    
    def test_add_relationship(self, populated_graph):
        """Test adding a custom relationship."""
        result = populated_graph.add_relationship(
            "concept:python", 
            "concept:testing", 
            "related_to",
            {"strength": "high"}
        )
        
        assert result == ("concept:python", "concept:testing")
        assert populated_graph.graph.has_edge("concept:python", "concept:testing")
        assert populated_graph.graph.edges["concept:python", "concept:testing"]["type"] == "related_to"
        assert populated_graph.graph.edges["concept:python", "concept:testing"]["strength"] == "high"
    
    def test_get_related_concepts(self, populated_graph):
        """Test getting concepts related to a video."""
        concepts = populated_graph.get_related_concepts("test_video")
        
        assert len(concepts) == 2
        concept_names = [c["name"] for c in concepts]
        assert "python" in concept_names
        assert "testing" in concept_names
    
    def test_get_related_videos(self, populated_graph):
        """Test getting videos related to a given video."""
        # First add another video that shares a concept
        populated_graph.add_video_node("related_video", {"title": "Related"})
        populated_graph.add_concept_node("python", "related_video")
        
        # Get related videos
        videos = populated_graph.get_related_videos("test_video")
        
        assert len(videos) == 1
        assert videos[0]["id"] == "video:related_video"
        assert videos[0]["relationship"]["concept"] == "python"
    
    def test_extract_concepts_from_transcript(self, empty_graph):
        """Test extracting concepts from a transcript."""
        # Add a video for concept attachment
        empty_graph.add_video_node("test_video", {"title": "Test"})
        
        # Create a test transcript with repeated words
        transcript = """
        This is a test transcript about Python programming.
        Python is a popular programming language.
        This transcript talks about testing Python code.
        Testing is important for Python programming.
        Python Python Python testing testing programming programming.
        """
        
        # Extract concepts
        concepts = empty_graph.extract_concepts_from_transcript(transcript, "test_video")
        
        # Check that high-frequency words were captured
        assert len(concepts) > 0
        concept_names = [c["name"] for c in concepts]
        assert "python" in concept_names
        
        # Check graph structure
        assert empty_graph.graph.has_node("concept:python")
        assert empty_graph.graph.has_edge("video:test_video", "concept:python")
    
    def test_get_context_for_summary(self, populated_graph):
        """Test getting context for enhancing a summary."""
        # Add another video that shares a concept for related videos
        populated_graph.add_video_node("related_video", {"title": "Related"})
        populated_graph.add_concept_node("python", "related_video")
        
        # Get context
        context = populated_graph.get_context_for_summary("test_video")
        
        assert context["video_id"] == "test_video"
        assert len(context["related_concepts"]) == 2
        assert len(context["related_videos"]) == 1
        assert "graph_size" in context
    
    @patch('src.rag.knowledge_graph.plt')
    def test_visualize(self, mock_plt, populated_graph, temp_graph_path):
        """Test knowledge graph visualization."""
        # Visualize with output path
        output_path = temp_graph_path + ".png"
        result = populated_graph.visualize(output_path=output_path)
        
        assert result == output_path
        assert mock_plt.savefig.called
        mock_plt.close.assert_called()
    
    def test_save_and_load_graph(self, populated_graph, temp_graph_path):
        """Test saving and loading the graph."""
        # Set the graph path
        populated_graph.graph_path = temp_graph_path
        
        # Save the graph
        save_path = populated_graph.save_graph()
        assert save_path == temp_graph_path
        assert os.path.exists(temp_graph_path)
        
        # Create a new graph and load
        new_graph = KnowledgeGraph()
        success = new_graph.load_graph(temp_graph_path)
        
        assert success is True
        assert new_graph.graph.number_of_nodes() == populated_graph.graph.number_of_nodes()
        assert new_graph.graph.number_of_edges() == populated_graph.graph.number_of_edges()
        assert "video:test_video" in new_graph.graph.nodes