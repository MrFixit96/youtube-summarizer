"""
Knowledge graph module for maintaining relationships between audio content.

This module provides:
1. Creation and management of a knowledge graph for video content
2. Entity and relationship extraction from transcripts
3. Context enhancement for summarization
"""

import os
import json
import logging
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Configure logger
logger = logging.getLogger('youtube_summarizer')

class KnowledgeGraph:
    def __init__(self, graph_path=None):
        """
        Initialize the knowledge graph.
        
        Args:
            graph_path (str): Path to save/load the graph
        """
        self.graph_path = graph_path
        self.graph = nx.DiGraph()
        
        # Load existing graph if available
        if graph_path and os.path.exists(graph_path):
            self.load_graph()
        else:
            logger.info("Creating new knowledge graph")
            
            # Set graph metadata
            self.graph.graph["created_at"] = datetime.now().isoformat()
            self.graph.graph["description"] = "Knowledge graph for YouTube video content"
            
    def add_video_node(self, video_id, metadata):
        """
        Add a video node to the graph.
        
        Args:
            video_id (str): YouTube video ID
            metadata (dict): Video metadata (title, description, etc.)
            
        Returns:
            str: Node ID
        """
        node_id = f"video:{video_id}"
        
        # Add the node with attributes
        self.graph.add_node(
            node_id,
            type="video",
            video_id=video_id,
            **metadata
        )
        
        logger.info(f"Added video node: {node_id}")
        return node_id
        
    def add_concept_node(self, concept, source_video_id=None):
        """
        Add a concept node to the graph.
        
        Args:
            concept (str): Concept name/description
            source_video_id (str): Source video ID (optional)
            
        Returns:
            str: Node ID
        """
        # Normalize concept name
        concept_norm = concept.lower().strip()
        node_id = f"concept:{concept_norm}"
        
        # Add node if it doesn't exist
        if not self.graph.has_node(node_id):
            self.graph.add_node(
                node_id,
                type="concept",
                name=concept,
                first_seen=datetime.now().isoformat()
            )
            logger.info(f"Added new concept node: {node_id}")
        
        # If source video provided, add relationship
        if source_video_id:
            video_node_id = f"video:{source_video_id}"
            if self.graph.has_node(video_node_id):
                self.add_relationship(video_node_id, node_id, "mentions")
                
        return node_id
        
    def add_segment_node(self, segment_id, video_id, metadata):
        """
        Add an audio segment node to the graph.
        
        Args:
            segment_id (str): Segment ID from vector store
            video_id (str): YouTube video ID
            metadata (dict): Segment metadata
            
        Returns:
            str: Node ID
        """
        node_id = f"segment:{segment_id}"
        
        # Add the node with attributes
        self.graph.add_node(
            node_id,
            type="segment",
            segment_id=segment_id,
            video_id=video_id,
            **metadata
        )
        
        # Connect to video node
        video_node_id = f"video:{video_id}"
        if self.graph.has_node(video_node_id):
            self.add_relationship(video_node_id, node_id, "contains")
            
        logger.info(f"Added segment node: {node_id}")
        return node_id
        
    def add_relationship(self, source_node_id, target_node_id, relationship_type, metadata=None):
        """
        Add a relationship between nodes.
        
        Args:
            source_node_id (str): Source node ID
            target_node_id (str): Target node ID
            relationship_type (str): Type of relationship
            metadata (dict): Additional relationship metadata
            
        Returns:
            tuple: (source_node_id, target_node_id)
        """
        if not self.graph.has_node(source_node_id) or not self.graph.has_node(target_node_id):
            logger.warning(f"Cannot add relationship: nodes {source_node_id} or {target_node_id} don't exist")
            return None
            
        # Add edge with attributes
        attrs = {"type": relationship_type}
        if metadata:
            attrs.update(metadata)
            
        self.graph.add_edge(source_node_id, target_node_id, **attrs)
        logger.info(f"Added relationship: {source_node_id} --[{relationship_type}]--> {target_node_id}")
        
        return (source_node_id, target_node_id)
        
    def get_related_concepts(self, video_id, max_distance=2):
        """
        Get concepts related to a video.
        
        Args:
            video_id (str): YouTube video ID
            max_distance (int): Maximum path distance to consider
            
        Returns:
            list: Related concept nodes
        """
        video_node_id = f"video:{video_id}"
        if not self.graph.has_node(video_node_id):
            logger.warning(f"Video node {video_node_id} not found")
            return []
            
        # Find all paths from the video node to concept nodes within max_distance
        concept_nodes = []
        
        # Get all nodes within max_distance
        for node in nx.single_source_shortest_path_length(self.graph, video_node_id, cutoff=max_distance):
            # Check if it's a concept node
            if node.startswith("concept:"):
                node_data = self.graph.nodes[node]
                concept_nodes.append({
                    "id": node,
                    "name": node_data.get("name", node.split(":", 1)[1]),
                    "data": node_data
                })
                
        logger.info(f"Found {len(concept_nodes)} concepts related to video {video_id}")
        return concept_nodes
        
    def get_related_videos(self, video_id, via_concepts=True, max_distance=3):
        """
        Get videos related to a given video.
        
        Args:
            video_id (str): YouTube video ID
            via_concepts (bool): Whether to find relationships via concepts
            max_distance (int): Maximum path distance to consider
            
        Returns:
            list: Related video nodes
        """
        video_node_id = f"video:{video_id}"
        if not self.graph.has_node(video_node_id):
            logger.warning(f"Video node {video_node_id} not found")
            return []
            
        related_videos = []
        
        if via_concepts:
            # Find related concepts first
            related_concepts = self.get_related_concepts(video_id, max_distance=1)
            
            # For each concept, find videos that mention it
            for concept in related_concepts:
                concept_id = concept["id"]
                
                # Find videos connected to this concept
                for neighbor in self.graph.predecessors(concept_id):
                    if neighbor.startswith("video:") and neighbor != video_node_id:
                        video_data = self.graph.nodes[neighbor]
                        
                        # Skip if already in results
                        if any(v["id"] == neighbor for v in related_videos):
                            continue
                            
                        related_videos.append({
                            "id": neighbor,
                            "video_id": video_data.get("video_id", neighbor.split(":", 1)[1]),
                            "data": video_data,
                            "relationship": {"via": "concept", "concept": concept["name"]}
                        })
        else:
            # Direct search for related videos (future implementation)
            pass
            
        logger.info(f"Found {len(related_videos)} videos related to {video_id}")
        return related_videos
        
    def extract_concepts_from_transcript(self, transcript, video_id):
        """
        Extract concepts from a transcript and add to graph.
        
        Args:
            transcript (str): Video transcript
            video_id (str): YouTube video ID
            
        Returns:
            list: Extracted concept nodes
        """
        # In a real implementation, you would use NLP techniques here
        # For example: named entity recognition, keyword extraction, etc.
        # For simplicity, we'll use a basic approach here
        
        logger.info(f"Extracting concepts from transcript for video {video_id}")
        
        # Simple keyword extraction - in a real system use a proper NLP approach
        # For now just extract some keywords based on frequency
        words = transcript.lower().split()
        word_freq = {}
        
        # Skip common words
        stopwords = set(['the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'on', 'with',
                        'that', 'this', 'it', 'as', 'from', 'at', 'be', 'are', 'by',
                        'was', 'were', 'have', 'has', 'had', 'not', 'or', 'but'])
        
        for word in words:
            # Skip short words and stopwords
            if len(word) < 4 or word in stopwords:
                continue
                
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Add concepts to graph
        added_concepts = []
        for keyword, freq in top_keywords:
            if freq > 3:  # Only add if frequency is above threshold
                concept_id = self.add_concept_node(keyword, video_id)
                added_concepts.append({
                    "id": concept_id,
                    "name": keyword,
                    "frequency": freq
                })
                
        logger.info(f"Added {len(added_concepts)} concepts from transcript")
        return added_concepts
        
    def get_context_for_summary(self, video_id):
        """
        Get context information for enhancing a video summary.
        
        Args:
            video_id (str): YouTube video ID
            
        Returns:
            dict: Context information
        """
        # Get related concepts
        related_concepts = self.get_related_concepts(video_id)
        
        # Get related videos
        related_videos = self.get_related_videos(video_id)
        
        # Build context object
        context = {
            "video_id": video_id,
            "related_concepts": related_concepts,
            "related_videos": related_videos,
            "graph_size": {
                "nodes": self.graph.number_of_nodes(),
                "edges": self.graph.number_of_edges()
            }
        }
        
        return context
        
    def visualize(self, output_path=None, max_nodes=50):
        """
        Visualize the knowledge graph.
        
        Args:
            output_path (str): Path to save the visualization
            max_nodes (int): Maximum number of nodes to include
            
        Returns:
            str: Path to the saved image
        """
        if self.graph.number_of_nodes() == 0:
            logger.warning("Cannot visualize empty graph")
            return None
            
        # Create a subgraph if the graph is too large
        if self.graph.number_of_nodes() > max_nodes:
            # Take a sample of nodes for visualization
            nodes = list(self.graph.nodes())[:max_nodes]
            graph_to_viz = self.graph.subgraph(nodes)
        else:
            graph_to_viz = self.graph
            
        # Set up colors for different node types
        node_colors = []
        for node in graph_to_viz.nodes():
            if node.startswith("video:"):
                node_colors.append("lightblue")
            elif node.startswith("concept:"):
                node_colors.append("lightgreen")
            elif node.startswith("segment:"):
                node_colors.append("lightsalmon")
            else:
                node_colors.append("gray")
                
        # Create plot
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(graph_to_viz, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(graph_to_viz, pos, node_color=node_colors, node_size=500, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(graph_to_viz, pos, edge_color="gray", alpha=0.5)
        
        # Draw labels
        labels = {}
        for node in graph_to_viz.nodes():
            # Shorten labels for readability
            if ":" in node:
                node_type, node_id = node.split(":", 1)
                if len(node_id) > 15:
                    node_id = node_id[:12] + "..."
                labels[node] = f"{node_type}:\n{node_id}"
            else:
                labels[node] = node
                
        nx.draw_networkx_labels(graph_to_viz, pos, labels=labels, font_size=8)
        
        # Set title and layout
        node_type_counts = {
            "videos": len([n for n in self.graph.nodes() if n.startswith("video:")]),
            "concepts": len([n for n in self.graph.nodes() if n.startswith("concept:")]),
            "segments": len([n for n in self.graph.nodes() if n.startswith("segment:")])
        }
        
        plt.title(f"Knowledge Graph (Total: {self.graph.number_of_nodes()} nodes, "
                f"{self.graph.number_of_edges()} edges)\n"
                f"Videos: {node_type_counts['videos']}, "
                f"Concepts: {node_type_counts['concepts']}, "
                f"Segments: {node_type_counts['segments']}")
        plt.axis("off")
        
        # Save or show
        if output_path:
            plt.savefig(output_path, bbox_inches="tight")
            logger.info(f"Graph visualization saved to {output_path}")
            plt.close()
            return output_path
        else:
            plt.show()
            plt.close()
            return None
            
    def save_graph(self, path=None):
        """
        Save the graph to a file.
        
        Args:
            path (str): Path to save the graph
            
        Returns:
            str: Path to the saved file
        """
        save_path = path or self.graph_path
        
        if not save_path:
            logger.warning("No path specified for saving graph")
            return None
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Update metadata
        self.graph.graph["updated_at"] = datetime.now().isoformat()
        self.graph.graph["nodes_count"] = self.graph.number_of_nodes()
        self.graph.graph["edges_count"] = self.graph.number_of_edges()
        
        # Save as GraphML
        nx.write_graphml(self.graph, save_path)
        logger.info(f"Graph saved to {save_path}")
        
        return save_path
        
    def load_graph(self, path=None):
        """
        Load the graph from a file.
        
        Args:
            path (str): Path to load the graph from
            
        Returns:
            bool: Success status
        """
        load_path = path or self.graph_path
        
        if not load_path or not os.path.exists(load_path):
            logger.warning(f"Graph file not found: {load_path}")
            return False
            
        try:
            self.graph = nx.read_graphml(load_path)
            logger.info(f"Loaded graph from {load_path} "
                      f"with {self.graph.number_of_nodes()} nodes and "
                      f"{self.graph.number_of_edges()} edges")
            return True
            
        except Exception as e:
            logger.error(f"Error loading graph: {str(e)}")
            return False