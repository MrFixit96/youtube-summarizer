"""
RAG pipeline module for YouTube Summarizer.

This module integrates:
1. Audio processing and embedding generation
2. Vector database storage and retrieval using FAISS
3. Knowledge graph for context
4. Enhanced summary generation
"""

import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.rag.audio_processor import AudioProcessor
from src.rag.vector_store import FAISSVectorStore
from src.rag.knowledge_graph import KnowledgeGraph
from langchain_core.documents import Document

# Configure logger
logger = logging.getLogger('youtube_summarizer')

class RagPipeline:
    def __init__(self, data_dir=None, embedding_model="clap"):
        """
        Initialize the RAG pipeline.
        
        Args:
            data_dir (str): Directory for storing data
            embedding_model (str): Model to use for embeddings
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), '../../data/rag')
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info(f"Initializing RAG pipeline with data directory: {self.data_dir}")
        
        # Initialize components
        vector_db_path = os.path.join(self.data_dir, 'vector_db')
        graph_path = os.path.join(self.data_dir, 'knowledge_graph.graphml')
        
        self.audio_processor = AudioProcessor(embedding_model=embedding_model)
        
        # Create embedding model wrapper for FAISS
        from langchain_core.embeddings import Embeddings
        
        class EmbeddingModelWrapper(Embeddings):
            """Wrapper for the audio processor embedding model to work with FAISS"""
            
            def __init__(self, audio_processor):
                self.audio_processor = audio_processor
                
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Convert texts to embeddings using the audio processor"""
                # In a real implementation, this would convert text to audio or use a text embedding model
                # For now, we'll use a dummy implementation
                return [[0.0] * 512 for _ in texts]  # Dummy 512-dim embeddings
                
            def embed_query(self, text: str) -> List[float]:
                """Convert a query to embeddings"""
                # Same dummy implementation for now
                return [0.0] * 512  # Dummy 512-dim embedding
        
        self.embedding_model = EmbeddingModelWrapper(self.audio_processor)
        self.vector_store = FAISSVectorStore(
            embedding_model=self.embedding_model,
            persist_directory=vector_db_path
        )
        self.knowledge_graph = KnowledgeGraph(graph_path=graph_path)
        
    def process_new_video(self, video_id, audio_path, transcript=None, metadata=None):
        """
        Process a new video through the RAG pipeline.
        
        Args:
            video_id (str): YouTube video ID
            audio_path (str): Path to the audio file
            transcript (str): Video transcript
            metadata (dict): Video metadata
            
        Returns:
            dict: Processing results
        """
        logger.info(f"Processing video {video_id} through RAG pipeline")
        results = {
            "video_id": video_id,
            "status": "processing"
        }
        
        try:
            # 1. Process audio chunks
            logger.info("Processing audio into chunks with embeddings")
            processed_chunks = self.audio_processor.process_audio_file(
                audio_path, 
                chunk_method="silence",
                min_silence_len=500,
                silence_thresh=-40
            )
            results["chunks_count"] = len(processed_chunks)
            
            # 2. Store in vector database (adapted for FAISS)
            logger.info("Storing audio chunks in vector database")
            
            # Prepare data for FAISS
            texts = []
            metadatas = []
            
            for chunk in processed_chunks:
                # Extract text content (use transcript segment or a placeholder)
                text_content = chunk.get("text", f"Audio segment from {chunk.get('start_time', 0)} to {chunk.get('end_time', 0)}")
                
                # Create metadata
                metadata = {
                    "video_id": video_id,
                    "start_time": chunk.get("start_time", 0),
                    "end_time": chunk.get("end_time", 0),
                    "duration": chunk.get("duration", 0),
                    "embedding": chunk.get("embedding", [])
                }
                
                texts.append(text_content)
                metadatas.append(metadata)
            
            # Add to FAISS
            self.vector_store.add_texts(texts, metadatas)
            chunk_ids = [f"{video_id}_{i}" for i in range(len(processed_chunks))]
            results["stored_chunks_count"] = len(chunk_ids)
            
            # 3. Update knowledge graph
            logger.info("Updating knowledge graph")
            
            # Add video node
            video_metadata = metadata or {}
            video_node = self.knowledge_graph.add_video_node(video_id, video_metadata)
            
            # Extract concepts if transcript available
            if transcript:
                concepts = self.knowledge_graph.extract_concepts_from_transcript(
                    transcript, video_id
                )
                results["extracted_concepts_count"] = len(concepts)
            
            # Add segment nodes and connect to video
            for chunk_id, chunk in zip(chunk_ids, processed_chunks):
                segment_metadata = {
                    "start_time": chunk.get("start_time"),
                    "end_time": chunk.get("end_time"),
                    "duration": chunk.get("duration")
                }
                self.knowledge_graph.add_segment_node(chunk_id, video_id, segment_metadata)
            
            # Save the updated graph
            self.knowledge_graph.save_graph()
            
            results["status"] = "complete"
            logger.info(f"RAG processing complete for video {video_id}")
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            results["status"] = "error"
            results["error"] = str(e)
            
        return results
        
    def retrieve_context(self, query_text=None, query_audio=None, video_id=None, n_results=5):
        """
        Retrieve context for enhancing a summary.
        
        Args:
            query_text (str): Text query
            query_audio (str): Path to audio query
            video_id (str): Filter by video ID
            n_results (int): Number of results to return
            
        Returns:
            dict: Retrieved context
        """
        logger.info(f"Retrieving context for {'query' if query_text or query_audio else video_id}")
        
        results = {
            "status": "processing",
            "query_type": "text" if query_text else "audio" if query_audio else "video_id"
        }
        
        try:
            # FAISS-based retrieval
            if query_text:
                # Text query - directly query FAISS
                docs = self.vector_store.similarity_search(query_text, k=n_results)
                
                # Filter by video_id if specified
                if video_id:
                    docs = [doc for doc in docs if doc.metadata.get("video_id") == video_id]
                
                # Format results similar to original
                vector_results = {
                    "documents": [doc.page_content for doc in docs],
                    "metadatas": [doc.metadata for doc in docs]
                }
                
            elif query_audio:
                # Audio query - process the audio first
                logger.info(f"Processing query audio: {query_audio}")
                chunks = self.audio_processor.process_audio_file(
                    query_audio, chunk_method="fixed", chunk_size_ms=5000
                )
                
                if chunks:
                    # For now, convert audio to text query (simplified)
                    # In a full implementation, you would use the audio embedding directly
                    query_segment = f"Audio segment from query"
                    docs = self.vector_store.similarity_search(query_segment, k=n_results)
                    
                    # Filter by video_id if specified
                    if video_id:
                        docs = [doc for doc in docs if doc.metadata.get("video_id") == video_id]
                    
                    # Format results similar to original
                    vector_results = {
                        "documents": [doc.page_content for doc in docs],
                        "metadatas": [doc.metadata for doc in docs]
                    }
                else:
                    raise ValueError("Failed to process query audio")
                    
            elif video_id:
                # Get all segments for the video - simplified approach
                # Since FAISS doesn't have direct filtering, we'll retrieve all and filter
                docs = self.vector_store.similarity_search("", k=100)  # Get more results to filter
                docs = [doc for doc in docs if doc.metadata.get("video_id") == video_id]
                
                # Format results similar to original
                vector_results = {
                    "documents": [doc.page_content for doc in docs],
                    "metadatas": [doc.metadata for doc in docs]
                }
                
            else:
                raise ValueError("Must provide query_text, query_audio, or video_id")
                
            # 2. Get knowledge graph context
            if video_id:
                graph_context = self.knowledge_graph.get_context_for_summary(video_id)
            else:
                # Try to extract video_id from the vector results
                result_video_ids = set()
                for metadata in vector_results['metadatas']:
                    if 'video_id' in metadata:
                        result_video_ids.add(metadata['video_id'])
                
                # If we have results from a single video, get context for it
                if len(result_video_ids) == 1:
                    result_video_id = next(iter(result_video_ids))
                    graph_context = self.knowledge_graph.get_context_for_summary(result_video_id)
                else:
                    graph_context = {"related_videos": [], "related_concepts": []}
            
            # 3. Combine results
            results.update({
                "status": "complete",
                "vector_results": vector_results,
                "graph_context": graph_context
            })
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            results["status"] = "error"
            results["error"] = str(e)
            
        return results
        
    def enhance_summary(self, original_summary, video_id, transcript=None):
        """
        Enhance a summary with RAG context.
        
        Args:
            original_summary (str): Original generated summary
            video_id (str): YouTube video ID
            transcript (str): Video transcript
            
        Returns:
            str: Enhanced summary
        """
        logger.info(f"Enhancing summary for video {video_id}")
        
        try:
            # 1. Retrieve context
            context = self.retrieve_context(video_id=video_id)
            
            if context["status"] != "complete":
                logger.warning("Failed to retrieve context for summary enhancement")
                return original_summary
                
            # 2. Extract related concepts and videos
            related_concepts = context['graph_context'].get('related_concepts', [])
            related_videos = context['graph_context'].get('related_videos', [])
            
            # 3. Enhance the summary (simplified version)
            # In a full implementation, you would use a language model here
            
            enhanced_summary = original_summary
            
            # Add context about related concepts if available
            if related_concepts:
                concept_names = [c['name'] for c in related_concepts[:5]]
                concepts_text = ", ".join(concept_names)
                concept_section = f"\n\nRelated topics: {concepts_text}"
                enhanced_summary += concept_section
                
            # Add context about related videos if available
            if related_videos:
                video_section = "\n\nRelated videos:"
                for video in related_videos[:3]:
                    video_data = video.get('data', {})
                    title = video_data.get('title', 'Unknown video')
                    video_section += f"\n- {title}"
                enhanced_summary += video_section
                
            logger.info("Summary enhanced with RAG context")
            return enhanced_summary
            
        except Exception as e:
            logger.error(f"Error enhancing summary: {str(e)}")
            # Return original if enhancement fails
            return original_summary