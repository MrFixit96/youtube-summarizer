"""
Vector store implementation for RAG pipeline using FAISS.

This module provides:
1. Storage of audio embeddings in FAISS
2. Retrieval of relevant audio segments based on queries
3. Persistence for efficient reuse
"""

import os
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

class FAISSVectorStore:
    """A simple vector store implementation using FAISS."""
    
    def __init__(self, embedding_model: Embeddings, persist_directory: str = None):
        """Initialize the FAISS vector store.
        
        Args:
            embedding_model: The embedding model to use
            persist_directory: Directory to save the vector store
        """
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.index = None
        self.documents = []
        self.document_embeddings = []
        
        if persist_directory and os.path.exists(persist_directory):
            self._load()
        else:
            # Create a new index
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index."""
        # We'll create the index when we add the first document
        self.index = None
        self.documents = []
        self.document_embeddings = []
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add texts to the vector store.
        
        Args:
            texts: List of text chunks to add
            metadatas: Optional metadata for each text chunk
        """
        if not texts:
            return
        
        # Create embeddings for the texts
        embeddings = self.embedding_model.embed_documents(texts)
        
        # Convert to documents with metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Add to our document store
        for i, (text, metadata) in enumerate(zip(texts, metadatas)):
            self.documents.append(Document(page_content=text, metadata=metadata))
            self.document_embeddings.append(embeddings[i])
        
        # Create or update the FAISS index
        self._update_index()
        
        # Save if we have a persist directory
        if self.persist_directory:
            self._save()
    
    def _update_index(self):
        """Update the FAISS index with current embeddings."""
        if not self.document_embeddings:
            return
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(self.document_embeddings).astype('float32')
        
        # Get dimensionality of embeddings
        dimension = embeddings_array.shape[1]
        
        # Create a new index if needed
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        
        # Clear the index and re-add all embeddings
        if isinstance(self.index, faiss.IndexFlatL2):
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_array)
        else:
            # Reset and re-add for other index types
            self.index.reset()
            self.index.add(embeddings_array)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for documents similar to the query text.
        
        Args:
            query: The query text
            k: Number of documents to return
        
        Returns:
            List of documents most similar to the query
        """
        if not self.index or not self.documents:
            return []
        
        # Create embedding for the query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Convert to numpy array
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search the index
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_array, k)
        
        # Get the documents
        result_documents = [self.documents[i] for i in indices[0]]
        return result_documents
    
    def _save(self):
        """Save the vector store to disk."""
        if not self.persist_directory:
            return
        
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Save the FAISS index
        index_path = os.path.join(self.persist_directory, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        # Save the documents and embeddings
        docs_path = os.path.join(self.persist_directory, "documents.pkl")
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save the embeddings
        emb_path = os.path.join(self.persist_directory, "embeddings.pkl")
        with open(emb_path, 'wb') as f:
            pickle.dump(self.document_embeddings, f)
    
    def _load(self):
        """Load the vector store from disk."""
        index_path = os.path.join(self.persist_directory, "index.faiss")
        docs_path = os.path.join(self.persist_directory, "documents.pkl")
        emb_path = os.path.join(self.persist_directory, "embeddings.pkl")
        
        # Check if all files exist
        if not (os.path.exists(index_path) and os.path.exists(docs_path) and os.path.exists(emb_path)):
            self._create_new_index()
            return
        
        # Load the FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load the documents
        with open(docs_path, 'rb') as f:
            self.documents = pickle.load(f)
        
        # Load the embeddings
        with open(emb_path, 'rb') as f:
            self.document_embeddings = pickle.load(f)