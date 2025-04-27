"""
Audio processor module for chunking audio and generating embeddings.

This module handles:
1. Chunking audio files into segments
2. Generating embeddings using CLAP or Wav2Vec models
3. Preparing data for storage in vector database
"""

import os
import torch
import numpy as np
import logging
from pathlib import Path
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import open_clip
import librosa

# Configure logger
logger = logging.getLogger('youtube_summarizer')

class AudioProcessor:
    def __init__(self, embedding_model="wav2vec2", device=None):
        """
        Initialize the AudioProcessor.
        
        Args:
            embedding_model (str): Model to use for embeddings ("wav2vec2" or "clap")
            device (str): Device to use for processing ("cuda" or "cpu")
        """
        self.embedding_model_type = embedding_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing AudioProcessor with model '{embedding_model}' on {self.device}")
        
        # Load embedding model based on selection
        if embedding_model == "wav2vec2":
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
        elif embedding_model == "clap":
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                "hf-hub:laion/clap-htsat-fused", 
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported embedding model: {embedding_model}")
            
    def chunk_audio(self, audio_path, chunk_method="silence", 
                   min_silence_len=500, silence_thresh=-40, 
                   keep_silence=100, chunk_size_ms=10000, 
                   chunk_overlap_ms=1000):
        """
        Split audio into chunks using silence detection or fixed intervals.
        
        Args:
            audio_path (str): Path to the audio file
            chunk_method (str): Method for chunking ("silence" or "fixed")
            min_silence_len (int): Minimum length of silence (ms) for silence detection
            silence_thresh (int): Silence threshold in dB for silence detection
            keep_silence (int): Amount of silence to keep at chunk boundaries (ms)
            chunk_size_ms (int): Size of each chunk in milliseconds for fixed chunking
            chunk_overlap_ms (int): Overlap between chunks in milliseconds for fixed chunking
            
        Returns:
            list: List of audio segments (AudioSegment objects) and their timestamps
        """
        logger.info(f"Loading audio file: {audio_path}")
        audio = AudioSegment.from_file(audio_path)
        
        chunks_with_timestamps = []
        
        if chunk_method == "silence":
            logger.info(f"Chunking audio by silence detection (min_silence={min_silence_len}ms, threshold={silence_thresh}dB)")
            chunks = split_on_silence(
                audio, 
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence
            )
            
            # Calculate timestamps for each chunk
            nonsilent_ranges = detect_nonsilent(
                audio, 
                min_silence_len=min_silence_len, 
                silence_thresh=silence_thresh
            )
            
            # If we have chunks but no detected ranges, fall back to fixed chunking
            if chunks and not nonsilent_ranges:
                logger.warning("Silence detection didn't find proper ranges. Falling back to fixed chunking.")
                return self.chunk_audio(audio_path, "fixed", chunk_size_ms=chunk_size_ms)
                
            for i, (chunk, (start_ms, end_ms)) in enumerate(zip(chunks, nonsilent_ranges)):
                chunks_with_timestamps.append({
                    "chunk": chunk,
                    "start_time": start_ms,
                    "end_time": end_ms,
                    "duration": end_ms - start_ms,
                    "chunk_id": i
                })
                
        elif chunk_method == "fixed":
            logger.info(f"Chunking audio at fixed intervals (size={chunk_size_ms}ms, overlap={chunk_overlap_ms}ms)")
            duration_ms = len(audio)
            starts = range(0, duration_ms, chunk_size_ms - chunk_overlap_ms)
            
            for i, start_ms in enumerate(starts):
                end_ms = min(start_ms + chunk_size_ms, duration_ms)
                chunk = audio[start_ms:end_ms]
                
                # Skip chunks that are too short (less than 1 second)
                if len(chunk) < 1000:
                    continue
                    
                chunks_with_timestamps.append({
                    "chunk": chunk,
                    "start_time": start_ms,
                    "end_time": end_ms,
                    "duration": end_ms - start_ms,
                    "chunk_id": i
                })
                
        else:
            raise ValueError(f"Unsupported chunking method: {chunk_method}")
            
        logger.info(f"Generated {len(chunks_with_timestamps)} audio chunks")
        return chunks_with_timestamps
        
    def _process_wav2vec2(self, audio_array, sample_rate=16000):
        """
        Generate embeddings using Wav2Vec2 model.
        
        Args:
            audio_array (numpy.ndarray): Audio array
            sample_rate (int): Sample rate of the audio
            
        Returns:
            numpy.ndarray: Audio embedding
        """
        # Resample if needed
        if sample_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            
        # Process with Wav2Vec2
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Use the mean of the last hidden state as the embedding
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings[0]  # Return the first embedding (batch size is 1)
        
    def _process_clap(self, audio_array, sample_rate=16000):
        """
        Generate embeddings using CLAP model.
        
        Args:
            audio_array (numpy.ndarray): Audio array
            sample_rate (int): Sample rate of the audio
            
        Returns:
            numpy.ndarray: Audio embedding
        """
        # CLAP needs specific preprocessing - ensure we've loaded the model properly
        waveform = torch.tensor(audio_array).float()
        # Normalize waveform if needed
        if waveform.abs().max() > 1.0:
            waveform = waveform / waveform.abs().max()
        
        # Convert to mono if needed
        if len(waveform.shape) > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
            
        # Process with CLAP
        waveform = waveform.to(self.device)
        with torch.no_grad():
            audio_embedding = self.model.encode_audio(waveform.unsqueeze(0), sample_rate)
            audio_embedding = audio_embedding.cpu().numpy()
            
        return audio_embedding[0]  # Return the first embedding (batch size is 1)
        
    def generate_embedding(self, audio_chunk):
        """
        Generate an embedding for an audio chunk.
        
        Args:
            audio_chunk (dict): Chunk data including the audio segment
            
        Returns:
            dict: Chunk data with added embedding
        """
        chunk = audio_chunk["chunk"]
        
        # Convert from pydub AudioSegment to numpy array
        audio_array = np.array(chunk.get_array_of_samples()).astype(np.float32)
        
        # Convert to mono if stereo
        if chunk.channels > 1:
            audio_array = audio_array.reshape((-1, chunk.channels)).mean(axis=1)
            
        # Normalize audio
        audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-10)
        
        # Generate embedding based on selected model
        if self.embedding_model_type == "wav2vec2":
            embedding = self._process_wav2vec2(audio_array, chunk.frame_rate)
        elif self.embedding_model_type == "clap":
            embedding = self._process_clap(audio_array, chunk.frame_rate)
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_model_type}")
            
        # Add embedding to chunk data
        result = audio_chunk.copy()
        result["embedding"] = embedding
        result["embedding_model"] = self.embedding_model_type
        result["embedding_dim"] = embedding.shape[0]
        
        return result
        
    def process_audio_file(self, audio_path, chunk_method="silence", **chunk_params):
        """
        Process an entire audio file: chunk it and generate embeddings.
        
        Args:
            audio_path (str): Path to the audio file
            chunk_method (str): Method for chunking
            **chunk_params: Additional parameters for chunking
            
        Returns:
            list: List of processed chunks with embeddings
        """
        # Get the audio file metadata
        filename = os.path.basename(audio_path)
        file_id = os.path.splitext(filename)[0]
        
        logger.info(f"Processing audio file: {filename}")
        
        # Chunk the audio
        chunks = self.chunk_audio(audio_path, chunk_method, **chunk_params)
        
        # Generate embeddings for each chunk
        processed_chunks = []
        
        for i, chunk_data in enumerate(chunks):
            logger.info(f"Generating embedding for chunk {i+1}/{len(chunks)}")
            try:
                # Generate embedding
                processed_chunk = self.generate_embedding(chunk_data)
                
                # Add file metadata
                processed_chunk["file_id"] = file_id
                processed_chunk["file_path"] = audio_path
                processed_chunk["chunk_method"] = chunk_method
                
                processed_chunks.append(processed_chunk)
                
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {str(e)}")
                # Continue with next chunk
                
        logger.info(f"Processed {len(processed_chunks)} chunks with embeddings")
        return processed_chunks
        
    def text_to_embedding(self, text):
        """
        Convert text to an audio-compatible embedding for hybrid search.
        
        Args:
            text (str): Text to convert to embedding
            
        Returns:
            numpy.ndarray: Embedding vector
        """
        # For CLAP model, we can directly convert text to embedding
        if self.embedding_model_type == "clap":
            text_tokens = open_clip.tokenize([text]).to(self.device)
            with torch.no_grad():
                text_embedding = self.model.encode_text(text_tokens)
                text_embedding = text_embedding.cpu().numpy()
            return text_embedding[0]
        
        # For Wav2Vec2, this is more challenging
        # One approach is to implement a mapping function or use a multimodal model
        # For now, we'll raise an exception
        else:
            raise NotImplementedError(
                f"Text-to-embedding not implemented for {self.embedding_model_type}"
            )