"""
Module for transcribing audio to text using modern Whisper models.
"""

import os
import torch
import logging
import traceback
import numpy as np
import subprocess
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from src.config.settings import WHISPER_MODEL, DEVICE, FFMPEG_PATH

# Configure logger
logger = logging.getLogger('youtube_summarizer')


class Transcription:
    def __init__(self, model_name=None):
        """
        Initialize the Transcription.
        
        Args:
            model_name (str): Whisper model size to use (tiny, base, small, medium, large, large-v2, large-v3)
                              or full Hugging Face model ID (e.g., 'openai/whisper-large-v3')
        """
        # Map simple model names to HF model IDs if a simple name is provided
        model_map = {
            'tiny': 'openai/whisper-tiny',
            'base': 'openai/whisper-base',
            'small': 'openai/whisper-small',
            'medium': 'openai/whisper-medium',
            'large': 'openai/whisper-large',
            'large-v2': 'openai/whisper-large-v2',
            'large-v3': 'openai/whisper-large-v3',
        }
        
        raw_model_name = model_name or WHISPER_MODEL
        # Convert simple model name to full HF model ID if needed
        self.model_name = model_map.get(raw_model_name, raw_model_name)
        
        # If model name doesn't include a slash, assume it's an OpenAI model
        if '/' not in self.model_name:
            self.model_name = f'openai/whisper-{self.model_name}'
        
        logger.info(f"Using Whisper model: {self.model_name}")
        
        # Store the configured FFmpeg path or find it
        self.ffmpeg_path = FFMPEG_PATH
        
        # If FFMPEG_PATH is a directory, find the ffmpeg executable
        if self.ffmpeg_path and os.path.isdir(self.ffmpeg_path):
            # Look for ffmpeg executable in the directory
            ffmpeg_exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            potential_path = os.path.join(self.ffmpeg_path, ffmpeg_exe)
            if os.path.exists(potential_path):
                self.ffmpeg_path = potential_path
        
        logger.info(f"Using FFmpeg path: {self.ffmpeg_path}")
    
    def _load_audio(self, audio_path):
        """
        Load audio file using librosa or scipy.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            numpy.ndarray: Audio array with 16kHz sample rate
        """
        try:
            logger.info(f"Loading audio file: {audio_path}")
            import librosa
            # Use librosa's load which automatically resamples to the target sample rate
            audio_array, _ = librosa.load(audio_path, sr=16000, mono=True)
            logger.info(f"Audio loaded with librosa: shape={audio_array.shape}")
            return audio_array
        except ImportError:
            logger.info("Librosa not available, falling back to scipy")
            from scipy.io import wavfile
            import numpy as np
            
            sample_rate, audio_array = wavfile.read(audio_path)
            
            # Convert to float32 and normalize if needed
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0
                
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Resample if needed
            if sample_rate != 16000:
                from scipy import signal
                audio_array = signal.resample(
                    audio_array, 
                    int(len(audio_array) * 16000 / sample_rate)
                )
                
            logger.info(f"Audio loaded with scipy: shape={audio_array.shape}")
            return audio_array
            
    def _set_ffmpeg_environment(self):
        """
        Set environment variables for FFmpeg.
        
        Returns:
            dict: Original environment variables
        """
        # Store original environment to restore later
        original_env = os.environ.copy()
        
        # Only proceed if we have a valid FFmpeg path
        if self.ffmpeg_path:
            logger.info(f"Setting FFmpeg environment variables with path: {self.ffmpeg_path}")
            
            # Set common environment variables that audio libraries look for
            if os.path.isfile(self.ffmpeg_path):
                # If it's a file path, set the binary directly
                os.environ["FFMPEG_BINARY"] = self.ffmpeg_path
                # Also set the directory in PATH
                ffmpeg_dir = os.path.dirname(self.ffmpeg_path)
                os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            else:
                # If it's a directory, add it to PATH
                os.environ["PATH"] = f"{self.ffmpeg_path}{os.pathsep}{os.environ.get('PATH', '')}"
            
            # Set additional environment variables that might be used
            if os.path.isfile(self.ffmpeg_path):
                os.environ["IMAGEIO_FFMPEG_EXE"] = self.ffmpeg_path
            else:
                ffmpeg_exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
                os.environ["IMAGEIO_FFMPEG_EXE"] = os.path.join(self.ffmpeg_path, ffmpeg_exe)
            
            # Verify FFmpeg is now accessible
            try:
                subprocess.run(
                    ["ffmpeg", "-version"], 
                    capture_output=True, 
                    check=False
                )
                logger.info("FFmpeg is now accessible in environment")
            except Exception as e:
                logger.warning(f"FFmpeg still not accessible after environment update: {e}")
        
        return original_env
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe audio to text using Hugging Face's Whisper implementation.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            # Verify audio file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            file_size_mb = os.path.getsize(audio_path) / 1024 / 1024
            logger.info(
                f"Transcribing audio file: {audio_path} (size: {file_size_mb:.2f} MB)"
            )
            
            # Check if the file might be a fallback/mock audio file
            if os.path.exists(audio_path + ".fallback_marker"):
                logger.warning(f"Detected fallback marker for {audio_path} - this may not be real audio data")
            
            # Check for zero-byte or extremely small files
            if file_size_mb < 0.01:  # Less than 10KB
                logger.warning(f"Audio file is suspiciously small ({file_size_mb:.2f} MB), transcription may fail")
            
            # Check if CUDA is available
            device = DEVICE
            logger.info(f"Using device: {device}")
            
            # Print PyTorch version
            logger.info(f"PyTorch version: {torch.__version__}")
            
            # Try multiple approaches in sequence until one succeeds
            transcript = None
            
            # Method 1: Use the Transformers pipeline with environment setup for FFmpeg
            try:
                logger.info(f"Loading Whisper model with pipeline: {self.model_name}")
                
                # Set environment variables for FFmpeg
                original_env = self._set_ffmpeg_environment()
                
                try:
                    # Create a transcription pipeline - ensure we're set up for full audio processing
                    transcriber = pipeline(
                        "automatic-speech-recognition",
                        model=self.model_name,
                        device=device,
                        chunk_length_s=30,    # Process in 30-second chunks for memory efficiency
                        stride_length_s=5,    # Overlap between chunks to ensure continuity
                        return_timestamps=False
                    )
                    
                    logger.info("Starting transcription with Transformers pipeline")
                    
                    # Pre-load audio to help with ffmpeg issues
                    audio_array = self._load_audio(audio_path)
                    
                    # Process the entire audio file - ensure we get all content
                    audio_length_seconds = len(audio_array) / 16000
                    logger.info(f"Processing full audio file: {audio_length_seconds:.2f} seconds")
                    
                    # FIXED: Don't pass sampling_rate directly to the pipeline - it should be in the audio kwarg options
                    result = transcriber(
                        {"array": audio_array, "sampling_rate": 16000},  # Pass as a dictionary with the array and sampling_rate
                        batch_size=1,         # Important: Process one batch at a time for memory efficiency
                        generate_kwargs={      # These settings ensure we get the full content
                            "task": "transcribe",
                            "language": "en",
                            "max_new_tokens": None,  # Don't limit output token length
                        }
                    )
                    
                    # Get the full transcript text
                    if isinstance(result, dict):
                        transcript = result["text"]
                    else:
                        transcript = result
                    
                    logger.info("Transcription completed successfully")
                    logger.info(f"Transcript length: {len(transcript.split())} words")
                    
                    # Log a sample of the transcript
                    sample = transcript[:100] + "..." if len(transcript) > 100 else transcript
                    logger.info(f"Transcript sample: {sample}")
                    
                except RuntimeError as rt_error:
                    # Check for specific CUDA out of memory errors
                    if "CUDA out of memory" in str(rt_error):
                        logger.warning(f"CUDA out of memory error: {rt_error}")
                        logger.warning("Falling back to CPU processing")
                        
                        # Try again with CPU
                        try:
                            logger.info("Recreating pipeline with CPU device")
                            transcriber = pipeline(
                                "automatic-speech-recognition",
                                model=self.model_name,
                                device="cpu",
                                chunk_length_s=30,
                                stride_length_s=5,
                                return_timestamps=False
                            )
                            
                            # Use the already loaded audio
                            result = transcriber(
                                {"array": audio_array, "sampling_rate": 16000},
                                batch_size=1,
                                generate_kwargs={
                                    "task": "transcribe",
                                    "language": "en",
                                    "max_new_tokens": None,
                                }
                            )
                            
                            # Get the full transcript text
                            if isinstance(result, dict):
                                transcript = result["text"]
                            else:
                                transcript = result
                                
                            logger.info("CPU transcription successful")
                        except Exception as cpu_error:
                            logger.warning(f"CPU fallback also failed: {cpu_error}")
                    else:
                        # Other runtime errors
                        logger.warning(f"Runtime error: {rt_error}")
                        
                except Exception as e:
                    logger.warning(f"Pipeline approach failed: {str(e)}")
                    
                finally:
                    # Restore original environment
                    os.environ.clear()
                    os.environ.update(original_env)
                    logger.info("Restored original environment variables")
                
            except Exception as e:
                logger.warning(f"Pipeline approach failed completely: {str(e)}")
            
            # Method 2: Manual processor and model approach if pipeline failed
            if transcript is None:
                logger.info("Falling back to manual processor and model approach")
                
                try:
                    logger.info(f"Loading Whisper processor and model: {self.model_name}")
                    processor = WhisperProcessor.from_pretrained(self.model_name)
                    model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(device)
                    
                    # Load the audio file as a numpy array
                    audio_array = self._load_audio(audio_path)
                    audio_length_seconds = len(audio_array) / 16000
                    logger.info(f"Audio length: {audio_length_seconds:.2f} seconds")
                    
                    # Process long audio in chunks (16000 * 30 = 480000 samples for 30 seconds)
                    chunk_size = 480000  # 30 seconds per chunk
                    stride = 80000      # 5 second overlap for continuity
                    
                    all_transcriptions = []
                    
                    # Calculate number of chunks
                    num_chunks = int(np.ceil(len(audio_array) / (chunk_size - stride)))
                    logger.info(f"Processing audio in {num_chunks} chunks")
                    
                    for i in range(num_chunks):
                        # Calculate chunk start and end, with overlap
                        start = max(0, i * (chunk_size - stride))
                        end = min(len(audio_array), start + chunk_size)
                        
                        # Extract the chunk
                        chunk = audio_array[start:end]
                        chunk_duration = len(chunk) / 16000
                        logger.info(
                            f"Processing chunk {i+1}/{num_chunks} ({chunk_duration:.2f} seconds)"
                        )
                        
                        # Process audio chunk
                        input_features = processor(
                            chunk, 
                            sampling_rate=16000, 
                            return_tensors="pt"
                        ).input_features.to(device)
                        
                        # Try to get forced decoder ids for English
                        try:
                            forced_decoder_ids = processor.get_decoder_prompt_ids(
                                language="en", 
                                task="transcribe"
                            )
                        except Exception as e:
                            logger.warning(f"Could not get decoder prompt ids: {e}")
                            forced_decoder_ids = None
                        
                        # Generate tokens for this chunk
                        try:
                            if forced_decoder_ids is not None:
                                predicted_ids = model.generate(
                                    input_features, 
                                    forced_decoder_ids=forced_decoder_ids,
                                    max_length=448
                                )
                            else:
                                predicted_ids = model.generate(
                                    input_features, 
                                    max_length=448
                                )
                            
                            # Decode the tokens to text
                            chunk_transcription = processor.batch_decode(
                                predicted_ids, 
                                skip_special_tokens=True
                            )[0]
                            all_transcriptions.append(chunk_transcription)
                            
                        except RuntimeError as rt_error:
                            # Check for CUDA out of memory
                            if "CUDA out of memory" in str(rt_error):
                                logger.warning(f"CUDA out of memory in chunk {i+1}, moving to CPU")
                                
                                # Try the current chunk on CPU
                                try:
                                    # Move everything to CPU
                                    input_features = input_features.to("cpu")
                                    model = model.to("cpu")
                                    
                                    # Retry generation on CPU
                                    if forced_decoder_ids is not None:
                                        predicted_ids = model.generate(
                                            input_features, 
                                            forced_decoder_ids=forced_decoder_ids,
                                            max_length=448
                                        )
                                    else:
                                        predicted_ids = model.generate(
                                            input_features, 
                                            max_length=448
                                        )
                                    
                                    # Decode the tokens to text
                                    chunk_transcription = processor.batch_decode(
                                        predicted_ids, 
                                        skip_special_tokens=True
                                    )[0]
                                    all_transcriptions.append(chunk_transcription)
                                    
                                    # Keep using CPU for remaining chunks
                                    device = "cpu"
                                    
                                except Exception as cpu_error:
                                    logger.error(f"CPU processing for chunk {i+1} also failed: {cpu_error}")
                                    # Add a placeholder for failed chunks instead of stopping completely
                                    all_transcriptions.append(f"[TRANSCRIPTION ERROR IN SEGMENT {i+1}]")
                            else:
                                logger.error(f"Error processing chunk {i+1}: {rt_error}")
                                # Add a placeholder for failed chunks
                                all_transcriptions.append(f"[TRANSCRIPTION ERROR IN SEGMENT {i+1}]")
                        
                        except Exception as chunk_error:
                            logger.error(f"Error processing chunk {i+1}: {chunk_error}")
                            # Add a placeholder for failed chunks
                            all_transcriptions.append(f"[TRANSCRIPTION ERROR IN SEGMENT {i+1}]")
                    
                    # Combine all chunk transcriptions
                    if all_transcriptions:
                        transcript = " ".join(all_transcriptions)
                        logger.info("Manual transcription completed successfully")
                        logger.info(f"Transcript length: {len(transcript.split())} words")
                        
                        # Log a sample
                        sample = transcript[:100] + "..." if len(transcript) > 100 else transcript
                        logger.info(f"Transcript sample: {sample}")
                    else:
                        raise Exception("No transcription could be generated")
                    
                except Exception as e:
                    logger.error(f"Manual approach also failed: {str(e)}")
            
            # Method 3: Try using a smaller model as last resort if still no transcript
            if transcript is None:
                smaller_models = ["openai/whisper-small", "openai/whisper-base", "openai/whisper-tiny"]
                
                # Don't try models we've already tried
                current_model_name = self.model_name
                if current_model_name in smaller_models:
                    smaller_models = [m for m in smaller_models if smaller_models.index(m) > smaller_models.index(current_model_name)]
                
                for fallback_model in smaller_models:
                    if transcript is not None:
                        break
                        
                    try:
                        logger.warning(f"Attempting transcription with smaller model: {fallback_model}")
                        
                        # Set environment variables for FFmpeg
                        original_env = self._set_ffmpeg_environment()
                        
                        try:
                            # Create transcription pipeline with smaller model
                            logger.info(f"Creating fallback pipeline with {fallback_model}")
                            transcriber = pipeline(
                                "automatic-speech-recognition",
                                model=fallback_model,
                                device="cpu",  # Use CPU for fallback to avoid CUDA issues
                                chunk_length_s=30,
                                stride_length_s=5,
                                return_timestamps=False
                            )
                            
                            # Load audio
                            audio_array = self._load_audio(audio_path)
                            
                            # Transcribe
                            result = transcriber(
                                {"array": audio_array, "sampling_rate": 16000},
                                batch_size=1,
                                generate_kwargs={
                                    "task": "transcribe",
                                    "language": "en",
                                    "max_new_tokens": None,
                                }
                            )
                            
                            # Get the transcript text
                            if isinstance(result, dict):
                                transcript = result["text"]
                            else:
                                transcript = result
                                
                            logger.info(f"Fallback transcription with {fallback_model} successful")
                            logger.info(f"Transcript length: {len(transcript.split())} words")
                            
                            # Log a sample
                            sample = transcript[:100] + "..." if len(transcript) > 100 else transcript
                            logger.info(f"Transcript sample: {sample}")
                            
                        except Exception as fallback_error:
                            logger.warning(f"Fallback with {fallback_model} failed: {fallback_error}")
                            
                        finally:
                            # Restore original environment
                            os.environ.clear()
                            os.environ.update(original_env)
                            
                    except Exception as e:
                        logger.warning(f"Fallback model {fallback_model} setup failed: {str(e)}")
            
            # Final fallback: provide a placeholder if all methods failed
            if transcript is None:
                logger.error("All transcription methods failed, returning placeholder text")
                transcript = "[TRANSCRIPTION FAILED: Could not process audio file. The audio file may be corrupted, empty, or in an unsupported format. Please check the audio file or try with a different file.]"
            
            return transcript
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {str(e)}")
            traceback.print_exc()
            # Return a clear error message instead of raising an exception
            return f"[TRANSCRIPTION ERROR: {str(e)}]"