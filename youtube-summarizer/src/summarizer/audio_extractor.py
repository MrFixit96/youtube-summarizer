"""
Module for extracting audio from video files.
"""

import os
import subprocess
import logging
import shutil
import time
import platform
from pathlib import Path
from src.utils.helpers import ensure_dir_exists
from src.config.settings import FFMPEG_PATH

# Configure logger
logger = logging.getLogger('youtube_summarizer')


class AudioExtractor:
    def __init__(self, output_dir="temp", output_format="wav"):
        """
        Initialize the AudioExtractor.
        
        Args:
            output_dir (str): Directory to save extracted audio files
            output_format (str): Format of the output audio file
        """
        self.output_dir = output_dir
        self.output_format = output_format
        ensure_dir_exists(output_dir)
        self.last_progress = 0
        self.extraction_start_time = 0
        
    def find_ffmpeg(self):
        """Find FFmpeg executable from settings or PATH."""
        # First check if FFMPEG_PATH is a direct path to the executable
        if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH) and os.access(FFMPEG_PATH, os.X_OK):
            logger.info(f"Using FFmpeg executable directly: {FFMPEG_PATH}")
            return FFMPEG_PATH
            
        # Check if FFMPEG_PATH is a directory containing the executable
        if FFMPEG_PATH and os.path.isdir(FFMPEG_PATH):
            ffmpeg_exe = "ffmpeg.exe" if os.name == 'nt' else "ffmpeg"
            ffmpeg_path = os.path.join(FFMPEG_PATH, ffmpeg_exe)
            if os.path.exists(ffmpeg_path) and os.access(ffmpeg_path, os.X_OK):
                logger.info(f"Using FFmpeg from configured directory: {ffmpeg_path}")
                return ffmpeg_path
        
        # Next try to find it in PATH
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            logger.info(f"Found FFmpeg in PATH: {ffmpeg_path}")
            return ffmpeg_path
        
        # Special case for Windows default installs
        if os.name == 'nt':
            # Common installation locations on Windows
            windows_paths = [
                os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'FFmpeg', 'bin', 'ffmpeg.exe'),
                os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 'FFmpeg', 'bin', 'ffmpeg.exe'),
                os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'ffmpeg', 'bin', 'ffmpeg.exe'),
                # Additional common Windows locations
                os.path.join(os.environ.get('APPDATA', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
                os.path.join(os.environ.get('USERPROFILE', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
                os.path.join(os.environ.get('USERPROFILE', ''), 'Downloads', 'ffmpeg', 'bin', 'ffmpeg.exe'),
                # Scoop package manager location
                os.path.join(os.environ.get('USERPROFILE', ''), 'scoop', 'shims', 'ffmpeg.exe'),
                # Portable/standalone installs
                'ffmpeg.exe'
            ]
            
            for path in windows_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    logger.info(f"Found FFmpeg in standard Windows location: {path}")
                    return path
        
        # Additional search for Linux/macOS systems
        elif os.name == 'posix':
            # Common installation locations on Linux/macOS
            posix_paths = [
                '/usr/bin/ffmpeg',
                '/usr/local/bin/ffmpeg',
                '/opt/local/bin/ffmpeg',
                '/opt/homebrew/bin/ffmpeg',  # Homebrew on Apple Silicon
                '/usr/local/opt/ffmpeg/bin/ffmpeg',  # Homebrew on Intel Macs
                os.path.expanduser('~/bin/ffmpeg')
            ]
            
            for path in posix_paths:
                if os.path.exists(path) and os.access(path, os.X_OK):
                    logger.info(f"Found FFmpeg in standard POSIX location: {path}")
                    return path
            
        logger.warning("FFmpeg not found in settings, PATH, or standard locations")
        return None
    
    def check_video_integrity(self, video_path):
        """
        Check if the video file is valid and can be processed.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            bool: True if video is valid, False otherwise
        """
        # Check if file exists and has size > 0
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
            
        if os.path.getsize(video_path) == 0:
            logger.error(f"Video file is empty: {video_path}")
            return False
            
        # Try to get video metadata using FFmpeg
        ffmpeg_path = self.find_ffmpeg()
        if ffmpeg_path:
            try:
                cmd = [
                    ffmpeg_path,
                    "-i", video_path,
                    "-v", "error",
                    "-f", "null",
                    "-t", "5",  # Just check the first 5 seconds
                    "-"
                ]
                result = subprocess.run(
                    cmd, 
                    stderr=subprocess.PIPE, 
                    stdout=subprocess.PIPE,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    logger.warning(f"Video integrity check failed: {result.stderr}")
                    return False
                    
                return True
            except Exception as e:
                logger.warning(f"Video integrity check error: {str(e)}")
                return True  # Continue anyway and let the extraction methods try
                
        return True  # If we can't check, assume it's valid and let extraction methods try
        
    def _progress_callback(self, stderr_line):
        """Track progress from FFmpeg output and log periodically"""
        if "time=" in stderr_line:
            # Extract the time
            time_parts = stderr_line.split("time=")[1].split()[0].split(":")
            if len(time_parts) == 3:
                hours, minutes, seconds = time_parts
                total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                
                # Calculate progress percentage if duration is known
                elapsed = time.time() - self.extraction_start_time
                
                # Only log every 10% progress or every 30 seconds
                if (total_seconds - self.last_progress > 30) or (elapsed > 30 and self.last_progress == 0):
                    logger.info(f"Audio extraction progress: {stderr_line.strip()} (elapsed: {elapsed:.1f}s)")
                    self.last_progress = total_seconds
    
    def extract_audio(self, video_path):
        """
        Extract audio from a video file.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            str: Path to the extracted audio file
        """
        # Ensure input video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Check video integrity
        if not self.check_video_integrity(video_path):
            logger.warning(f"Video integrity check failed, but attempting extraction anyway: {video_path}")
            
        # Create output filename
        video_filename = os.path.basename(video_path)
        audio_filename = os.path.splitext(video_filename)[0] + f".{self.output_format}"
        audio_path = os.path.join(self.output_dir, audio_filename)
        
        # Reset progress tracking
        self.last_progress = 0
        self.extraction_start_time = time.time()
        
        # Check for existing audio file
        if os.path.exists(audio_path):
            # Check if it's a valid audio file (not zero bytes)
            file_size = os.path.getsize(audio_path)
            if file_size > 0:
                logger.info(f"Audio file already exists at {audio_path} (size: {file_size} bytes)")
                # Check for fallback marker to provide accurate information
                if os.path.exists(audio_path + ".fallback_marker"):
                    logger.warning(f"Note: Existing audio file was created as a fallback and may not be optimal quality")
                else:
                    # Verify the audio file integrity with FFmpeg
                    ffmpeg_path = self.find_ffmpeg()
                    if ffmpeg_path:
                        try:
                            cmd = [
                                ffmpeg_path, 
                                "-v", "error", 
                                "-i", audio_path, 
                                "-f", "null", 
                                "-"
                            ]
                            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                            if result.returncode == 0:
                                logger.info("Verified existing audio file integrity")
                                return audio_path
                            else:
                                logger.warning(f"Existing audio file is corrupt, will recreate: {result.stderr}")
                                try:
                                    os.remove(audio_path)
                                except Exception as e:
                                    logger.warning(f"Could not remove corrupt audio file: {str(e)}")
                        except Exception as e:
                            logger.warning(f"Failed to verify audio integrity: {str(e)}")
                            # Still return the file if we can't verify it
                            return audio_path
                    return audio_path
            else:
                logger.warning(f"Existing audio file at {audio_path} is empty, will recreate")
                # Try to remove the empty file
                try:
                    os.remove(audio_path)
                except Exception as e:
                    logger.warning(f"Could not remove empty audio file: {str(e)}")
        
        # Find FFmpeg
        ffmpeg_path = self.find_ffmpeg()
        
        # Track success of each method
        extraction_successful = False
        
        try:
            # Method 1: Try to use FFmpeg (preferred method)
            if ffmpeg_path:
                try:
                    # Extract audio using ffmpeg directly with subprocess
                    command = [
                        ffmpeg_path,
                        '-i', video_path,
                        '-vn',  # No video
                        '-acodec', 'pcm_s16le',  # PCM 16-bit audio codec
                        '-ac', '1',  # Mono
                        '-ar', '16000',  # 16kHz sample rate
                        '-y',  # Overwrite output file if it exists
                        audio_path
                    ]
                    
                    logger.info(f"Running FFmpeg command: {' '.join(command)}")
                    
                    # Use process pipes for progress tracking
                    process = subprocess.Popen(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # Process the output to track progress
                    while True:
                        stderr_line = process.stderr.readline()
                        if not stderr_line and process.poll() is not None:
                            break
                        if stderr_line:
                            self._progress_callback(stderr_line)
                    
                    returncode = process.wait()
                    
                    if returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                        logger.info(f"Audio extracted to: {audio_path} (elapsed: {time.time() - self.extraction_start_time:.1f}s)")
                        extraction_successful = True
                        return audio_path
                    else:
                        error_message = f"FFmpeg error (code {returncode})"
                        logger.warning(error_message)
                        # Will try fallback methods below
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg process timed out, trying alternative methods")
                except Exception as e:
                    logger.warning(f"FFmpeg extraction failed with error: {str(e)}")
            else:
                logger.warning("FFmpeg not found, will try alternative methods")
            
            # If FFmpeg failed or wasn't found, try fallback methods
            
            # Method 2: Try using moviepy as fallback
            if not extraction_successful:
                try:
                    import moviepy.editor as mp
                    logger.info("Attempting audio extraction using moviepy")
                    
                    # Wrap the moviepy operation in a try-except block for each step
                    try:
                        logger.info(f"Loading video file with moviepy: {video_path}")
                        video_clip = mp.VideoFileClip(video_path)
                        
                        if video_clip.audio is not None:
                            audio_clip = video_clip.audio
                            logger.info(f"Writing audio to {audio_path} using moviepy")
                            audio_clip.write_audiofile(audio_path, fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
                            video_clip.close()
                            audio_clip.close()
                            
                            if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                                logger.info(f"Audio extracted using moviepy to: {audio_path}")
                                extraction_successful = True
                                return audio_path
                            else:
                                logger.warning("Moviepy created an empty audio file")
                        else:
                            logger.warning("Video has no audio track according to moviepy")
                            video_clip.close()
                    except Exception as e:
                        logger.warning(f"Error during moviepy processing step: {str(e)}")
                        # Try to clean up any open resources
                        try:
                            if 'video_clip' in locals() and video_clip is not None:
                                video_clip.close()
                            if 'audio_clip' in locals() and audio_clip is not None:
                                audio_clip.close()
                        except:
                            pass
                except ImportError:
                    logger.warning("Moviepy not available, skipping this fallback method")
                except Exception as e:
                    logger.warning(f"Moviepy extraction failed: {str(e)}")
            
            # Method 3: Try using pydub as another fallback
            if not extraction_successful:
                try:
                    from pydub import AudioSegment
                    logger.info("Attempting audio extraction using pydub")
                    
                    # For MP4 files
                    if video_path.lower().endswith('.mp4'):
                        audio = AudioSegment.from_file(video_path, format="mp4")
                        audio.export(audio_path, format=self.output_format)
                    # For other formats
                    else:
                        audio = AudioSegment.from_file(video_path)
                        audio.export(audio_path, format=self.output_format)
                    
                    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                        logger.info(f"Audio extracted using pydub to: {audio_path}")
                        extraction_successful = True
                        return audio_path
                    else:
                        logger.warning("Pydub created an empty audio file")
                except ImportError:
                    logger.warning("Pydub not available, skipping this fallback method")
                except Exception as e:
                    logger.warning(f"Pydub extraction failed: {str(e)}")
            
            # Method 4: Try using pytube for YouTube videos
            if not extraction_successful and ('youtube' in video_path.lower() or 'youtu.be' in video_path.lower()):
                try:
                    from pytube import YouTube
                    logger.info("Detected YouTube URL, attempting extraction using pytube")
                    
                    # Download audio only using pytube
                    yt = YouTube(video_path)
                    # Get the highest quality audio stream
                    audio_stream = yt.streams.filter(only_audio=True).order_by('abr').desc().first()
                    
                    if audio_stream:
                        # Download to temp file first
                        temp_file = audio_stream.download(output_path=self.output_dir, 
                                                        filename=f"temp_{os.path.splitext(audio_filename)[0]}")
                        
                        # Convert to desired format if needed
                        if not temp_file.lower().endswith(f".{self.output_format.lower()}"):
                            if os.path.exists(ffmpeg_path):
                                # Use FFmpeg to convert to desired format
                                convert_cmd = [
                                    ffmpeg_path,
                                    '-i', temp_file,
                                    '-vn',
                                    '-acodec', 'pcm_s16le',
                                    '-ac', '1',
                                    '-ar', '16000',
                                    '-y',
                                    audio_path
                                ]
                                try:
                                    subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
                                    # Remove temp file after conversion
                                    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                                        os.remove(temp_file)
                                except Exception as e:
                                    logger.warning(f"FFmpeg conversion of pytube download failed: {str(e)}")
                                    # If conversion fails, just rename the temp file
                                    if os.path.exists(temp_file):
                                        os.rename(temp_file, audio_path)
                            else:
                                # Just rename the file if FFmpeg isn't available
                                os.rename(temp_file, audio_path)
                        else:
                            # Already in correct format, just rename
                            os.rename(temp_file, audio_path)
                        
                        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                            logger.info(f"Audio extracted using pytube to: {audio_path}")
                            extraction_successful = True
                            return audio_path
                    else:
                        logger.warning("No audio stream found in YouTube video")
                except ImportError:
                    logger.warning("Pytube not available, skipping this fallback method")
                except Exception as e:
                    logger.warning(f"Pytube extraction failed: {str(e)}")
            
            # Method 5: Try direct download for YouTube with yt-dlp as a final fallback
            if not extraction_successful and ('youtube' in video_path.lower() or 'youtu.be' in video_path.lower()):
                try:
                    # Try to use yt-dlp if available
                    yt_dlp_path = shutil.which("yt-dlp")
                    if yt_dlp_path:
                        logger.info("Attempting YouTube audio extraction using yt-dlp")
                        temp_output = os.path.join(self.output_dir, f"yt_dlp_temp_{os.path.splitext(audio_filename)[0]}.%(ext)s")
                        
                        # Run yt-dlp to extract audio
                        cmd = [
                            yt_dlp_path,
                            "-x",  # Extract audio
                            "--audio-format", "wav" if self.output_format == "wav" else "best",
                            "--audio-quality", "0",  # Best quality
                            "-o", temp_output,
                            video_path
                        ]
                        
                        try:
                            process = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
                            
                            # Find the downloaded file
                            parent_dir = Path(self.output_dir)
                            matching_files = list(parent_dir.glob(f"yt_dlp_temp_{os.path.splitext(audio_filename)[0]}.*"))
                            
                            if matching_files:
                                downloaded_file = str(matching_files[0])
                                # Convert or rename to final output
                                if ffmpeg_path and not downloaded_file.lower().endswith(f".{self.output_format.lower()}"):
                                    # Convert to desired format
                                    convert_cmd = [
                                        ffmpeg_path,
                                        '-i', downloaded_file,
                                        '-vn',
                                        '-acodec', 'pcm_s16le',
                                        '-ac', '1',
                                        '-ar', '16000',
                                        '-y',
                                        audio_path
                                    ]
                                    subprocess.run(convert_cmd, check=True, capture_output=True, text=True)
                                    # Remove original file after conversion
                                    if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                                        os.remove(downloaded_file)
                                else:
                                    # Just rename the file
                                    os.rename(downloaded_file, audio_path)
                                
                                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                                    logger.info(f"Audio extracted using yt-dlp to: {audio_path}")
                                    extraction_successful = True
                                    return audio_path
                        except subprocess.TimeoutExpired:
                            logger.warning("yt-dlp process timed out after 5 minutes")
                        except Exception as e:
                            logger.warning(f"yt-dlp extraction failed: {str(e)}")
                except Exception as e:
                    logger.warning(f"Failed to use yt-dlp: {str(e)}")
            
            # If all extraction methods fail, create a fallback marker and raise an exception
            if not extraction_successful:
                logger.error(f"All audio extraction methods failed for {video_path} after {time.time() - self.extraction_start_time:.1f}s")
                # Mark file as fallback if we can't extract it properly
                with open(audio_path + ".fallback_marker", "w") as f:
                    f.write(f"Audio extraction failed: {time.time()}")
                raise Exception("Failed to extract audio from video after trying all available methods")
                
            return audio_path
            
        except Exception as e:
            error_message = f"Failed to extract audio: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message)