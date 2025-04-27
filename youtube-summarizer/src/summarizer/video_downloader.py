"""
Module for downloading YouTube videos.
"""

import os
import logging
import subprocess
import re
import yt_dlp
from src.utils.helpers import ensure_dir_exists
from src.config.settings import FFMPEG_PATH

# Configure logger
logger = logging.getLogger('youtube_summarizer')


class VideoDownloader:
    def __init__(self, output_dir="temp"):
        """
        Initialize the VideoDownloader.
        
        Args:
            output_dir (str): Directory to save downloaded videos
        """
        self.output_dir = output_dir
        ensure_dir_exists(output_dir)
    
    def extract_video_id(self, url):
        """
        Extract the video ID from a YouTube URL.
        
        Args:
            url (str): The YouTube URL
            
        Returns:
            str: The video ID
        """
        # Common YouTube URL patterns
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard URLs
            r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URLs
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})', # Watch URLs
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # Short URLs
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
                
        # If no match is found
        logger.warning(f"Could not extract video ID from URL: {url}")
        # Generate a simple hash as fallback
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()[:11]
    
    def download_video(self, url, return_info=False):
        """
        Download a YouTube video using yt-dlp.
        
        Args:
            url (str): The YouTube video URL
            return_info (bool): Whether to return video info with the path
            
        Returns:
            str or tuple: Path to the downloaded video file, or (path, info) if return_info=True
        """
        # Set up output path
        output_template = os.path.join(self.output_dir, '%(title)s.%(ext)s')
        
        # Get FFmpeg path from settings
        ffmpeg_path = self.find_ffmpeg()
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]/best',  # Prefer mp4 format, single stream as fallback
            'outtmpl': output_template,
            'quiet': False,
            'no_warnings': False,
            'ignoreerrors': False,
            'noplaylist': True,  # Only download single video, not playlist
            'noprogress': False,
            'logger': logger,
            'cachedir': False,
        }
        
        # If FFmpeg is found, we can use it for better quality downloads
        if ffmpeg_path:
            ydl_opts['ffmpeg_location'] = ffmpeg_path
            # With FFmpeg available, we can download best quality video+audio
            ydl_opts['format'] = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
            logger.info("Using FFmpeg for high-quality download")
        else:
            logger.warning("FFmpeg not found, using simpler format that doesn't require merging")
        
        try:
            # Extract video info first to get the title
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                logger.info(f"Video title: {info.get('title', 'Unknown title')}")
                
                # Now download the video
                logger.info(f"Downloading video from {url}")
                ydl.download([url])
                
                # Determine the output filename
                video_filename = ydl.prepare_filename(info)
                if os.path.exists(video_filename):
                    logger.info(f"Video downloaded to: {video_filename}")
                    return (video_filename, info) if return_info else video_filename
                else:
                    # Try to find the file with a different extension
                    base_name = os.path.splitext(video_filename)[0]
                    for ext in ['.mp4', '.mkv', '.webm', '.m4a']:
                        potential_file = base_name + ext
                        if os.path.exists(potential_file):
                            logger.info(f"Video downloaded to: {potential_file}")
                            return (potential_file, info) if return_info else potential_file
                    
                    # If we can't find the file, return the original predicted name
                    logger.warning(
                        f"Can't find downloaded file, returning predicted name: {video_filename}"
                    )
                    return (video_filename, info) if return_info else video_filename
                
        except Exception as e:
            error_message = f"Failed to download video: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message)
    
    def find_ffmpeg(self):
        """Find FFmpeg executable from settings or PATH."""
        if FFMPEG_PATH and os.path.exists(
            os.path.join(FFMPEG_PATH, "ffmpeg.exe" if os.name == 'nt' else "ffmpeg")
        ):
            ffmpeg_dir = FFMPEG_PATH
            logger.info(f"Using FFmpeg from settings: {ffmpeg_dir}")
            return ffmpeg_dir
        
        # Fall back to checking in PATH
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                logger.info("FFmpeg found in PATH")
                return None  # Let yt-dlp find it in PATH
        except Exception:
            pass
            
        logger.warning("FFmpeg not found in PATH or settings")
        return None