"""
Helper functions for the YouTube Video Summarizer application.
"""

import os
import logging
import shutil
from datetime import datetime


def ensure_dir_exists(directory):
    """
    Create directory if it doesn't exist.
    
    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def setup_logging(log_level=logging.INFO, verbose=False):
    """
    Set up logging for the application.
    
    Args:
        log_level: Logging level
        verbose: Whether to enable verbose output
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create logs directory if it doesn't exist
    ensure_dir_exists("logs")
    
    # Set up logging
    logger = logging.getLogger("youtube_summarizer")
    logger.setLevel(log_level)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else log_level)
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(f"logs/youtube_summarizer_{timestamp}.log")
    file_handler.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def clean_temp_files(file_paths):
    """
    Remove temporary files.
    
    Args:
        file_paths (list): List of file paths to remove
    """
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)
            

def get_video_id_from_url(url):
    """
    Extract video ID from YouTube URL.
    
    Args:
        url (str): YouTube URL
        
    Returns:
        str: YouTube video ID or None if not found
    """
    from urllib.parse import urlparse, parse_qs
    
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Get video ID from query parameters (for URLs like youtube.com/watch?v=VIDEO_ID)
    if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            query_params = parse_qs(parsed_url.query)
            return query_params.get('v', [None])[0]
    
    # Get video ID from path (for URLs like youtu.be/VIDEO_ID)
    elif parsed_url.hostname in ('youtu.be'):
        return parsed_url.path.lstrip('/')
    
    return None