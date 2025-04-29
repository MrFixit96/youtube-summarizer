#!/usr/bin/env python3
"""
YouTube Video Summarizer - Main Application

This script orchestrates the YouTube video summarization process:
1. Downloads a YouTube video
2. Extracts audio
3. Transcribes the audio
4. Generates a summary of the content
5. Optionally enhances summaries with RAG and knowledge graph
"""

import os
import sys
import argparse
import time
from dotenv import load_dotenv

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.summarizer.video_downloader import VideoDownloader
from src.summarizer.audio_extractor import AudioExtractor
from src.summarizer.transcription import Transcription
from src.summarizer.summary_generator import SummaryGenerator
from src.utils.helpers import setup_logging, clean_temp_files
from src.config.settings import LOCAL_MODEL_NAME

# Load environment variables
load_dotenv()

def process_video(url, model_name=LOCAL_MODEL_NAME, rag_enabled=False, logger=None):
    """
    Process a YouTube video through the summarization pipeline.
    
    Args:
        url (str): The YouTube video URL
        model_name (str): Name of the summarization model to use
        rag_enabled (bool): Whether to use RAG for enhanced summary
        logger: Logger instance
        
    Returns:
        tuple: (success, result_or_error_message)
    """
    try:
        # Step 1: Download the video
        logger.info(f"Downloading video from {url}...")
        downloader = VideoDownloader()
        video_path, video_info = downloader.download_video(url, return_info=True)
        logger.info(f"Downloaded video to {video_path}")
        
        # Extract video ID from URL
        video_id = downloader.extract_video_id(url)
        
        # Get video metadata
        metadata = {
            "title": video_info.get("title", "Unknown"),
            "author": video_info.get("author", "Unknown"),
            "duration": video_info.get("length", 0),
            "publish_date": str(video_info.get("publish_date", "Unknown"))
        }
        
        # Step 2: Extract audio from the video
        logger.info("Extracting audio...")
        audio_extractor = AudioExtractor()
        audio_path = audio_extractor.extract_audio(video_path)
        logger.info(f"Extracted audio to {audio_path}")
        
        # Step 3: Transcribe the audio to text
        logger.info("Transcribing audio (this may take a while)...")
        transcriber = Transcription()
        transcript = transcriber.transcribe_audio(audio_path)
        logger.info("Transcription completed")
        
        # Step 4: Generate a summary from the transcript
        logger.info("Generating summary...")
        summary_generator = SummaryGenerator(model_name=model_name)
        summary = summary_generator.generate_summary(transcript)
        logger.info("Summary generated")
        
        # Step 5 (optional): Process with RAG pipeline if enabled
        if rag_enabled:
            from src.rag.rag_pipeline import RagPipeline
            
            logger.info("Processing with RAG pipeline...")
            rag_pipeline = RagPipeline()
            
            # Process the video through RAG pipeline
            # This stores the video content in the vector database and updates the knowledge graph
            # to enhance future summaries with more contextual information
            rag_pipeline.process_new_video(
                video_id=video_id,
                audio_path=audio_path,
                transcript=transcript,
                metadata=metadata
            )
            
            # Enhance the summary with RAG context
            # This uses the existing knowledge base to add additional context
            # and create a more comprehensive summary
            enhanced_summary = rag_pipeline.enhance_summary(
                original_summary=summary,
                video_id=video_id,
                transcript=transcript
            )
            
            logger.info("Summary enhanced with RAG")
            summary = enhanced_summary
        
        # Return the results of the processing pipeline
        return True, {
            "video_path": video_path,
            "audio_path": audio_path,
            "transcript": transcript,
            "summary": summary,
            "metadata": metadata,
            "video_id": video_id,
            "rag_enhanced": rag_enabled
        }
        
    except Exception as e:
        error_msg = f"Error in processing pipeline: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def display_progress(message, duration=0.5):
    """Display a simple progress indicator with a message."""
    print(f"\r{message}", end="", flush=True)
    time.sleep(duration)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="YouTube Video Summarizer")
    parser.add_argument("url", help="YouTube video URL to summarize")
    parser.add_argument("--output", "-o", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "summary.txt"), 
                        help="Output file for the summary (default: summary.txt in project root)")
    parser.add_argument("--keep-files", action="store_true", 
                        help="Keep temporary files after processing")
    parser.add_argument("--verbose", "-v", action="store_true", 
                        help="Enable verbose output")
    parser.add_argument("--model", default=LOCAL_MODEL_NAME,
                        help=f"Summarization model to use (default: {LOCAL_MODEL_NAME})")
    parser.add_argument("--save-transcript", action="store_true",
                        help="Save the transcript to a file")
    parser.add_argument("--rag", action="store_true",
                        help="Enable RAG for enhanced summaries with context")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(verbose=args.verbose)
    
    # Display welcome message
    print("\n📺 YouTube Video Summarizer 📝\n")
    print(f"Processing: {args.url}")
    print("Starting pipeline...\n")
    
    # Process the video with progress updates
    start_time = time.time()
    stages = ["Downloading video", "Extracting audio", "Transcribing content", "Generating summary"]
    
    # Add RAG stage if enabled
    if args.rag:
        stages.append("Enhancing with RAG")
    
    # Only show progress bar in non-verbose mode
    if not args.verbose:
        for stage in stages:
            for i in range(3):
                display_progress(f"⏳ {stage}{' .' * i}")
    
    success, result = process_video(args.url, args.model, args.rag, logger)
    
    if not success:
        print(f"\n❌ {result}")
        return 1
    
    # Write summary to file
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result["summary"])
        logger.info(f"Summary written to {args.output}")
        
        # Optionally save transcript
        if args.save_transcript:
            transcript_file = os.path.splitext(args.output)[0] + "_transcript.txt"
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(result["transcript"])
            logger.info(f"Transcript written to {transcript_file}")
            print(f"Transcript saved to {transcript_file}")
        
        # Clean up temporary files if requested
        if not args.keep_files:
            clean_temp_files([result["video_path"], result["audio_path"]])
            logger.info("Temporary files removed")
        
        elapsed_time = time.time() - start_time
        rag_indicator = " (RAG-enhanced)" if result.get("rag_enhanced") else ""
        print(f"\n✅ Summary{rag_indicator} saved to {args.output}")
        print(f"Process completed in {elapsed_time:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Error in final steps: {str(e)}")
        print(f"\n❌ Error saving results: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())