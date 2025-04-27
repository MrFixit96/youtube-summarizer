# YouTube Video Summarizer

A Python application that downloads YouTube videos, transcribes their content,
and generates summaries using open-source AI models. Now with Audio RAG (Retrieval-Augmented Generation) and knowledge graph integration!

## Features

- Download videos from YouTube
- Extract audio from videos
- Transcribe audio to text using WhisperX
- Generate concise summaries using open-source models like BART or T5
- **NEW**: Audio RAG with knowledge graph integration for enhanced summaries
  - Process audio into embeddings using CLAP or Wav2Vec2 models
  - Store and retrieve audio segments using vector database (ChromaDB)
  - Build a knowledge graph of video content and concepts
  - Generate contextually enhanced summaries with related topics

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/youtube-summarizer.git
   cd youtube-summarizer
   ```

2. Create a virtual environment using UV (recommended):

   ```bash
   # Install UV if not already installed
   pip install uv

   # Create and activate virtual environment with Python 3.12
   uv venv --python=3.12
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

   Alternatively, use traditional venv:

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required packages:

   ```bash
   # Using UV (faster)
   uv pip install -r requirements.txt
   
   # Or using pip
   pip install -r requirements.txt
   ```

4. Install FFmpeg (required for audio extraction):
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and 
     add to PATH
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`

## Configuration

1. Create a `.env` file in the project root with your settings:

   ```env
   WHISPER_MODEL=base  # tiny, base, small, medium, large
   LOCAL_MODEL_NAME=facebook/bart-large-cnn  # or another model from Hugging Face
   ```

## Project Structure

```plaintext
youtube-summarizer/
├── README.md             # This documentation
├── requirements.txt      # Project dependencies
├── setup.py              # Installation script
├── summary.txt           # Default output file for summaries
├── data/                 # Data storage
│   └── rag/              # RAG data storage
│       └── vector_db/    # Vector database storage
├── logs/                 # Log files from application runs
├── output/               # Directory for summary outputs
├── src/                  # Source code
│   ├── main.py           # Main application entry point
│   ├── config/           # Configuration files
│   ├── rag/              # RAG modules
│   │   ├── audio_processor.py  # Audio chunking and embedding
│   │   ├── vector_store.py     # Vector database interface
│   │   ├── knowledge_graph.py  # Knowledge graph implementation
│   │   └── rag_pipeline.py     # RAG workflow integration
│   ├── summarizer/       # Core summarization modules
│   └── utils/            # Helper functions and utilities
├── temp/                 # Temporary files during processing
└── tests/                # Test modules
```

## Available Summarization Models

You can use various models from Hugging Face for summarization:

- `facebook/bart-large-cnn` (default) - Good quality but requires more resources
- `sshleifer/distilbart-cnn-12-6` - Smaller version of BART, faster but less accurate
- `t5-small` - Smaller T5 model, requires fewer resources
- `philschmid/flan-t5-base-samsum` - Fine-tuned for dialogue summarization

## Usage

Basic usage:
```bash
python src/main.py https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID
```

With RAG-enhanced summaries:
```bash
python src/main.py https://www.youtube.com/watch?v=YOUTUBE_VIDEO_ID --rag
```

### Options

- `--output`, `-o`: Specify the output file for the summary (default: summary.txt)
- `--keep-files`: Keep temporary video and audio files after processing
- `--verbose`, `-v`: Enable verbose output
- `--model`: Specify a different summarization model from Hugging Face
- `--save-transcript`: Save the transcript to a file
- `--rag`: Enable RAG for enhanced summaries with knowledge graph context

## How It Works

1. **Download**: The application downloads the specified YouTube video.
2. **Audio Extraction**: It extracts the audio track from the video.
3. **Transcription**: The audio is transcribed to text using WhisperX.
4. **Summarization**: The transcribed text is summarized using an open-source model from Hugging Face.
5. **RAG Enhancement** (when enabled):
   - The audio is processed into chunks and embedded using CLAP or Wav2Vec2 models
   - Embeddings are stored in a vector database (ChromaDB) for retrieval
   - A knowledge graph is built to maintain relationships between videos and concepts
   - The summary is enhanced with contextual information from the knowledge graph

## Audio RAG System

The Audio RAG system enhances summaries by leveraging previously processed content:

1. **Audio Processing**: Audio is chunked using silence detection or fixed intervals
2. **Embedding Generation**: Audio chunks are converted to vector embeddings using CLAP or Wav2Vec2 models
3. **Vector Storage**: Embeddings are stored in ChromaDB for efficient retrieval
4. **Knowledge Graph**: Relationships between videos, concepts, and audio segments are maintained
5. **Enhanced Summaries**: Summaries include related topics and videos from the knowledge graph

As more videos are processed, the system builds a richer knowledge base, allowing for more contextually relevant summaries.

## Requirements

- Python 3.12+
- FFmpeg
- Internet connection
- At least 4GB RAM (8GB+ recommended for larger models)
- GPU (optional, but recommended for faster processing)

## Troubleshooting

### Common Issues

1. **FFmpeg Not Found**: Ensure FFmpeg is installed and added to your system PATH

   ```bash
   # Check if FFmpeg is installed
   ffmpeg -version
   ```

2. **Out of Memory Errors**: Try using a smaller model

   ```bash
   # In your .env file
   WHISPER_MODEL=tiny  # Use a smaller transcription model
   LOCAL_MODEL_NAME=sshleifer/distilbart-cnn-12-6  # Use smaller summarization model
   ```

3. **RAG-Related Issues**: Check the logs in the `logs/` directory for detailed error messages.
   If you encounter issues with the RAG system:
   - Ensure you have enough disk space for the vector database
   - Try using a different embedding model: modify `audio_processor.py` to use "wav2vec2" instead of "clap"

### Logs

All application runs are logged in the `logs/` directory with timestamps.
These logs can be useful for troubleshooting issues.

## License

MIT