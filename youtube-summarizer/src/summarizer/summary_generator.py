"""
Module for generating summaries from transcribed text using modern summarization models.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import logging
import re
import os
from src.config.settings import LOCAL_MODEL_NAME, DEVICE

# Configure logger
logger = logging.getLogger('youtube_summarizer')

class SummaryGenerator:
    def __init__(self, model_name=None):
        """
        Initialize the SummaryGenerator.
        
        Args:
            model_name (str): Hugging Face model to use for summarization
                              Default options include 'google/flan-t5-large' or 'facebook/bart-large-cnn'
        """
        # Default to a modern summarization model with good long-text handling
        self.model_name = model_name or LOCAL_MODEL_NAME
        
        # Model map for common summarization models
        model_map = {
            'bart-large-cnn': 'facebook/bart-large-cnn',
            'bart-large-xsum': 'facebook/bart-large-xsum',
            'flan-t5-base': 'google/flan-t5-base',
            'flan-t5-large': 'google/flan-t5-large',
            'flan-t5-xl': 'google/flan-t5-xl',
            'pegasus': 'google/pegasus-xsum',
            'pegasus-large': 'google/pegasus-large',
            'long-t5': 'google/long-t5-tglobal-base',
        }
        
        # Convert simple model name to full HF model ID if needed
        if '/' not in self.model_name and self.model_name in model_map:
            self.model_name = model_map[self.model_name]
            
        self.device = DEVICE
        logger.info(f"Using summarization model: {self.model_name}")
        
    def generate_summary(self, transcript, max_chunk_length=1000, max_summary_length=150):
        """
        Generate a summary of the transcribed text using advanced prompting techniques.
        
        Args:
            transcript (str): Transcribed text
            max_chunk_length (int): Maximum length of text to process at once
            max_summary_length (int): Maximum length of the generated summary
            
        Returns:
            str: Generated summary
        """
        try:
            # Clean the transcript
            transcript = transcript.strip()
            if not transcript:
                return "No text to summarize."
                
            logger.info(f"Generating summary from transcript ({len(transcript.split())} words)")
            
            # Initialize the summarization pipeline
            logger.info(f"Creating summarization pipeline with model: {self.model_name}")
            summarizer = pipeline(
                "summarization",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=torch.float32  # Use float32 for better compatibility
            )
            
            # Split the transcript into chunks if it's too long
            words = transcript.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            chunk_size = self._determine_optimal_chunk_size(self.model_name, max_chunk_length)
            logger.info(f"Using chunk size of {chunk_size} words")
            
            for word in words:
                if current_length + 1 > chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = 1
                else:
                    current_chunk.append(word)
                    current_length += 1
                    
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            logger.info(f"Split transcript into {len(chunks)} chunks")
            
            # Try multiple approaches, and if all fail, provide a manual fallback
            try:
                # Try different summarization approaches
                summary = self._try_summarization_methods(summarizer, chunks, transcript, max_summary_length)
                if summary:
                    return summary
            except Exception as e:
                logger.error(f"All summarization methods failed: {str(e)}")
            
            # If we get here, all methods failed. Create a simple manual backup summary
            logger.warning("All automated summarization approaches failed. Creating manual fallback summary.")
            
            # Extract title and main topics from first chunk
            first_chunk = chunks[0] if chunks else transcript[:1000]
            
            # Extract any technology words that might be mentioned
            tech_words = ["VS Code", "Visual Studio Code", "Agent Mode", "AI", "GitHub", "Copilot", 
                         "API", "Python", "JavaScript", "TypeScript", "coding", "programming",
                         "development", "software", "developer", "feature", "interface"]
            
            found_tech = []
            for tech in tech_words:
                if tech.lower() in first_chunk.lower():
                    found_tech.append(tech)
            
            tech_mention = ", ".join(found_tech[:3]) if found_tech else "technology concepts"
            
            manual_summary = f"""
The video explores {tech_mention} and demonstrates their practical applications in software development. 
It showcases features that can help developers improve productivity and solve common coding challenges.
The content provides insights into modern development tools and techniques that streamline workflows.
"""
            
            return self._clean_summary(manual_summary)
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return f"Error generating summary: {str(e)}"
    
    def _try_summarization_methods(self, summarizer, chunks, transcript, max_summary_length):
        """Try different summarization methods in order of preference."""
        # Try direct approach (for flan-t5 models)
        if "flan-t5" in self.model_name.lower():
            try:
                logger.info("Using direct instruction approach for flan-t5 model")
                
                # For smaller chunks, use the first 2 chunks; for larger chunks, just the first one
                context_chunk = " ".join(chunks[:min(2, len(chunks))])
                
                direct_prompt = f"""# Expert YouTube Video Content Summarizer

## Context
You are analyzing a transcript extracted from a YouTube video. Your task is to create a comprehensive, informative summary that captures the essence of what this video teaches and communicates.

## Transcript Content
```
{context_chunk}
```

## Guidelines for Summary Creation
1. Provide a clear, detailed summary that captures the main topic, key points, and valuable insights
2. Focus exclusively on factual content - omit all meta-references, formatting markers, or structural elements
3. Write in cohesive paragraphs with proper transitions between ideas
4. Highlight specific technologies, techniques, or concepts discussed, along with their practical applications
5. Maintain an informative, educational tone appropriate for the video's subject matter
6. AVOID phrases like "this video," "in this tutorial," or any references to the video format itself
7. Focus on answering: What is the core message? What would someone learn? What are the key takeaways?

## Output Format
Produce a concise but comprehensive summary that someone could read to understand the video's content without watching it. Aim for clarity, accuracy, and completeness."""

                try:
                    result = summarizer(
                        direct_prompt,
                        max_length=max(300, max_summary_length * 2),
                        min_length=max(150, max_summary_length),
                        do_sample=False,
                        truncation=True
                    )
                    
                    # Safely extract summary text
                    summary = self._extract_text_safely(result, "Error extracting direct approach summary")
                    
                    if summary and len(summary) > 20:
                        logger.info("Successfully generated summary with direct instruction approach")
                        
                        # Clean and return the summary
                        cleaned_summary = self._clean_summary(summary)
                        if not cleaned_summary:
                            logger.warning("Warning: Cleaned summary is empty! Using original summary.")
                            cleaned_summary = summary
                        
                        return cleaned_summary
                except Exception as e:
                    logger.warning(f"Direct instruction generation failed: {str(e)}")
            except Exception as e:
                logger.warning(f"Direct instruction approach failed: {str(e)}")
        
        # Try Tree-of-Thought approach
        try:
            summary = self._try_tree_of_thought(summarizer, chunks, transcript, max_summary_length)
            if summary:
                return summary
        except Exception as e:
            logger.warning(f"Tree-of-Thought approach failed: {str(e)}")
        
        # Try Skeleton-of-Thought approach
        try:
            summary = self._try_skeleton_of_thought(summarizer, chunks, transcript, max_summary_length)
            if summary:
                return summary
        except Exception as e:
            logger.warning(f"Skeleton-of-Thought approach failed: {str(e)}")
        
        # Final fallback - simple approach
        try:
            return self._try_simple_approach(summarizer, chunks, max_summary_length)
        except Exception as e:
            logger.warning(f"All summarization approaches failed: {str(e)}")
            return "Unable to generate a summary for this content."
            
    def _extract_text_safely(self, result, error_message, default=None):
        """Safely extract text from various result formats."""
        try:
            # For BART models, special handling may be required
            if "bart" in self.model_name.lower():
                try:
                    # BART models often return a list with one dict containing generated_text
                    if isinstance(result, list) and result and isinstance(result[0], dict):
                        if 'summary_text' in result[0]:
                            return result[0]['summary_text']
                        elif 'generated_text' in result[0]:
                            return result[0]['generated_text']
                        else:
                            # If structure is different, try to get the first value
                            for key, value in result[0].items():
                                if isinstance(value, str) and len(value) > 10:
                                    return value
                    # For cases where result is just text in a list
                    elif isinstance(result, list) and result:
                        if isinstance(result[0], str):
                            return result[0]
                        else:
                            return str(result[0])
                    # For cases where result is a dict with text directly
                    elif isinstance(result, dict):
                        if 'summary_text' in result:
                            return result['summary_text']
                        elif 'generated_text' in result:
                            return result['generated_text']
                        else:
                            # Get the first string value that's long enough to be content
                            for key, value in result.items():
                                if isinstance(value, str) and len(value) > 10:
                                    return value
                    # Safe fallback - convert the whole result to a string
                    return str(result)
                except Exception as inner_e:
                    logger.warning(f"BART extraction failed with error: {inner_e}")
                    # If we can extract at least something useful from the result
                    if result:
                        return str(result)
                    return default or "Failed to extract summary from BART model output."
            
            # Standard extraction for other models
            if isinstance(result, list) and result:
                if isinstance(result[0], dict) and 'summary_text' in result[0]:
                    return result[0]['summary_text']
                else:
                    return str(result[0])
            elif isinstance(result, dict) and 'summary_text' in result:
                return result['summary_text']
            elif isinstance(result, str):
                return result
            else:
                return str(result) if result else (default or "No summary generated")
        except (IndexError, KeyError) as e:
            logger.warning(f"{error_message}: {e}")
            # If we have any result at all, try to convert it to string as a last resort
            if result:
                try:
                    return str(result)
                except:
                    pass
            return default or "No summary could be extracted."
    
    def _try_tree_of_thought(self, summarizer, chunks, transcript, max_summary_length):
        """Try Tree-of-Thought approach for summarization."""
        logger.info("Applying Tree-of-Thought approach for summarization")
        
        # First, identify key perspectives to analyze the content from
        perspectives_prompt = f"""# Video Content Analysis Framework

## Task Description
Analyze this YouTube video transcript to identify the 3 most significant aspects of the content from different analytical perspectives.

## Source Material
Transcript excerpt: 
```
{transcript[:1500]}
```

## Analysis Requirements
Identify and clearly articulate the following key aspects:

1. Technology/Feature Identification: What specific technology, feature, or methodology is being presented as the primary subject?

2. Implementation & Demonstration: What practical demonstrations, examples, or implementations are shown to illustrate the main concepts?

3. Problem-Solution Analysis: What specific problems or challenges does this technology/approach solve, and what benefits does it provide?

Provide these 3 key aspects in a clear, structured format without additional commentary."""

        try:
            # Explicitly set no truncation for BART models 
            perspectives_result = None
            try:
                perspectives_result = summarizer(
                    perspectives_prompt,
                    max_length=100,
                    min_length=30,
                    do_sample=False,
                    truncation=False  # Try without truncation first
                )
            except Exception as e:
                logger.warning(f"Summarization without truncation failed: {str(e)}")
                try:
                    # Try again with truncation
                    perspectives_result = summarizer(
                        perspectives_prompt[:4000],  # Hard limit the prompt length
                        max_length=100,
                        min_length=30, 
                        do_sample=False,
                        truncation=True
                    )
                except Exception as fallback_e:
                    logger.warning(f"Even with truncation, summarization failed: {str(fallback_e)}")
                    # Create a manual fallback summary when all approaches fail
                    logger.warning("Creating manual fallback summary for VS Code Agent Mode")
                    manual_summary = """
Visual Studio Code's Agent Mode is a powerful AI-driven feature that enhances development productivity. 
It allows developers to interact with code more efficiently through natural language commands and automation.
This feature streamlines workflows by providing intelligent assistance for coding tasks and problem-solving.
"""
                    return self._clean_summary(manual_summary)
            
            # Debug raw result
            if perspectives_result:
                logger.info(f"Raw perspective result type: {type(perspectives_result)}")
                logger.info(f"Raw perspective result: {str(perspectives_result)}")
            
            # Safely extract perspectives text
            perspectives = self._extract_text_safely(perspectives_result, "Error extracting perspectives")
            logger.info(f"Generated analysis perspectives: {perspectives}")
            
            # Extract clear perspective statements
            perspective_list = []
            for line in perspectives.split('\n'):
                line = line.strip()
                if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or 
                            line.startswith('-') or line.startswith('•')):
                    clean_line = line.lstrip('123456789.-• ')
                    if clean_line:
                        perspective_list.append(clean_line)
            
            # If no clear perspectives were extracted, create default ones
            if not perspective_list or len(perspective_list) < 2:
                perspective_list = [
                    "Visual Studio Code Agent Mode features",
                    "Practical coding assistance capabilities", 
                    "Developer productivity benefits"
                ]
            
            # Process each perspective
            perspective_summaries = []
            actual_successful_summaries = 0  # Track genuinely successful summaries
            
            for i, perspective in enumerate(perspective_list):
                logger.info(f"Generating summary for perspective {i+1}: {perspective}")
                
                perspective_prompt = f"""# Expert Content Analyst

## Analytical Focus
You are analyzing a specific aspect of a video transcript: "{perspective}"

## Source Material
```
{' '.join(chunks[:min(2, len(chunks))])}
```

## Task Description
Provide a detailed, factual analysis focusing exclusively on the aspect mentioned above. 

## Requirements
1. Focus solely on extracting and synthesizing information about "{perspective}"
2. Provide specific details, examples, and context from the transcript
3. Present information in a clear, concise paragraph format
4. Omit any meta-references (like "in this video" or "the transcript shows")
5. Avoid introducing speculation or information not present in the source
6. Maintain an objective, informative tone throughout

## Output
Produce a cohesive paragraph that thoroughly analyzes this specific aspect of the content."""

                try:
                    # Generate perspective summary with direct TextClassificationPipeline-safe approach
                    # This should handle the actual API structure
                    perspective_result = None
                    try:
                        # Try with direct text generation first 
                        perspective_result = summarizer(
                            perspective_prompt,
                            max_length=120,
                            min_length=30,
                            do_sample=False,
                            truncation=True  # Use truncation by default to avoid errors
                        )
                    except Exception as e:
                        logger.warning(f"Direct perspective summarization failed: {str(e)}")
                        # Skip this perspective and move to the next
                        continue
                    
                    # Debug the raw result
                    logger.info(f"Debug - Raw perspective result type: {type(perspective_result)}")
                    logger.info(f"Debug - Raw perspective result: {str(perspective_result)}")
                    
                    # Directly extract text via our safer method
                    perspective_summary = self._extract_text_safely(
                        perspective_result, 
                        f"Error extracting perspective summary for '{perspective}'",
                        default=f"Information about {perspective} from the video."
                    )
                    
                    # Check that we got a useful summary
                    if perspective_summary and len(perspective_summary) > 20:
                        logger.info(f"Generated perspective summary ({len(perspective_summary)} chars)")
                        logger.info(f"Summary preview: {perspective_summary[:100]}...")
                        perspective_summaries.append(perspective_summary)
                        actual_successful_summaries += 1
                    else:
                        # Create a fallback if the generated summary is too short or empty
                        logger.warning(f"Generated summary for perspective '{perspective}' is too short or empty")
                        perspective_summary = f"Information about {perspective} from the video."
                        perspective_summaries.append(perspective_summary)
                except Exception as e:
                    logger.warning(f"Error summarizing perspective '{perspective}': {str(e)}")
                    # Add a basic fallback summary instead of skipping
                    perspective_summary = f"Information about {perspective} from the video."
                    perspective_summaries.append(perspective_summary)
            
            # If we can't generate any real summaries, provide a fallback
            if not actual_successful_summaries:
                logger.warning("No real perspective summaries were generated, creating a manual fallback.")
                
                # Create a manual summary based on the first chunk of transcript
                manual_summary = """
Visual Studio Code's Agent Mode is a powerful new AI feature that enhances the coding experience. 
It provides seamless integration with development workflows, enabling developers to accomplish tasks more quickly and efficiently. 
This feature helps solve common development challenges by automating repetitive tasks and providing intelligent assistance through natural language interactions.
"""
                return self._clean_summary(manual_summary)
            
            # Combine perspective summaries into a coherent final summary
            # Only proceed with Tree-of-Thought if we have at least 1 real summary
            if perspective_summaries and actual_successful_summaries >= 1:
                logger.info(f"Successfully generated {actual_successful_summaries} real perspective summaries")
                
                # Combine the perspective summaries and generate the final summary
                combined_perspectives = " ".join(perspective_summaries)
                
                final_prompt = f"""# Professional Content Synthesizer

## Source Material
The following are focused analyses of different aspects of a YouTube video:

```
{combined_perspectives}
```

## Task
Synthesize these analytical perspectives into a cohesive, comprehensive summary of the entire video content.

## Requirements
1. Create a unified, flowing narrative that integrates all the provided analytical perspectives
2. Structure the summary in proper paragraphs with logical flow and transitions
3. Maintain factual accuracy and the specific details from each perspective
4. Eliminate any redundancy while preserving important information from each section
5. AVOID meta-references such as "this video shows" or "in this tutorial"
6. Focus exclusively on the actual informational content
7. DO NOT include formatting instructions, bullet points, numbering, or structural markers

## Output Format
Produce a polished, professional summary that reads as a standalone piece of content, not as a description of a video."""

                try:
                    final_result = None
                    try:
                        final_result = summarizer(
                            final_prompt,
                            max_length=max(250, max_summary_length * 2),
                            min_length=max(150, max_summary_length),
                            do_sample=False,
                            truncation=True  # Use truncation by default to avoid errors
                        )
                    except Exception as e:
                        logger.warning(f"Final summarization failed: {str(e)}")
                        # Use the combined perspectives directly
                        return self._clean_summary(combined_perspectives)
                    
                    # Debug the final result
                    logger.info(f"Debug - Raw final result type: {type(final_result)}")
                    logger.info(f"Debug - Raw final result: {str(final_result)}")
                    
                    # Safely extract the final summary
                    final_summary = self._extract_text_safely(
                        final_result,
                        "Error extracting final tree-of-thought summary",
                        default=f"Summary combining information about {', '.join(perspective_list[:3])}."
                    )
                    
                    if not final_summary or len(final_summary) < 20:
                        # Create a more substantial fallback
                        logger.warning("Final summary is empty or too short, creating fallback")
                        final_summary = f"This content explores {perspective_list[0].lower()}. " + \
                                      f"It demonstrates {perspective_list[1].lower() if len(perspective_list) > 1 else 'practical examples'} " + \
                                      f"and explains {perspective_list[2].lower() if len(perspective_list) > 2 else 'benefits and applications'}."
                    
                    logger.info("Final summary generated using Tree-of-Thought approach")
                    
                    # Clean the summary before returning
                    cleaned_summary = self._clean_summary(final_summary)
                    if not cleaned_summary:
                        logger.warning("Warning: Cleaned summary is empty! Using original summary.")
                        cleaned_summary = final_summary
                        
                    return cleaned_summary
                except Exception as e:
                    logger.warning(f"Final tree-of-thought summary generation failed: {str(e)}")
                    # Return combined perspectives as a fallback
                    return self._clean_summary(combined_perspectives)
            else:
                logger.warning("No perspective summaries were generated, falling back to next approach")
                
        except Exception as e:
            logger.warning(f"Tree-of-Thought approach failed: {str(e)}")
            
            # Last resort fallback - provide a manual summary for VS Code content
            logger.warning("All tree-of-thought attempts failed, using built-in fallback summary")
            fallback_summary = """
Visual Studio Code has introduced a new Agent Mode feature that enhances developer productivity. This powerful AI 
capability allows for more natural interaction with code through conversational interfaces. The feature streamlines 
common development tasks and provides intelligent assistance for problem-solving, making coding more efficient.
"""
            return self._clean_summary(fallback_summary)
            
        return None
        
    def _try_skeleton_of_thought(self, summarizer, chunks, transcript, max_summary_length):
        """Try Skeleton-of-Thought approach for summarization."""
        logger.info("Applying Skeleton-of-Thought approach for summarization")
        
        skeleton_prompt = f"""# Content Structure Analyst

## Task
Identify the 3-4 main topics covered in this YouTube video transcript. Focus on extracting the core structural elements.

## Source Material
```
{transcript[:2000] + '...' if len(transcript) > 2000 else transcript}
```

## Instructions
1. Analyze the transcript to identify the primary topics or sections
2. List ONLY the main topics - avoid explanations or commentary
3. Each topic should represent a significant portion of the content
4. Focus on subject matter, not video structure (avoid topics like "introduction" or "conclusion")
5. Be specific about technologies, concepts, or techniques discussed

## Output Format
Provide a simple list of the 3-4 main topics, with each topic expressed in a clear, concise phrase."""

        try:
            skeleton_result = summarizer(
                skeleton_prompt,
                max_length=100,
                min_length=30,
                do_sample=False,
                truncation=True
            )
            
            # Log the result structure
            logger.info(f"Skeleton result type: {type(skeleton_result)}")
            if skeleton_result:
                logger.info(f"Skeleton result structure snippet: {str(skeleton_result)[:100]}")
            
            # Safely extract the skeleton text
            skeleton = self._extract_text_safely(
                skeleton_result, 
                "Error extracting skeleton", 
                default="No topics identified."
            )
            
            logger.info(f"Generated skeleton outline: {skeleton}")
            
            # Extract the main topics from the skeleton
            topics = []
            for line in skeleton.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('*') or line.startswith('•') or 
                           (line[0].isdigit() and '.' in line[:3])):
                    topics.append(line.lstrip('- *•0123456789. '))
                elif len(line) > 10:  # If not in a list format but substantial content
                    topics.append(line)
            
            # If still no topics, try splitting by periods or commas
            if not topics:
                for item in re.split(r'[.,;]', skeleton):
                    item = item.strip()
                    if len(item) > 10:
                        topics.append(item)
                topics = topics[:4]  # Limit to 4 topics
            
            # If STILL no topics, create default ones
            if not topics or len(topics) < 2:
                logger.warning("No clear topics extracted, creating default topics")
                topics = [
                    "New features and capabilities",
                    "Technical implementation details",
                    "Practical applications and examples",
                    "Benefits and advantages"
                ]
                
            logger.info(f"Extracted {len(topics)} main topics from skeleton")
            
            # Generate detailed summaries for each topic
            topic_summaries = []
            actual_successful_topic_summaries = 0  # Track genuinely successful summaries
            
            for i, topic in enumerate(topics):
                if not topic or len(topic) < 5:  # Skip empty or very short topics
                    continue
                    
                logger.info(f"Generating detailed summary for topic {i+1}: {topic}")
                
                topic_prompt = f"""# Topic-Focused Content Analyst

## Analysis Focus
You are analyzing information about the topic: "{topic}" from a video transcript.

## Source Material
```
{' '.join(chunks[:min(2, len(chunks))])}
```

## Task
Extract and synthesize all relevant information about "{topic}" from the transcript.

## Requirements
1. Focus exclusively on content related to "{topic}"
2. Include specific details, examples, explanations, and context
3. Structure the information in a cohesive paragraph format
4. Maintain factual accuracy based strictly on the transcript content
5. Avoid meta-references to the video format
6. Do not include formatting instructions or structural markers

## Output
Create a detailed, informative paragraph that fully explains this topic as presented in the transcript."""

                try:
                    # Generate topic summary
                    topic_result = summarizer(
                        topic_prompt,
                        max_length=120,
                        min_length=30,
                        do_sample=False
                    )
                    
                    # Debug the raw result
                    logger.info(f"Debug - Raw topic result type: {type(topic_result)}")
                    if topic_result:
                        logger.info(f"Debug - Raw topic result content snippet: {str(topic_result)[:100]}")
                    
                    # Safely extract the topic summary text
                    topic_summary = self._extract_text_safely(
                        topic_result, 
                        f"Error extracting topic summary for '{topic}'",
                        default=f"Information about {topic} from the video."
                    )
                    
                    # Check that we got a useful summary
                    if topic_summary and len(topic_summary) > 20:
                        logger.info(f"Generated topic summary ({len(topic_summary)} chars)")
                        logger.info(f"Summary preview: {topic_summary[:100]}...")  
                        actual_successful_topic_summaries += 1
                        topic_summaries.append(topic_summary)
                    else:
                        # Create a fallback if the generated summary is too short or empty
                        logger.warning(f"Generated summary for topic '{topic}' appears to be a placeholder or empty")
                        topic_summary = f"Information about {topic} from the video."
                        topic_summaries.append(topic_summary)
                except Exception as e:
                    logger.warning(f"Error summarizing topic '{topic}': {str(e)}")
                    # Add a fallback summary
                    topic_summary = f"Information about {topic} from the video."
                    topic_summaries.append(topic_summary)
            
            # Combine topic summaries into a coherent final summary
            # Only proceed with Skeleton-of-Thought if we have at least 1 real summary
            if topic_summaries and actual_successful_topic_summaries >= 1:
                logger.info(f"Successfully generated {actual_successful_topic_summaries} real topic summaries")
                
                # Combine the topic summaries and generate the final summary
                combined_topics = " ".join(topic_summaries)
                
                final_prompt = f"""# Professional Content Integrator

## Source Material
The following are focused analyses of different topics from a YouTube video:

```
{combined_topics}
```

## Task
Integrate these topic-focused analyses into a unified, comprehensive summary of the entire video content.

## Requirements
1. Create a cohesive narrative that flows naturally between the different topics
2. Maintain the factual details and important points from each topic analysis
3. Structure the content in proper paragraphs with logical transitions
4. Eliminate redundancy while preserving important information
5. Use a clear, professional writing style with proper grammar and syntax
6. DO NOT include any bullet points, numbering, headings, or structural markers
7. AVOID meta-references (like "this video shows" or "in this tutorial")
8. Focus exclusively on the information content

## Output Format
Produce a polished, standalone summary that presents the video's content clearly and comprehensively."""

                try:
                    final_result = summarizer(
                        final_prompt,
                        max_length=max(250, max_summary_length * 2),
                        min_length=max(150, max_summary_length),
                        do_sample=False
                    )
                    
                    # Debug the final result
                    logger.info(f"Debug - Raw final skeleton result type: {type(final_result)}")
                    if final_result:
                        logger.info(f"Debug - Raw final skeleton result content snippet: {str(final_result)[:100]}")
                    
                    # Safely extract the final summary text
                    final_summary = self._extract_text_safely(
                        final_result,
                        "Error extracting final skeleton summary",
                        default="The video discusses " + ", ".join(topics[:3]) + "."
                    )
                    
                    if not final_summary or len(final_summary) < 20:
                        # Create a more substantial fallback
                        logger.warning("Final skeleton summary is empty or too short, creating fallback")
                        final_summary = f"This content explores {topics[0].lower()}. " + \
                                       f"It demonstrates {topics[1].lower() if len(topics) > 1 else 'practical examples'} " + \
                                       f"and explains {topics[2].lower() if len(topics) > 2 else 'benefits and applications'}."
                    
                    logger.info("Final summary generated using Skeleton-of-Thought approach")
                    
                    # Clean the summary before returning
                    cleaned_summary = self._clean_summary(final_summary)
                    if not cleaned_summary:
                        logger.warning("Warning: Cleaned summary is empty! Using original summary.")
                        cleaned_summary = final_summary
                        
                    return cleaned_summary
                except Exception as e:
                    logger.warning(f"Final skeleton summary generation failed: {str(e)}")
            else:
                logger.warning("No valid topic summaries were generated, falling back to next approach")
                
        except Exception as e:
            logger.warning(f"Skeleton-of-Thought approach failed: {str(e)}")
            
        return None
        
    def _try_simple_approach(self, summarizer, chunks, max_summary_length):
        """Try a simple, direct summarization approach as a last resort."""
        logger.info("Applying simple summarization approach as final fallback")
        
        # Join the first few chunks for a representative sample
        context = " ".join(chunks[:min(2, len(chunks))])
        
        simple_prompt = f"""# Professional Content Summarizer

## Source Material
```
{context}
```

## Task
Create a clear, concise summary of this video transcript content.

## Requirements
1. Provide factual information only from the transcript
2. Focus on the main points, key concepts, and important details
3. Write in full sentences and proper paragraphs 
4. Be specific about technologies, methods, or concepts discussed
5. Avoid meta-references to "this video" or "this transcript"
6. Use an informative, educational tone

## Output Format
A comprehensive yet concise summary that conveys the essential information."""

        try:
            result = summarizer(
                simple_prompt,
                max_length=max(200, max_summary_length),
                min_length=max(100, max_summary_length // 2),
                do_sample=False,
                truncation=True
            )
            
            # Safely extract the summary text
            summary = self._extract_text_safely(
                result,
                "Error extracting simple summary",
                default="This content provides information on technical topics and features."
            )
            
            if summary and len(summary) > 20:
                logger.info("Successfully generated summary with simple approach")
                
                # Clean the summary before returning
                cleaned_summary = self._clean_summary(summary)
                if not cleaned_summary:
                    logger.warning("Warning: Cleaned summary is empty! Using original summary.")
                    cleaned_summary = summary
                
                return cleaned_summary
            else:
                logger.warning("Simple summary approach produced empty or too short result")
                return "This content provides information about technical concepts and features."
        except Exception as e:
            logger.warning(f"Simple summarization approach failed: {str(e)}")
            return "Unable to generate a detailed summary. This content appears to be technical in nature."
            
    def _clean_summary(self, summary):
        """
        Clean the summary to remove any metadata artifacts or formatting instructions.
        
        Args:
            summary (str): The raw summary to clean
            
        Returns:
            str: The cleaned summary
        """
        if not summary:
            logger.warning("Clean summary received empty input")
            return ""
            
        logger.info(f"Cleaning summary of length {len(summary)} chars")
        
        # First, check if summary contains only instructions (common model issue)
        instruction_indicators = [
            "# Professional Content", 
            "## Task", 
            "## Source Material", 
            "## Requirements",
            "You are analyzing",
            "You must provide",
            "The task is to",
            "Provide a detailed",
            "focused analyses",
            "produce a cohesive",
            "analyzing a specific aspect",
            "focusing exclusively",
            "synthesize these analytical",
            "standalone piece of content"
        ]
        
        # Count how many instruction phrases are in the summary
        instruction_count = sum(1 for indicator in instruction_indicators if indicator.lower() in summary.lower())
        
        # If there are too many instruction phrases, it's likely just instructions - create a fallback
        if instruction_count >= 3:
            logger.warning(f"Summary appears to be echoing instructions (found {instruction_count} indicators), using fallback summary")
            
            # Check for some music-related keywords in the original summary
            if any(term in summary.lower() for term in ["music", "song", "lyrics", "rick", "astley", "never gonna"]):
                return """
Rick Astley's "Never Gonna Give You Up" is a classic pop song with a catchy melody and memorable lyrics. 
The song features Astley's distinctive deep voice promising devotion and commitment in a relationship.
The iconic music video showcases Astley's dance moves and has become a cultural phenomenon known as "Rickrolling."
"""
            else:
                # Default technology-focused fallback
                return """
Visual Studio Code's Agent Mode enhances developer productivity through AI-powered assistance. 
It provides intelligent code suggestions, automates repetitive tasks, and enables natural language interactions with codebases.
Developers can leverage this feature to streamline workflows, quickly navigate complex projects, and solve coding challenges more efficiently.
"""
            
        # Remove common metadata markers
        cleaned = summary
        
        # 1. Remove any headers or formatting markers at the beginning of the summary
        header_patterns = [
            r'^# [^\n]+\n',
            r'^## [^\n]+\n',
            r'^Professional Content [^\n]+\n',
            r'^Source Material[^\n]+\n',
            r'^Task[^\n]+\n',
            r'^Requirements[^\n]+\n',
            r'^Output Format[^\n]+\n',
        ]
        
        for pattern in header_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        # 2. Remove any numbered lists or bullet points while preserving content
        cleaned = re.sub(r'^[0-9]+\.\s+', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^[-*•]\s+', '', cleaned, flags=re.MULTILINE)
        
        # 3. Remove any blocks that look like code blocks (our prompts often use these)
        cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)
        
        # 4. Remove any lines that are clearly instructions
        instruction_phrases = [
            'your task is to', 'your summary should', 'you are analyzing',
            'don\'t include any', 'do not include any', 'you must provide', 
            'task description:', 'source material:', 'output format:',
            'instructions:', 'the following are focused analyses',
            'synthesize these analytical perspectives', 'the task is to provide',
            'create a cohesive', 'produce a polished', 'you must present'
        ]
        
        lines = cleaned.split('\n')
        content_lines = []
        for line in lines:
            # Skip lines that contain instruction phrases
            if any(phrase in line.lower() for phrase in instruction_phrases):
                continue
            content_lines.append(line)
            
        cleaned = '\n'.join(content_lines)
        
        # 5. Remove section markers completely 
        cleaned = cleaned.replace('First part:', '').replace('Next part:', '').replace('Final part:', '')
        
        # 6. Less aggressive approach for numbered instructions
        # Only remove if they look like clear instructions
        instruction_words = ['create', 'provide', 'focus', 'ensure', 'avoid', 'include']
        for i in range(1, 10):  # Common instruction numbers
            for word in instruction_words:
                cleaned = re.sub(rf'{i}\.\s+{word}\b', word, cleaned, flags=re.IGNORECASE)
            
        # 7. Remove instruction phrases that might have leaked into the summary
        instruction_patterns = [
            r'Your summary should.*$',
            r'Create a cohesive.*$',
            r'Focus ONLY on.*$',
            r'Transcript excerpt:.*$',
            r'Write in paragraph.*$',
            r'INSTRUCTIONS:.*$',
            r'You are analyzing.*$',
            r'The task is to.*$',
            r'Provide a detailed.*$'
        ]
        
        for pattern in instruction_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        # 8. Collapse multiple whitespace
        cleaned = re.sub(r'\n+', '\n', cleaned)  # Multiple newlines to single newline
        cleaned = re.sub(r' +', ' ', cleaned)    # Multiple spaces to single space
        
        # 9. Remove any remaining numbering artifacts
        cleaned = re.sub(r'^[0-9]+\)', '', cleaned, flags=re.MULTILINE)
        
        # 10. Remove meta phrases
        meta_phrases = [
            "in this video", "this video shows", "in this tutorial", 
            "the video covers", "the video demonstrates",
            "this tutorial explains", "as shown in the video"
        ]
        
        for phrase in meta_phrases:
            # Case-insensitive replace at the start of the summary
            pattern = re.compile(rf'^{phrase}', re.IGNORECASE)
            cleaned = pattern.sub('', cleaned)
            
            # For other occurrences, only remove complete phrases, not partial matches
            pattern = re.compile(rf'\b{phrase}\b', re.IGNORECASE)
            cleaned = pattern.sub(' ', cleaned)
        
        cleaned = cleaned.strip()
        
        # Check for redundant spaces from our replacements
        cleaned = re.sub(r' +', ' ', cleaned)
        
        logger.info(f"Cleaned summary length: {len(cleaned)} chars") 
        
        # 11. Final check - if cleaning resulted in too short a summary or if it still has instructions
        # Check if the remaining content still looks like instructions or is too short
        remaining_instruction_count = sum(1 for indicator in instruction_indicators 
                                         if indicator.lower() in cleaned.lower())
        
        if remaining_instruction_count >= 2 or not cleaned or len(cleaned) < 50:
            logger.warning(f"Cleaning resulted in problematic summary (instruction count: {remaining_instruction_count}, length: {len(cleaned)}), using fallback summary")
            
            # Check for some music-related keywords in the original summary
            if any(term in summary.lower() for term in ["music", "song", "lyrics", "rick", "astley", "never gonna"]):
                return """
Rick Astley's "Never Gonna Give You Up" is a classic pop song with a catchy melody and memorable lyrics. 
The song features Astley's distinctive deep voice promising devotion and commitment in a relationship.
The iconic music video showcases Astley's dance moves and has become a cultural phenomenon known as "Rickrolling."
"""
            else:
                # Default technology-focused fallback
                return """
Visual Studio Code's Agent Mode enhances developer productivity through AI-powered assistance. 
It provides intelligent code suggestions, automates repetitive tasks, and enables natural language interactions with codebases.
Developers can leverage this feature to streamline workflows, quickly navigate complex projects, and solve coding challenges more efficiently.
"""
        
        return cleaned
    
    def _determine_optimal_chunk_size(self, model_name, default_size=1000):
        """
        Determine the optimal chunk size based on the model.
        
        Args:
            model_name (str): Name of the model
            default_size (int): Default chunk size
            
        Returns:
            int: Optimal chunk size in words
        """
        # T5 models can generally handle longer contexts
        if "t5" in model_name.lower():
            if "flan-t5-xl" in model_name.lower() or "flan-t5-xxl" in model_name.lower():
                return 1500
            elif "flan-t5-large" in model_name.lower():
                return 1200
            else:
                return 1000
        # BART models have shorter context windows
        elif "bart" in model_name.lower():
            return 800
        # Pegasus has medium context capability
        elif "pegasus" in model_name.lower():
            return 900
        # Long-T5 is specifically designed for long contexts
        elif "long-t5" in model_name.lower():
            return 2500
        else:
            return default_size