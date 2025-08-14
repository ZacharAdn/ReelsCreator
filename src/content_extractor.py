"""
Main Content Extractor class
"""

import logging
import time
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm

from .models import Segment, ProcessingConfig, ProcessingResult
from .video_processing import VideoProcessor
from .transcription import WhisperTranscriber
from .segmentation import SegmentProcessor
from .embeddings import EmbeddingGenerator
from .evaluation import ContentEvaluator
from .speaker_analysis import SpeakerDiarizer
from .language_processor import LanguageProcessor

logger = logging.getLogger(__name__)


class ContentExtractor:
    """Main class for extracting valuable content from educational recordings"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize Content Extractor
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        
        # Initialize components
        self.video_processor = VideoProcessor()
        self.transcriber = WhisperTranscriber(
            self.config.whisper_model, 
            self.config.primary_language
        )
        self.segment_processor = SegmentProcessor(
            self.config.segment_duration,
            self.config.overlap_duration
        )
        self.embedding_generator = EmbeddingGenerator(self.config.embedding_model)
        self.evaluator = ContentEvaluator(
            model_name=self.config.evaluation_model,
            batch_size=self.config.evaluation_batch_size
        )
        
        # Speaker and language processing
        self.speaker_diarizer = SpeakerDiarizer(self.config.speaker_batch_size) if self.config.enable_speaker_detection else None
        self.language_processor = LanguageProcessor(
            self.config.primary_language, 
            self.config.technical_language
        ) if self.config.preserve_technical_terms else None
        
        logger.info("Content Extractor initialized")
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None) -> ProcessingResult:
        """
        Complete processing pipeline for a video file
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save results
            
        Returns:
            ProcessingResult with all segments and analysis
        """
        start_time = time.time()
        
        logger.info(f"Starting processing of: {video_path}")
        
        try:
            # Progress tracking
            total_steps = 8 if self.config.enable_speaker_detection else 6
            progress_bar = tqdm(total=total_steps, desc="Processing video", unit="step")
            
            # Step 1: Extract audio from video
            progress_bar.set_description("Extracting audio from video")
            logger.info("Step 1: Extracting audio from video...")
            audio_path = self.video_processor.process_video_file(
                video_path,
                keep_audio=self.config.keep_audio,
            )
            progress_bar.update(1)
            
            # Step 2: Speaker diarization (if enabled)
            speaker_analysis = None
            if self.config.enable_speaker_detection and self.speaker_diarizer:
                progress_bar.set_description("Analyzing speakers (this may take time)")
                logger.info("Step 2: Analyzing speakers...")
                speaker_analysis = self.speaker_diarizer.analyze_speakers(audio_path)
                progress_bar.update(1)
            
            # Step 3: Transcribe audio
            progress_bar.set_description("Transcribing audio (this may take several minutes)")
            logger.info(f"Step {3 if speaker_analysis else 2}: Transcribing audio...")
            original_segments = self.transcriber.process_audio_file(audio_path)
            progress_bar.update(1)
            
            if not original_segments:
                raise ValueError("No segments extracted from video file")
            
            # Step 4: Process segments (simplified approach)
            progress_bar.set_description("Processing segments (simplified)")
            logger.info(f"Step {4 if speaker_analysis else 3}: Processing segments (direct from Whisper)...")
            
            # Use direct Whisper segments without overlapping
            # Filter by minimum duration (keep segments >= 2 seconds for natural speech)
            min_duration = 2.0
            processed_segments = [
                seg for seg in original_segments 
                if (seg.end_time - seg.start_time) >= min_duration
            ]
            
            logger.info(f"🎯 Using {len(processed_segments)} segments directly from Whisper (filtered by {min_duration}s duration)")
            progress_bar.update(1)
            
            if not processed_segments:
                raise ValueError("No segments after processing")
            
            # Step 5: Speaker filtering (if enabled)
            if self.config.enable_speaker_detection and speaker_analysis and self.config.primary_speaker_only:
                progress_bar.set_description("Filtering by primary speaker")
                logger.info("Step 5: Filtering segments by primary speaker...")
                processed_segments = self.speaker_diarizer.filter_segments_by_speaker(
                    processed_segments, speaker_analysis, primary_only=True
                )
                progress_bar.update(1)
            
            # Step 6: Language processing (if enabled)
            if self.config.enable_technical_terms and self.config.preserve_technical_terms and self.language_processor:
                progress_bar.set_description("Processing language and technical terms")
                logger.info(f"Step {6 if speaker_analysis else 5}: Processing multilingual content...")
                technical_term_count = 0
                for segment in processed_segments:
                    technical_terms = self.language_processor.extract_technical_terms(segment.text)
                    if technical_terms:
                        technical_term_count += len(technical_terms)
                        if hasattr(segment, 'technical_terms'):
                            segment.technical_terms = technical_terms
                
                logger.info(f"🎯 Found {technical_term_count} technical terms across {len(processed_segments)} segments")
                progress_bar.update(1)
            
            # Step 7: Generate embeddings (if needed)
            if self.config.enable_similarity_analysis or not self.config.minimal_mode:
                progress_bar.set_description("Generating embeddings")
                logger.info(f"Step {7 if speaker_analysis else 5}: Generating embeddings...")
                segments_with_embeddings = self.embedding_generator.add_embeddings_to_segments(
                    processed_segments,
                    batch_size=self.config.embedding_batch_size,
                )
                progress_bar.update(1)
            else:
                logger.info("⚡ Skipping embedding generation (minimal mode)")
                segments_with_embeddings = processed_segments
                progress_bar.update(1)
            
            # Step 8: Evaluate content
            progress_bar.set_description(f"Evaluating {len(segments_with_embeddings)} segments (this may take time)")
            logger.info(f"Step {8 if speaker_analysis else 6}: Evaluating content...")
            
            # Enhanced evaluation with language context for Hebrew educational content
            evaluated_segments = self.evaluator.evaluate_segments(segments_with_embeddings)
            progress_bar.update(1)
            
            # Final step: Filter high-value segments
            progress_bar.set_description("Filtering high-value segments")
            logger.info("Final step: Filtering high-value segments...")
            high_value_segments = self.evaluator.filter_high_value_segments(
                evaluated_segments, 
                self.config.min_score_threshold
            )
            progress_bar.set_description("Processing complete!")
            progress_bar.close()
            
            # Calculate processing time and total duration
            processing_time = time.time() - start_time
            total_duration = original_segments[-1].end_time if original_segments else 0
            
            # Log processing performance
            if total_duration > 0:
                realtime_factor = total_duration / processing_time
                logger.info(f"✅ Processing completed in {processing_time:.2f}s")
                logger.info(f"🎵 Audio duration: {total_duration:.1f}s")
                logger.info(f"🚀 Processing speed: {realtime_factor:.1f}x realtime")
            else:
                logger.info(f"✅ Processing completed in {processing_time:.2f}s")
            
            # Create result
            result = ProcessingResult(
                segments=evaluated_segments,
                config=self.config,
                processing_time=processing_time,
                total_duration=total_duration,
                high_value_segments=high_value_segments
            )
            
            # Save results if output path provided
            if output_path:
                result.save_to_file(output_path)
                logger.info(f"Results saved to: {output_path}")
            
            # Log summary
            summary = result.to_dict()["summary"]
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            logger.info(f"Total segments: {summary['total_segments']}")
            logger.info(f"High-value segments: {summary['high_value_count']}")
            logger.info(f"Average score: {summary['average_score']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def process_multiple_files(self, video_files: List[str], output_dir: Optional[str] = None) -> List[ProcessingResult]:
        """
        Process multiple video files
        
        Args:
            video_files: List of video file paths
            output_dir: Optional directory to save results
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        for i, video_file in enumerate(video_files):
            logger.info(f"Processing file {i+1}/{len(video_files)}: {video_file}")
            
            output_path = None
            if output_dir:
                output_path = Path(output_dir) / f"{Path(video_file).stem}_results.json"
            
            try:
                result = self.process_video_file(video_file, output_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {video_file}: {e}")
                continue
        
        return results
    
    def get_similar_segments(self, segments: List[Segment], threshold: float = 0.7) -> List[List[int]]:
        """
        Find groups of similar segments
        
        Args:
            segments: List of segments with embeddings
            threshold: Similarity threshold
            
        Returns:
            List of segment group indices
        """
        return self.embedding_generator.find_similar_segments(segments, threshold)
    
    def get_evaluation_summary(self, segments: List[Segment]) -> dict:
        """
        Get summary of evaluation results
        
        Args:
            segments: List of evaluated segments
            
        Returns:
            Summary statistics
        """
        return self.evaluator.get_evaluation_summary(segments)
    
    def export_segments_to_csv(self, segments: List[Segment], output_path: str) -> None:
        """
        Export segments to CSV file with improved time formatting
        
        Args:
            segments: List of segments to export
            output_path: Path to save CSV file
        """
        import pandas as pd
        
        def format_time(seconds):
            """Convert seconds to MM:SS format"""
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}:{secs:02d}"
        
        data = []
        for i, segment in enumerate(segments):
            data.append({
                "index": i,
                "start_time": format_time(segment.start_time),
                "end_time": format_time(segment.end_time), 
                "start_seconds": segment.start_time,  # Keep raw seconds for reference
                "end_seconds": segment.end_time,
                "duration": f"{segment.duration():.1f}s",
                "text": segment.text,
                "confidence": f"{segment.confidence:.3f}",
                "value_score": f"{segment.value_score:.2f}" if segment.value_score else "",
                "reasoning": segment.reasoning or ""
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Segments exported to CSV: {output_path}") 