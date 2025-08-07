"""
Main Content Extractor class
"""

import logging
import time
from typing import List, Optional
from pathlib import Path

from .models import Segment, ProcessingConfig, ProcessingResult
from .video_processing import VideoProcessor
from .transcription import WhisperTranscriber
from .segmentation import SegmentProcessor
from .embeddings import EmbeddingGenerator
from .evaluation import ContentEvaluator

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
        self.transcriber = WhisperTranscriber(self.config.whisper_model)
        self.segment_processor = SegmentProcessor(
            self.config.segment_duration,
            self.config.overlap_duration
        )
        self.embedding_generator = EmbeddingGenerator(self.config.embedding_model)
        self.evaluator = ContentEvaluator()
        
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
            # Step 1: Extract audio from video
            logger.info("Step 1: Extracting audio from video...")
            audio_path = self.video_processor.process_video_file(video_path)
            
            # Step 2: Transcribe audio
            logger.info("Step 2: Transcribing audio...")
            original_segments = self.transcriber.process_audio_file(audio_path)
            
            if not original_segments:
                raise ValueError("No segments extracted from video file")
            
            # Step 3: Create overlapping segments
            logger.info("Step 3: Creating overlapping segments...")
            processed_segments = self.segment_processor.process_segments(original_segments)
            
            if not processed_segments:
                raise ValueError("No segments after processing")
            
            # Step 4: Generate embeddings
            logger.info("Step 4: Generating embeddings...")
            segments_with_embeddings = self.embedding_generator.add_embeddings_to_segments(processed_segments)
            
            # Step 5: Evaluate content
            logger.info("Step 5: Evaluating content...")
            evaluated_segments = self.evaluator.evaluate_segments(segments_with_embeddings)
            
            # Step 6: Filter high-value segments
            logger.info("Step 6: Filtering high-value segments...")
            high_value_segments = self.evaluator.filter_high_value_segments(
                evaluated_segments, 
                self.config.min_score_threshold
            )
            
            # Calculate processing time and total duration
            processing_time = time.time() - start_time
            total_duration = original_segments[-1].end_time if original_segments else 0
            
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
        Export segments to CSV file
        
        Args:
            segments: List of segments to export
            output_path: Path to save CSV file
        """
        import pandas as pd
        
        data = []
        for i, segment in enumerate(segments):
            data.append({
                "index": i,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "duration": segment.duration(),
                "text": segment.text,
                "confidence": segment.confidence,
                "value_score": segment.value_score,
                "reasoning": segment.reasoning
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Segments exported to CSV: {output_path}") 