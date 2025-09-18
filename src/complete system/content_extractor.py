"""
Main Content Extractor class
"""

import logging
import time
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm

from .models import Segment, ProcessingConfig, ProcessingResult
from .stages._01_audio_extraction.code.video_processing import VideoProcessor
from .stages._02_transcription.code.transcription import WhisperTranscriber
from .stages._03_content_segmentation.code.segmentation import SegmentProcessor
from .stages._05_content_evaluation.code.embeddings import EmbeddingGenerator
from .stages._05_content_evaluation.code.evaluation import ContentEvaluator
from .stages._04_speaker_segmentation.code.speaker_analysis import SpeakerDiarizer
from .stages._02_transcription.code.language_processor import LanguageProcessor

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
        # Use transcription_model if available, fallback to whisper_model for compatibility
        transcription_model = getattr(self.config, 'transcription_model', self.config.whisper_model)
        self.transcriber = WhisperTranscriber(
            model_name=transcription_model,
            primary_language=self.config.primary_language,
            smart_model_selection=not getattr(self.config, 'force_transcription_model', False),
            force_model=getattr(self.config, 'force_transcription_model', False),
            force_cpu=getattr(self.config, 'force_cpu', False)
        )
        self.segment_processor = SegmentProcessor(
            self.config.segment_duration,
            self.config.overlap_duration
        )
        self.embedding_generator = EmbeddingGenerator(self.config.embedding_model)
        self.evaluator = ContentEvaluator(
            model_name=self.config.evaluation_model,
            batch_size=self.config.evaluation_batch_size,
            use_rule_based=self.config.use_rule_based_scoring,
            enable_evaluation=self.config.enable_content_evaluation
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
            # Performance monitoring
            step_times = {}
            
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
            
            # Step 4: Process segments (create proper Reels-length segments)
            progress_bar.set_description("Creating overlapping segments for Reels")
            logger.info(f"Step {4 if speaker_analysis else 3}: Creating overlapping segments for Reels content...")
            
            # Use proper segmentation for Reels (15-45s segments)
            step_start = time.time()
            processed_segments = self.segment_processor.process_segments(
                original_segments, 
                min_duration=15.0  # Minimum 15 seconds for Reels
            )
            
            logger.info(f"🎯 Created {len(processed_segments)} Reels-ready segments (15-45s duration)")
            step_times['segmentation'] = time.time() - step_start
            progress_bar.update(1)
            
            if not processed_segments:
                raise ValueError("No segments after processing")
            
            # Step 5: Speaker filtering (if enabled)
            if self.config.enable_speaker_detection and speaker_analysis and self.config.primary_speaker_only:
                step_start = time.time()
                progress_bar.set_description("Filtering by primary speaker")
                logger.info("Step 5: Filtering segments by primary speaker...")
                processed_segments = self.speaker_diarizer.filter_segments_by_speaker(
                    processed_segments, speaker_analysis, primary_only=True
                )
                step_times['speaker_filtering'] = time.time() - step_start
                progress_bar.update(1)
            else:
                step_times['speaker_filtering'] = 0.0
            
            # Step 6: Language processing (if enabled)
            if self.config.enable_technical_terms and self.config.preserve_technical_terms and self.language_processor:
                step_start = time.time()
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
                step_times['language_processing'] = time.time() - step_start
                progress_bar.update(1)
            else:
                step_times['language_processing'] = 0.0
            
            # Step 7: Generate embeddings (if needed)
            if self.config.enable_similarity_analysis or not self.config.minimal_mode:
                step_start = time.time()
                progress_bar.set_description("Generating embeddings")
                logger.info(f"Step {7 if speaker_analysis else 5}: Generating embeddings...")
                segments_with_embeddings = self.embedding_generator.add_embeddings_to_segments(
                    processed_segments,
                    batch_size=self.config.embedding_batch_size,
                )
                step_times['embeddings'] = time.time() - step_start
                progress_bar.update(1)
            else:
                step_times['embeddings'] = 0.0
                logger.info("⚡ Skipping embedding generation (minimal mode)")
                segments_with_embeddings = processed_segments
                progress_bar.update(1)
            
            # Step 8: Evaluate content
            step_start = time.time()
            progress_bar.set_description(f"Evaluating {len(segments_with_embeddings)} segments (this may take time)")
            logger.info(f"Step {8 if speaker_analysis else 6}: Evaluating content...")
            
            # Enhanced evaluation with language context for Hebrew educational content
            evaluated_segments = self.evaluator.evaluate_segments(segments_with_embeddings)
            step_times['evaluation'] = time.time() - step_start
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
            
            # Log detailed performance metrics
            if total_duration > 0:
                realtime_factor = total_duration / processing_time
                logger.info(f"✅ Processing completed in {processing_time:.2f}s")
                logger.info(f"🎵 Audio duration: {total_duration:.1f}s")
                logger.info(f"🚀 Processing speed: {realtime_factor:.1f}x realtime")
                
                # Log step-by-step performance
                logger.info("📊 Step-by-step performance:")
                for step_name, step_time in step_times.items():
                    percentage = (step_time / processing_time) * 100
                    logger.info(f"   {step_name}: {step_time:.2f}s ({percentage:.1f}%)")
                
                # Identify bottlenecks (only if there are meaningful step times)
                if step_times:
                    # Find the actual bottleneck (excluding steps with 0 time)
                    meaningful_steps = [(name, time) for name, time in step_times.items() if time > 0.1]
                    
                    if meaningful_steps:
                        bottleneck_step = max(meaningful_steps, key=lambda x: x[1])
                        logger.info(f"🔍 Main bottleneck: {bottleneck_step[0]} ({bottleneck_step[1]:.2f}s)")
                        
                        # Performance recommendations
                        if bottleneck_step[1] > processing_time * 0.4:  # More than 40% of total time
                            self._log_performance_recommendations(bottleneck_step[0], self.config)
                    else:
                        logger.info("⚡ All processing steps completed very quickly - no significant bottlenecks")
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
            
            # Add performance metrics to result
            if hasattr(result, '__dict__'):
                result.step_times = step_times
                result.bottleneck = max(step_times.items(), key=lambda x: x[1])[0] if step_times else None
            
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
    
    def _log_performance_recommendations(self, bottleneck_step: str, config) -> None:
        """
        Log performance optimization recommendations based on bottleneck
        
        Args:
            bottleneck_step: Name of the bottleneck step
            config: Processing configuration
        """
        logger.info("💡 Performance optimization recommendations:")
        
        if bottleneck_step == 'transcription':
            if config.whisper_model != 'tiny':
                logger.info("   - Use --profile draft for 80% faster processing")
                logger.info("   - Try smaller Whisper model: --whisper-model tiny")
            logger.info("   - Ensure GPU acceleration is working (MPS/CUDA)")
            
        elif bottleneck_step == 'evaluation':
            if config.enable_content_evaluation:
                logger.info("   - Use --profile fast for rule-based evaluation")
                logger.info("   - Use --profile draft to disable LLM evaluation entirely")
                logger.info(f"   - Increase evaluation batch size: --evaluation-batch-size {config.evaluation_batch_size * 2}")
            
        elif bottleneck_step == 'speaker_analysis':
            logger.info("   - Disable speaker detection for speed: remove --enable-speaker-detection")
            logger.info("   - Use frequency-only analysis in draft mode")
            
        elif bottleneck_step == 'embeddings':
            logger.info("   - Use --minimal-mode to skip embeddings")
            logger.info("   - Disable similarity analysis: remove --enable-similarity")
            logger.info(f"   - Increase embedding batch size: --embedding-batch-size {config.embedding_batch_size * 2}")
        
        logger.info("   - For maximum speed: python -m src video.mp4 --profile draft")
    
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
                "duration": f"{segment.duration():.1f}s",
                "text": segment.text,
                "confidence": f"{segment.confidence:.3f}",
                "value_score": f"{segment.value_score:.2f}" if segment.value_score else "",
                "reasoning": segment.reasoning or ""
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Segments exported to CSV: {output_path}") 