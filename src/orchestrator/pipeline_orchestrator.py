"""
Main pipeline orchestrator that coordinates all stages
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
import os
from datetime import datetime

import sys
from pathlib import Path

# Add src directory to path for absolute imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from orchestrator.performance_monitor import PerformanceMonitor
from shared.models import ProcessingConfig
from shared.exceptions import PipelineException, StageException

# Import all stage classes
from stages._01_audio_extraction.code import AudioExtractionStage
from stages._04_speaker_segmentation.code.stage_wrapper import SpeakerSegmentationStage  
from stages._02_transcription.code import TranscriptionStage
from stages._03_content_segmentation.code import ContentSegmentationStage
from stages._05_content_evaluation.code import ContentEvaluationStage
from stages._06_output_generation.code import OutputGenerationStage

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Main orchestrator that manages the entire processing pipeline
    """
    
    def __init__(self, config: ProcessingConfig):
        """
        Initialize the orchestrator
        
        Args:
            config: Processing configuration
        """
        self.config = config
        self.performance_monitor = PerformanceMonitor()
        self.stages = {}
        self.results = {}
        
        logger.info(f"Initialized PipelineOrchestrator with profile: {config.processing_profile}")
        
        # Initialize stage output directory if needed
        self.stage_output_path = None
        if self.config.save_stage_outputs:
            self.stage_output_path = self._create_stage_output_directory()
        
        # Initialize stages based on configuration
        self._initialize_stages()
    
    def _initialize_stages(self) -> None:
        """Initialize all pipeline stages based on configuration"""
        
        # Stage 1: Audio Extraction (always required)
        self.stages['audio_extraction'] = AudioExtractionStage(self.config)
        
        # Stage 2: Speaker Segmentation (optional)
        if self.config.enable_speaker_detection:
            self.stages['speaker_segmentation'] = SpeakerSegmentationStage(self.config)
        
        # Stage 3: Transcription (always required)
        self.stages['transcription'] = TranscriptionStage(self.config)
        
        # Stage 4: Content Segmentation (always required)
        self.stages['content_segmentation'] = ContentSegmentationStage(self.config)
        
        # Stage 5: Content Evaluation (optional based on config)
        if self.config.enable_content_evaluation or self.config.use_rule_based_scoring:
            self.stages['content_evaluation'] = ContentEvaluationStage(self.config)
        
        # Stage 6: Output Generation (always required)
        self.stages['output_generation'] = OutputGenerationStage(self.config)
        
        logger.info(f"Initialized {len(self.stages)} pipeline stages")
    
    def _create_stage_output_directory(self) -> Path:
        """Create timestamped directory for stage outputs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(self.config.stage_output_dir) / f"run_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created stage output directory: {output_path}")
        return output_path
    
    def _save_stage_output(self, stage_name: str, stage_data: Dict[str, Any], stage_number: int) -> None:
        """
        Save stage output to individual file
        
        Args:
            stage_name: Name of the stage
            stage_data: Data from the stage
            stage_number: Stage number for file naming
        """
        if not self.stage_output_path:
            return
            
        # Create filename with stage number and name
        filename = f"{stage_number:02d}_{stage_name}.json"
        filepath = self.stage_output_path / filename
        
        try:
            # Convert data to JSON-serializable format
            serializable_data = self._make_json_serializable(stage_data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {stage_name} output to: {filepath}")
        except Exception as e:
            logger.error(f"Failed to save {stage_name} output: {e}")
    
    def _make_json_serializable(self, data: Any) -> Any:
        """Convert data to JSON-serializable format"""
        if hasattr(data, 'to_dict'):
            return data.to_dict()
        elif isinstance(data, dict):
            return {k: self._make_json_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_json_serializable(item) for item in data]
        elif hasattr(data, '__dict__'):
            return {k: self._make_json_serializable(v) for k, v in data.__dict__.items()}
        else:
            # For basic types (str, int, float, bool, None)
            try:
                json.dumps(data)
                return data
            except (TypeError, ValueError):
                return str(data)
    
    def process_video(self, video_path: str) -> Dict[str, Any]:
        """
        Process a video through the entire pipeline
        
        Args:
            video_path: Path to video file
            
        Returns:
            Complete processing results
        """
        try:
            self.performance_monitor.start_pipeline()
            logger.info(f"Starting pipeline processing for: {video_path}")
            
            # Stage 1: Audio Extraction
            logger.info("=== STAGE 1: Audio Extraction ===")
            pipeline_data = self.stages['audio_extraction'].run(video_path)
            self.performance_monitor.add_stage_metrics(self.stages['audio_extraction'].get_metrics())
            self.results['audio_extraction'] = pipeline_data
            self._save_stage_output('audio_extraction', pipeline_data, 1)
            
            # Stage 2: Speaker Segmentation (if enabled)
            if 'speaker_segmentation' in self.stages:
                logger.info("=== STAGE 2: Speaker Segmentation ===")
                speaker_data = self.stages['speaker_segmentation'].run(pipeline_data)
                self.performance_monitor.add_stage_metrics(self.stages['speaker_segmentation'].get_metrics())
                self.results['speaker_segmentation'] = speaker_data
                self._save_stage_output('speaker_segmentation', speaker_data, 2)
                # Update pipeline data
                pipeline_data.update(speaker_data)
            
            # Stage 3: Transcription
            logger.info("=== STAGE 3: Transcription ===")
            transcription_data = self.stages['transcription'].run(pipeline_data)
            self.performance_monitor.add_stage_metrics(self.stages['transcription'].get_metrics())
            self.results['transcription'] = transcription_data
            self._save_stage_output('transcription', transcription_data, 3)
            # Update pipeline data
            pipeline_data.update(transcription_data)
            
            # Stage 4: Content Segmentation
            logger.info("=== STAGE 4: Content Segmentation ===")
            segmentation_data = self.stages['content_segmentation'].run(pipeline_data)
            self.performance_monitor.add_stage_metrics(self.stages['content_segmentation'].get_metrics())
            self.results['content_segmentation'] = segmentation_data
            self._save_stage_output('content_segmentation', segmentation_data, 4)
            # Update pipeline data
            pipeline_data.update(segmentation_data)
            
            # Stage 5: Content Evaluation (if enabled)
            if 'content_evaluation' in self.stages:
                logger.info("=== STAGE 5: Content Evaluation ===")
                evaluation_data = self.stages['content_evaluation'].run(pipeline_data)
                self.performance_monitor.add_stage_metrics(self.stages['content_evaluation'].get_metrics())
                self.results['content_evaluation'] = evaluation_data
                self._save_stage_output('content_evaluation', evaluation_data, 5)
                # Update pipeline data
                pipeline_data.update(evaluation_data)
            
            # Stage 6: Output Generation
            logger.info("=== STAGE 6: Output Generation ===")
            output_data = self.stages['output_generation'].run(pipeline_data)
            self.performance_monitor.add_stage_metrics(self.stages['output_generation'].get_metrics())
            self.results['output_generation'] = output_data
            self._save_stage_output('output_generation', output_data, 6)
            
            # Finalize performance monitoring
            total_duration = pipeline_data.get('duration', 0)
            pipeline_metrics = self.performance_monitor.finish_pipeline(total_duration)
            
            # Create final results
            final_results = {
                'config': self.config,
                'pipeline_metrics': pipeline_metrics,
                'stage_results': self.results,
                'video_path': video_path,
                'success': True
            }
            
            logger.info("✅ Pipeline processing completed successfully")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline processing failed: {str(e)}")
            
            # Record failure metrics
            pipeline_metrics = self.performance_monitor.finish_pipeline()
            
            # Create failure results
            failure_results = {
                'config': self.config,
                'pipeline_metrics': pipeline_metrics,
                'stage_results': self.results,
                'video_path': video_path,
                'success': False,
                'error': str(e)
            }
            
            raise PipelineException(f"Pipeline processing failed: {str(e)}") from e
    
    def generate_performance_report(self) -> str:
        """
        Generate a detailed performance report
        
        Returns:
            Performance report string
        """
        return self.performance_monitor.generate_performance_report()
    
    def save_results(self, results: Dict[str, Any], output_path: Optional[str] = None) -> None:
        """
        Save processing results to files
        
        Args:
            results: Processing results to save
            output_path: Optional output path (uses config default if None)
        """
        if output_path is None:
            output_path = getattr(self.config, 'output_dir', 'results')
        
        # Ensure output directory exists
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save performance report
        report_path = output_dir / 'performance_report.txt'
        with open(report_path, 'w') as f:
            f.write(self.generate_performance_report())
        
        logger.info(f"Results saved to: {output_dir}")
    
    def get_stage_info(self, stage_name: str) -> Dict[str, Any]:
        """
        Get information about a specific stage
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Stage information
        """
        if stage_name not in self.stages:
            return {'error': f'Stage not found: {stage_name}'}
        
        return self.stages[stage_name].get_stage_info()
    
    def list_stages(self) -> Dict[str, str]:
        """
        List all available stages
        
        Returns:
            Dictionary of stage names and their status
        """
        return {
            name: 'initialized' for name in self.stages.keys()
        }