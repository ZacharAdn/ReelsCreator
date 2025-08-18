"""
Configuration management for the pipeline
"""

import argparse
from typing import Dict, Any
from pathlib import Path

import sys
from pathlib import Path

# Add src directory to path for absolute imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from shared.models import ProcessingConfig


class ConfigManager:
    """
    Manages configuration for the entire pipeline
    """
    
    @staticmethod
    def load_from_args() -> ProcessingConfig:
        """
        Load configuration from command line arguments
        
        Returns:
            ProcessingConfig object
        """
        parser = argparse.ArgumentParser(description='Reels Content Extractor')
        
        # Basic arguments
        parser.add_argument('video_path', help='Path to video file to process')
        parser.add_argument('--output', '-o', help='Output directory for results')
        
        # Processing profile
        parser.add_argument('--profile', choices=['draft', 'fast', 'balanced', 'quality'], 
                          default='balanced', help='Processing profile')
        
        # Stage controls
        parser.add_argument('--enable-speaker-detection', action='store_true', 
                          help='Enable speaker detection')
        parser.add_argument('--enable-content-evaluation', action='store_true',
                          help='Enable content evaluation')
        parser.add_argument('--minimal-mode', action='store_true',
                          help='Skip embeddings for faster processing')
        
        # Model configurations  
        parser.add_argument('--whisper-model', default='base',
                          help='Whisper model size (tiny, base, small, medium, large)')
        parser.add_argument('--evaluation-model', default='microsoft/Phi-3-mini-4k-instruct',
                          help='LLM model for content evaluation')
        
        # Performance settings
        parser.add_argument('--batch-size', type=int, default=5,
                          help='Batch size for evaluation')
        parser.add_argument('--keep-audio', action='store_true',
                          help='Keep extracted audio files')
        
        # Language settings
        parser.add_argument('--language', default='he',
                          help='Primary language code')
        parser.add_argument('--technical-language', default='en',
                          help='Technical terms language')
        
        args = parser.parse_args()
        
        # Create config based on profile and arguments
        if args.profile:
            config = ProcessingConfig.create_profile(args.profile)
        else:
            config = ProcessingConfig()
        
        # Override with command line arguments
        config.whisper_model = args.whisper_model
        config.evaluation_model = args.evaluation_model
        config.evaluation_batch_size = args.batch_size
        config.keep_audio = args.keep_audio
        config.primary_language = args.language
        config.technical_language = args.technical_language
        
        # Stage controls
        if args.enable_speaker_detection:
            config.enable_speaker_detection = True
        if args.enable_content_evaluation:
            config.enable_content_evaluation = True
        if args.minimal_mode:
            config.minimal_mode = True
        
        # Add extra attributes
        config.video_path = args.video_path
        config.output_dir = args.output or 'results'
        
        return config
    
    @staticmethod
    def load_from_file(config_path: str) -> ProcessingConfig:
        """
        Load configuration from JSON file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ProcessingConfig object
        """
        # TODO: Implement JSON config loading
        raise NotImplementedError("JSON config loading not yet implemented")
    
    @staticmethod
    def save_to_file(config: ProcessingConfig, output_path: str) -> None:
        """
        Save configuration to JSON file
        
        Args:
            config: Configuration to save
            output_path: Path to save configuration
        """
        # TODO: Implement JSON config saving  
        raise NotImplementedError("JSON config saving not yet implemented")