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
        
        # Transcription model configurations  
        parser.add_argument('--transcription-model', '--whisper-model', dest='transcription_model',
                          default='auto',
                          help='Transcription model: auto, tiny, base, small, medium, large, large-v3, large-v3-turbo, or Hebrew models: ivrit-v2-d4, ivrit-v2-d3-e3')
        parser.add_argument('--force-model', action='store_true',
                          help='Force specified model, disable automatic selection based on duration')
        parser.add_argument('--list-models', action='store_true',
                          help='List all available transcription models and exit')
        
        # LLM evaluation model
        parser.add_argument('--evaluation-model', default='microsoft/Phi-3-mini-4k-instruct',
                          help='LLM model for content evaluation')
        
        # Segmentation settings
        parser.add_argument('--segment-duration', type=int, default=90,
                          help='Segment duration in seconds (default: 90)')
        parser.add_argument('--overlap-duration', type=int, default=20,
                          help='Overlap duration in seconds (default: 20)')
        
        # Performance settings
        parser.add_argument('--batch-size', type=int, default=5,
                          help='Batch size for evaluation')
        parser.add_argument('--keep-audio', action='store_true',
                          help='Keep extracted audio files')
        parser.add_argument('--force-cpu', action='store_true',
                          help='Force CPU processing, disable MPS/CUDA acceleration (fixes MPS backend issues on M1 Mac)')
        parser.add_argument('--save-stage-outputs', action='store_true',
                          help='Save intermediate outputs from each pipeline stage to separate files')
        parser.add_argument('--stage-output-dir', default='stage_outputs',
                          help='Directory name for stage outputs (default: stage_outputs)')
        
        # Language settings
        parser.add_argument('--language', default='he',
                          help='Primary language code')
        parser.add_argument('--technical-language', default='en',
                          help='Technical terms language')
        
        args = parser.parse_args()
        
        # Handle --list-models flag
        if getattr(args, 'list_models', False):
            ConfigManager._list_available_models()
            import sys
            sys.exit(0)
        
        # Create config based on profile and arguments
        if args.profile:
            config = ProcessingConfig.create_profile(args.profile)
        else:
            config = ProcessingConfig()
        
        # Override with command line arguments
        config.transcription_model = args.transcription_model
        config.force_transcription_model = getattr(args, 'force_model', False)
        config.evaluation_model = args.evaluation_model
        
        # Backward compatibility with whisper_model
        config.whisper_model = args.transcription_model
        
        # Segmentation settings
        config.segment_duration = args.segment_duration
        config.overlap_duration = args.overlap_duration
        
        config.evaluation_batch_size = args.batch_size
        config.keep_audio = args.keep_audio
        config.force_cpu = getattr(args, 'force_cpu', False)
        config.save_stage_outputs = getattr(args, 'save_stage_outputs', False)
        config.stage_output_dir = getattr(args, 'stage_output_dir', 'stage_outputs')
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
    
    @staticmethod
    def _list_available_models() -> None:
        """List all available transcription models"""
        print("🎙️  Available Transcription Models:")
        print("=" * 50)
        
        print("\n📌 Standard OpenAI Whisper Models:")
        standard_models = [
            ("tiny", "Fastest, lowest accuracy, ~2GB VRAM"),
            ("base", "Good balance, ~4GB VRAM"), 
            ("small", "Better accuracy, ~6GB VRAM"),
            ("medium", "High accuracy, ~8GB VRAM"),
            ("large", "Highest accuracy, ~12GB VRAM"),
            ("large-v3", "Latest Whisper v3, ~12GB VRAM"),
            ("large-v3-turbo", "5.4x faster than v2, ~12GB VRAM")
        ]
        
        for model, desc in standard_models:
            print(f"  • {model:<15} - {desc}")
        
        print("\n🇮🇱 Hebrew-Optimized Models (Ivrit.AI):")
        hebrew_models = [
            ("ivrit-v2-d4", "Latest Hebrew-tuned Whisper v2 Large (Recommended for Hebrew)"),
            ("ivrit-v2-d3-e3", "Alternative Hebrew-tuned Whisper v2 Large")
        ]
        
        for model, desc in hebrew_models:
            print(f"  • {model:<15} - {desc}")
        
        print("\n🤖 Automatic Selection:")
        print(f"  • {'auto':<15} - Smart selection based on video duration (default)")
        
        print(f"\n💡 Usage Examples:")
        print(f"  # Use Hebrew-optimized model")
        print(f"  python -m src video.mp4 --transcription-model ivrit-v2-d4")
        print(f"  # Force large model regardless of duration")
        print(f"  python -m src video.mp4 --transcription-model large --force-model")
        print(f"  # Use latest Whisper turbo model")
        print(f"  python -m src video.mp4 --transcription-model large-v3-turbo")
        
        print(f"\n⚠️  Note: Hebrew models require additional setup. See documentation for details.")