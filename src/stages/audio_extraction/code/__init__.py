"""
Audio Extraction Stage - Extract audio from video files
"""

import logging
from typing import Dict, Any
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.base_stage import BaseStage
from shared.exceptions import StageException
from .video_processing import VideoProcessor

logger = logging.getLogger(__name__)


class AudioExtractionStage(BaseStage):
    """
    Stage 1: Extract audio from video files
    
    Input: video_path (str)
    Output: {
        'audio_path': str,
        'duration': float,
        'sample_rate': int,
        'video_info': dict
    }
    """
    
    def __init__(self, config):
        super().__init__(config, "AudioExtraction")
        self.processor = VideoProcessor()
    
    def validate_input(self, video_path: str) -> bool:
        """Validate video file exists and is supported format"""
        super().validate_input(video_path)
        
        if not isinstance(video_path, str):
            raise StageException(self.stage_name, "Input must be a string path")
        
        if not Path(video_path).exists():
            raise StageException(self.stage_name, f"Video file not found: {video_path}")
        
        if not self.processor.is_supported_format(video_path):
            raise StageException(self.stage_name, f"Unsupported video format: {video_path}")
        
        return True
    
    def execute(self, video_path: str) -> Dict[str, Any]:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with audio information
        """
        try:
            # Get video info first
            video_info = self.processor.get_video_info(video_path)
            
            # Extract audio
            audio_path = self.processor.process_video_file(
                video_path, 
                keep_audio=getattr(self.config, 'keep_audio', False)
            )
            
            return {
                'audio_path': audio_path,
                'duration': video_info['duration'],
                'sample_rate': video_info.get('audio_fps', 16000),
                'video_info': video_info,
                'original_video_path': video_path
            }
            
        except Exception as e:
            raise StageException(self.stage_name, f"Audio extraction failed: {str(e)}", e)