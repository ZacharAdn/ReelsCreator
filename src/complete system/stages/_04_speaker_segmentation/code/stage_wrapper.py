"""
Speaker Segmentation Stage - Detect and segment speakers
"""

import logging
from typing import Dict, Any, List, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.base_stage import BaseStage
from shared.exceptions import StageException
from .hybrid_detector import HybridSpeakerDetector

logger = logging.getLogger(__name__)


class SpeakerSegmentationStage(BaseStage):
    """
    Stage 2: Speaker segmentation and analysis
    
    Input: {
        'audio_path': str,
        'duration': float,
        ...
    }
    Output: {
        'speaker_segments': List[Dict],
        'analysis_summary': Dict,
        'processing_summary': Dict
    }
    """
    
    def __init__(self, config):
        super().__init__(config, "SpeakerSegmentation")
        
        # Initialize hybrid detector
        sample_rate = getattr(config, 'sample_rate', 16000)
        enable_pyannote = getattr(config, 'enable_speaker_detection', True)
        
        self.detector = HybridSpeakerDetector(
            sample_rate=sample_rate,
            enable_pyannote=enable_pyannote
        )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input contains audio path"""
        super().validate_input(input_data)
        
        if not isinstance(input_data, dict):
            raise StageException(self.stage_name, "Input must be a dictionary")
        
        if 'audio_path' not in input_data:
            raise StageException(self.stage_name, "Input must contain 'audio_path'")
        
        audio_path = input_data['audio_path']
        if not audio_path or not isinstance(audio_path, str):
            raise StageException(self.stage_name, "audio_path must be a valid string")
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform speaker segmentation
        
        Args:
            input_data: Dictionary containing audio_path and metadata
            
        Returns:
            Dictionary with speaker segments and analysis
        """
        try:
            audio_path = input_data['audio_path']
            
            # Run hybrid speaker analysis
            result = self.detector.analyze_audio_file(
                audio_path, 
                use_temporal_smoothing=True
            )
            
            return {
                'speaker_segments': result.segments,
                'analysis_summary': result.processing_summary,
                'accuracy_improvements': result.accuracy_improvements,
                'fallback_used': result.fallback_used,
                'detector_info': self.detector.get_analysis_summary(),
                'audio_path': audio_path,  # Pass through for next stages
                'duration': input_data.get('duration', 0)
            }
            
        except Exception as e:
            raise StageException(self.stage_name, f"Speaker segmentation failed: {str(e)}", e)