"""
Content Segmentation Stage - Create overlapping segments for Reels
"""

import logging
from typing import Dict, Any, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.base_stage import BaseStage
from shared.exceptions import StageException
from shared.models import Segment
from .segmentation import SegmentProcessor

logger = logging.getLogger(__name__)


class ContentSegmentationStage(BaseStage):
    """
    Stage 4: Create overlapping content segments optimized for Reels
    
    Input: {
        'transcribed_segments': List[Segment],
        'duration': float,
        ...
    }
    Output: {
        'reels_segments': List[Segment],
        'segmentation_summary': Dict
    }
    """
    
    def __init__(self, config):
        super().__init__(config, "ContentSegmentation")
        
        # Initialize segment processor with Reels-optimized settings
        self.segment_processor = SegmentProcessor(
            segment_duration=getattr(config, 'segment_duration', 45),  # 45s segments for Reels
            overlap_duration=getattr(config, 'overlap_duration', 10)   # 10s overlap
        )
        
        self.min_duration = getattr(config, 'min_segment_duration', 15.0)  # 15s minimum for Reels
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input contains transcribed segments"""
        super().validate_input(input_data)
        
        if not isinstance(input_data, dict):
            raise StageException(self.stage_name, "Input must be a dictionary")
        
        if 'transcribed_segments' not in input_data:
            raise StageException(self.stage_name, "Input must contain 'transcribed_segments'")
        
        segments = input_data['transcribed_segments']
        if not segments or not isinstance(segments, list):
            raise StageException(self.stage_name, "transcribed_segments must be a non-empty list")
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create overlapping segments optimized for Reels content
        
        Args:
            input_data: Dictionary containing transcribed segments
            
        Returns:
            Dictionary with Reels-ready segments
        """
        try:
            transcribed_segments = input_data['transcribed_segments']
            
            logger.info(f"Creating Reels segments from {len(transcribed_segments)} transcribed segments")
            
            # Create overlapping segments
            overlapping_segments = self.segment_processor.process_segments(
                transcribed_segments,
                min_duration=self.min_duration
            )
            
            if not overlapping_segments:
                raise StageException(self.stage_name, "No segments created after processing")
            
            # Filter and optimize for Reels
            reels_segments = self._optimize_for_reels(overlapping_segments)
            
            # Create summary
            segmentation_summary = {
                'original_segments': len(transcribed_segments),
                'overlapping_segments': len(overlapping_segments),
                'final_reels_segments': len(reels_segments),
                'average_duration': sum(s.duration() for s in reels_segments) / len(reels_segments) if reels_segments else 0,
                'segment_duration_target': self.segment_processor.segment_duration,
                'overlap_duration': self.segment_processor.overlap_duration,
                'min_duration_filter': self.min_duration
            }
            
            logger.info(f"Segmentation completed: {len(reels_segments)} Reels-ready segments")
            
            return {
                'reels_segments': reels_segments,
                'segmentation_summary': segmentation_summary,
                # Pass through previous data
                'transcribed_segments': transcribed_segments,
                'audio_path': input_data.get('audio_path'),
                'duration': input_data.get('duration', 0),
                'speaker_segments': input_data.get('speaker_segments', [])
            }
            
        except Exception as e:
            raise StageException(self.stage_name, f"Content segmentation failed: {str(e)}", e)
    
    def _optimize_for_reels(self, segments: List[Segment]) -> List[Segment]:
        """
        Optimize segments specifically for Reels format
        
        Args:
            segments: Input segments
            
        Returns:
            Optimized segments for Reels
        """
        optimized = []
        
        for segment in segments:
            # Ensure segment is within Reels duration range (15-60s)
            duration = segment.duration()
            
            if 15 <= duration <= 60:
                # Good duration for Reels
                optimized.append(segment)
            elif duration > 60:
                # Too long - could split into multiple segments in future
                # For now, skip or truncate
                logger.debug(f"Segment too long ({duration:.1f}s), skipping")
                continue
            else:
                # Too short - already filtered by min_duration
                continue
        
        logger.info(f"Optimized {len(segments)} segments to {len(optimized)} Reels-ready segments")
        return optimized