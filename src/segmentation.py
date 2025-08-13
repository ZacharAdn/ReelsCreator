"""
Segmentation module for creating overlapping segments
"""

import logging
from typing import List
from .models import Segment

logger = logging.getLogger(__name__)


class SegmentProcessor:
    """Handles segment processing and creation of overlapping segments"""
    
    def __init__(self, segment_duration: int = 45, overlap_duration: int = 10):
        """
        Initialize segment processor
        
        Args:
            segment_duration: Target duration for each segment in seconds
            overlap_duration: Overlap between segments in seconds
        """
        self.segment_duration = segment_duration
        self.overlap_duration = overlap_duration
        logger.info(f"Initialized with segment_duration={segment_duration}s, overlap_duration={overlap_duration}s")
    
    def create_overlapping_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Create overlapping segments from original segments
        
        Args:
            segments: List of original segments from transcription
            
        Returns:
            List of overlapping segments
        """
        if not segments:
            return []
        
        overlapping_segments = []
        current_start = segments[0].start_time
        
        # Find the end time of the last segment
        total_duration = segments[-1].end_time
        
        while current_start < total_duration:
            current_end = min(current_start + self.segment_duration, total_duration)
            
            # Find segments that fall within this window
            window_segments = [
                seg for seg in segments 
                if (seg.start_time < current_end and seg.end_time > current_start)
            ]
            
            if window_segments:
                # Combine text from overlapping segments
                combined_text = self._combine_segment_text(window_segments, current_start, current_end)
                
                # Calculate average confidence
                avg_confidence = sum(seg.confidence for seg in window_segments) / len(window_segments)
                
                # Create new segment
                new_segment = Segment(
                    start_time=current_start,
                    end_time=current_end,
                    text=combined_text,
                    confidence=avg_confidence
                )
                
                overlapping_segments.append(new_segment)
            
            # Move to next segment with overlap
            current_start = current_end - self.overlap_duration
        
        logger.info(f"Created {len(overlapping_segments)} overlapping segments from {len(segments)} original segments")
        return overlapping_segments
    
    def _combine_segment_text(self, segments: List[Segment], start_time: float, end_time: float) -> str:
        """
        Combine text from segments within a time window
        
        Args:
            segments: List of segments to combine
            start_time: Start time of window
            end_time: End time of window
            
        Returns:
            Combined text
        """
        combined_parts: List[str] = []
        seen_texts: set[str] = set()

        for segment in segments:
            # Calculate overlap with window
            overlap_start = max(segment.start_time, start_time)
            overlap_end = min(segment.end_time, end_time)

            if overlap_end > overlap_start and segment.text:
                segment_duration = segment.end_time - segment.start_time
                overlap_duration = overlap_end - overlap_start

                if segment_duration > 0:
                    include_full = (overlap_duration / segment_duration) > 0.5
                    text_to_add = segment.text
                    if not include_full and len(segment.text.split()) > 12:
                        # Approximate partial inclusion: take a middle slice
                        words = segment.text.split()
                        start_idx = max(0, int(len(words) * 0.25))
                        end_idx = min(len(words), int(len(words) * 0.75))
                        text_to_add = " ".join(words[start_idx:end_idx])

                    if text_to_add and text_to_add not in seen_texts:
                        combined_parts.append(text_to_add)
                        seen_texts.add(text_to_add)

        return " ".join(combined_parts).strip()
    
    def filter_segments_by_duration(self, segments: List[Segment], min_duration: float = 10.0) -> List[Segment]:
        """
        Filter segments by minimum duration
        
        Args:
            segments: List of segments to filter
            min_duration: Minimum duration in seconds
            
        Returns:
            Filtered list of segments
        """
        filtered_segments = [
            seg for seg in segments 
            if seg.duration() >= min_duration
        ]
        
        logger.info(f"Filtered {len(segments)} segments to {len(filtered_segments)} segments (min_duration={min_duration}s)")
        return filtered_segments
    
    def process_segments(self, segments: List[Segment], min_duration: float = 10.0) -> List[Segment]:
        """
        Complete segment processing pipeline
        
        Args:
            segments: List of original segments
            min_duration: Minimum duration for segments
            
        Returns:
            Processed overlapping segments
        """
        # Create overlapping segments
        overlapping_segments = self.create_overlapping_segments(segments)
        
        # Filter by duration
        filtered_segments = self.filter_segments_by_duration(overlapping_segments, min_duration)
        
        return filtered_segments 