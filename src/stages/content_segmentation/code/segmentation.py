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
        import time
        from tqdm import tqdm
        
        start_time = time.time()
        
        if not segments:
            logger.warning("No segments provided for overlapping")
            return []
        
        logger.info(f"🎵 Starting overlapping segments creation from {len(segments)} segments")
        
        overlapping_segments = []
        current_start = segments[0].start_time
        
        # Find the end time of the last segment
        total_duration = segments[-1].end_time
        logger.info(f"📊 Audio duration: {total_duration:.2f}s, segment duration: {self.segment_duration}s, overlap: {self.overlap_duration}s")
        
        # Calculate expected windows
        step_size = self.segment_duration - self.overlap_duration
        expected_windows = int((total_duration - segments[0].start_time) / step_size) + 1
        logger.info(f"📈 Expected windows: {expected_windows}")
        
        window_count = 0
        progress_bar = tqdm(total=expected_windows, desc="Creating overlapping segments", unit="window")
        
        while current_start < total_duration:
            window_count += 1
            current_end = min(current_start + self.segment_duration, total_duration)
            
            # Find segments that fall within this window
            window_start_time = time.time()
            window_segments = [
                seg for seg in segments 
                if (seg.start_time < current_end and seg.end_time > current_start)
            ]
            
            if window_segments:
                # Combine text from overlapping segments
                text_start_time = time.time()
                combined_text = self._combine_segment_text(window_segments, current_start, current_end)
                text_time = time.time() - text_start_time
                
                # Log slow text combinations
                if text_time > 0.5:
                    logger.warning(f"⚠️ Slow text combination: {text_time:.2f}s for window {window_count} ({len(window_segments)} segments)")
                
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
            
            # Progress update every 10 windows
            if window_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = window_count / elapsed if elapsed > 0 else 0
                logger.debug(f"📈 Progress: {window_count}/{expected_windows} windows ({rate:.1f} windows/s)")
            
            # Move to next segment with overlap (step = segment_duration - overlap_duration)
            current_start += (self.segment_duration - self.overlap_duration)
            progress_bar.update(1)
        
        progress_bar.close()
        
        total_time = time.time() - start_time
        logger.info(f"✅ Created {len(overlapping_segments)} overlapping segments from {len(segments)} original segments in {total_time:.2f}s")
        logger.info(f"📊 Processing rate: {window_count/total_time:.1f} windows/s")
        
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
        if not segments:
            return ""
            
        combined_parts: List[str] = []
        seen_texts: set[str] = set()

        # Sort segments by start time for better text flow
        sorted_segments = sorted(segments, key=lambda x: x.start_time)

        for segment in sorted_segments:
            # Calculate overlap with window
            overlap_start = max(segment.start_time, start_time)
            overlap_end = min(segment.end_time, end_time)

            if overlap_end > overlap_start and segment.text:
                segment_duration = segment.end_time - segment.start_time
                overlap_duration = overlap_end - overlap_start

                if segment_duration > 0:
                    overlap_ratio = overlap_duration / segment_duration
                    # Include segment if it has significant overlap (>30%) with window
                    include_full = overlap_ratio > 0.3
                    text_to_add = segment.text
                    
                    if not include_full and len(segment.text.split()) > 12:
                        # Approximate partial inclusion: take a middle slice
                        words = segment.text.split()
                        start_idx = max(0, int(len(words) * 0.25))
                        end_idx = min(len(words), int(len(words) * 0.75))
                        text_to_add = " ".join(words[start_idx:end_idx])

                    # Deduplicate using normalized text
                    text_key = ' '.join(text_to_add.split()).lower()
                    if text_to_add and text_key not in seen_texts and len(text_key.strip()) > 0:
                        combined_parts.append(text_to_add)
                        seen_texts.add(text_key)

        # Join with spaces and clean up
        result = " ".join(combined_parts).strip()
        return ' '.join(result.split())  # Normalize whitespace
    
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
        import time
        total_start = time.time()
        
        logger.info(f"🔄 Starting segment processing: {len(segments)} input segments")
        
        # Create overlapping segments
        overlap_start = time.time()
        logger.info("📝 Creating overlapping segments...")
        overlapping_segments = self.create_overlapping_segments(segments)
        overlap_time = time.time() - overlap_start
        logger.info(f"✅ Overlapping segments created: {len(overlapping_segments)} segments in {overlap_time:.2f}s")
        
        # Filter by duration
        filter_start = time.time()
        logger.info(f"🔍 Filtering segments by minimum duration ({min_duration}s)...")
        filtered_segments = self.filter_segments_by_duration(overlapping_segments, min_duration)
        filter_time = time.time() - filter_start
        logger.info(f"✅ Segments filtered: {len(filtered_segments)} segments remaining in {filter_time:.2f}s")
        
        total_time = time.time() - total_start
        logger.info(f"🎯 Segment processing completed: {total_time:.2f}s total")
        
        return filtered_segments 