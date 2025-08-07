#!/usr/bin/env python3
"""
Example usage of Content Extractor
"""

import os
import logging
from pathlib import Path

from src.content_extractor import ContentExtractor
from src.models import ProcessingConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Example usage of Content Extractor"""
    
    # Configuration
    config = ProcessingConfig(
        segment_duration=45,      # 45 second segments
        overlap_duration=10,      # 10 second overlap
        min_score_threshold=0.7,  # Only keep segments with score >= 0.7
        whisper_model="base",     # Use base Whisper model
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # Initialize extractor
    extractor = ContentExtractor(config)
    
    # Example video file path (replace with your actual file)
    video_file = "path/to/your/video/file.mp4"
    
    # Check if file exists
    if not Path(video_file).exists():
        logger.error(f"Video file not found: {video_file}")
        logger.info("Please update the video_file path in this script")
        return
    
    try:
        # Process the video file
        logger.info(f"Processing: {video_file}")
        result = extractor.process_video_file(
            video_file, 
            output_path="results.json"
        )
        
        # Print summary
        summary = result.to_dict()["summary"]
        print(f"\n=== Processing Summary ===")
        print(f"Total segments: {summary['total_segments']}")
        print(f"High-value segments: {summary['high_value_count']}")
        print(f"Average score: {summary['average_score']:.2f}")
        print(f"Processing time: {result.processing_time:.2f} seconds")
        
        # Show top high-value segments
        if result.high_value_segments:
            print(f"\n=== Top High-Value Segments ===")
            for i, segment in enumerate(result.high_value_segments[:5]):
                print(f"\n{i+1}. Score: {segment.value_score:.2f}")
                print(f"   Time: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
                print(f"   Duration: {segment.duration():.1f}s")
                print(f"   Text: {segment.text[:150]}...")
                if segment.reasoning:
                    print(f"   Reasoning: {segment.reasoning}")
        
        # Export to CSV for further analysis
        extractor.export_segments_to_csv(result.segments, "segments.csv")
        logger.info("Results exported to segments.csv")
        
        # Find similar segments
        similar_groups = extractor.get_similar_segments(result.segments, threshold=0.8)
        if similar_groups:
            print(f"\n=== Similar Segment Groups ===")
            for i, group in enumerate(similar_groups[:3]):
                print(f"Group {i+1}: {len(group)} similar segments")
                for idx in group[:3]:  # Show first 3 segments in group
                    seg = result.segments[idx]
                    print(f"  - {seg.start_time:.1f}s: {seg.text[:50]}...")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main() 