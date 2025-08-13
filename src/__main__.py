"""
CLI interface for Content Extractor
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from .content_extractor import ContentExtractor
from .models import ProcessingConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Content Extractor for educational videos")
    
    parser.add_argument("video_file", help="Path to video file to process")
    parser.add_argument("-o", "--output", help="Output path for results JSON")
    parser.add_argument("-c", "--config", help="Path to config JSON file")
    parser.add_argument("--segment-duration", type=int, default=45, help="Segment duration in seconds")
    parser.add_argument("--overlap-duration", type=int, default=10, help="Overlap duration in seconds")
    parser.add_argument("--min-score", type=float, default=0.7, help="Minimum score threshold")
    parser.add_argument("--whisper-model", default="base", help="Whisper model size")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence transformer model name")
    parser.add_argument("--keep-audio", action="store_true", help="Keep the extracted audio file")
    parser.add_argument("--include-embeddings", action="store_true", help="Include embeddings in JSON output")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--export-csv", help="Export results to CSV file")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load config if provided
        config = None
        if args.config:
            import json
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                config = ProcessingConfig.from_dict(config_data)
        else:
            config = ProcessingConfig(
                segment_duration=args.segment_duration,
                overlap_duration=args.overlap_duration,
                min_score_threshold=args.min_score,
                whisper_model=args.whisper_model,
                embedding_model=args.embedding_model,
                include_embeddings_in_json=args.include_embeddings,
                keep_audio=args.keep_audio,
                embedding_batch_size=args.embedding_batch_size,
            )
        
        # Initialize extractor
        extractor = ContentExtractor(config)
        
        # Process video file
        logger.info(f"Processing: {args.video_file}")
        result = extractor.process_video_file(args.video_file, args.output)
        
        # Export to CSV if requested
        if args.export_csv:
            extractor.export_segments_to_csv(result.segments, args.export_csv)
        
        # Print summary
        summary = result.to_dict()["summary"]
        print(f"\nProcessing Summary:")
        print(f"Total segments: {summary['total_segments']}")
        print(f"High-value segments: {summary['high_value_count']}")
        print(f"Average score: {summary['average_score']:.2f}")
        print(f"Processing time: {result.processing_time:.2f} seconds")
        
        if result.high_value_segments:
            print(f"\nTop high-value segments:")
            for i, segment in enumerate(result.high_value_segments[:5]):
                print(f"{i+1}. Score: {segment.value_score:.2f} | Time: {segment.start_time:.1f}-{segment.end_time:.1f}s")
                print(f"   Text: {segment.text[:100]}...")
                print()
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 