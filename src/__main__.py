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
    parser.add_argument("--evaluation-model", default="Qwen/Qwen2.5-0.5B-Instruct", help="LLM model for content evaluation")
    parser.add_argument("--keep-audio", action="store_true", help="Keep the extracted audio file")
    parser.add_argument("--include-embeddings", action="store_true", help="Include embeddings in JSON output")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--export-csv", help="Export results to CSV file")
    
    # Speaker and language processing
    parser.add_argument("--enable-speaker-detection", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--primary-speaker-only", action="store_true", help="Keep only primary speaker segments")
    parser.add_argument("--speaker-batch-size", type=int, default=8, help="Batch size for speaker processing")
    parser.add_argument("--preserve-technical-terms", action="store_true", default=True, help="Preserve technical terminology")
    parser.add_argument("--primary-language", default="he", help="Primary language (he/en)")
    parser.add_argument("--technical-language", default="en", help="Technical terms language")
    
    # Performance optimization options
    parser.add_argument("--profile", choices=["draft", "fast", "balanced", "quality"], default="balanced",
                       help="Processing profile: draft (80%% faster, no LLM), fast (60%% faster, rule-based), balanced (default), quality (20%% slower)")
    parser.add_argument("--evaluation-batch-size", type=int, default=5, help="LLM evaluation batch size")
    parser.add_argument("--enable-similarity", action="store_true", help="Enable similarity analysis")
    parser.add_argument("--minimal-mode", action="store_true", help="Skip non-essential processing for speed")
    
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
        elif args.profile != "balanced":
            # Use optimized profile
            config = ProcessingConfig.create_profile(args.profile)
            # Override with any explicit arguments
            if args.segment_duration != 45:
                config.segment_duration = args.segment_duration
            if args.overlap_duration != 10:
                config.overlap_duration = args.overlap_duration
            if args.min_score != 0.7:
                config.min_score_threshold = args.min_score
            if args.whisper_model != "base":
                config.whisper_model = args.whisper_model
            if args.embedding_model != "all-MiniLM-L6-v2":
                config.embedding_model = args.embedding_model
            if args.evaluation_model != "Qwen/Qwen2.5-0.5B-Instruct":
                config.evaluation_model = args.evaluation_model
        else:
            config = ProcessingConfig(
                segment_duration=args.segment_duration,
                overlap_duration=args.overlap_duration,
                min_score_threshold=args.min_score,
                whisper_model=args.whisper_model,
                embedding_model=args.embedding_model,
                evaluation_model=args.evaluation_model,
                include_embeddings_in_json=args.include_embeddings,
                keep_audio=args.keep_audio,
                embedding_batch_size=args.embedding_batch_size,
                # Speaker and language processing
                enable_speaker_detection=args.enable_speaker_detection,
                primary_speaker_only=args.primary_speaker_only,
                speaker_batch_size=args.speaker_batch_size,
                preserve_technical_terms=args.preserve_technical_terms,
                primary_language=args.primary_language,
                technical_language=args.technical_language,
                # Performance optimization options
                evaluation_batch_size=args.evaluation_batch_size,
                enable_similarity_analysis=args.enable_similarity,
                minimal_mode=args.minimal_mode,
                processing_profile=args.profile,
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