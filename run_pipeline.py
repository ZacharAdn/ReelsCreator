#!/usr/bin/env python3
"""
Simple entry point that works around import issues
"""

import sys
import logging
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point with simplified workflow"""
    parser = argparse.ArgumentParser(description='Reels Content Extractor - Modular Pipeline')
    
    parser.add_argument('video_path', help='Path to video file to process')
    parser.add_argument('--profile', choices=['draft', 'fast', 'balanced', 'quality'], 
                      default='draft', help='Processing profile')
    parser.add_argument('--output', '-o', help='Output directory for results', default='results')
    parser.add_argument('--enable-speaker-detection', action='store_true', 
                      help='Enable speaker detection (requires librosa)')
    parser.add_argument('--minimal-mode', action='store_true',
                      help='Skip embeddings for faster processing')
    
    args = parser.parse_args()
    
    # Validate video file exists
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return 1
    
    print(f"🎬 Starting Reels extraction for: {video_path}")
    print(f"📋 Profile: {args.profile}")
    print(f"📁 Output: {args.output}")
    
    try:
        # Import components
        from shared.models import ProcessingConfig
        
        # Create configuration
        config = ProcessingConfig.create_profile(args.profile)
        config.enable_speaker_detection = args.enable_speaker_detection
        config.minimal_mode = args.minimal_mode
        config.output_dir = args.output
        config.video_path = str(video_path)
        
        print("✅ Configuration created successfully")
        print(f"   Whisper model: {config.whisper_model}")
        print(f"   Evaluation enabled: {config.enable_content_evaluation}")
        print(f"   Speaker detection: {config.enable_speaker_detection}")
        
        # For now, just show what would be processed
        print("\n🏗️ Pipeline stages that would run:")
        stages = [
            "1. Audio Extraction - Extract audio from video",
            "2. Transcription - Convert audio to text using Whisper", 
            "3. Content Segmentation - Create 15-45s Reels segments",
            "4. Content Evaluation - Score segments for quality",
            "5. Output Generation - Export CSV, JSON, reports"
        ]
        
        if config.enable_speaker_detection:
            stages.insert(1, "2. Speaker Segmentation - Detect teacher/student speakers")
            stages = [s.replace("2. Transcription", "3. Transcription") for s in stages[2:]]
            stages = stages[:2] + [s.replace("3. Content", "4. Content").replace("4. Content Evaluation", "5. Content Evaluation").replace("5. Output", "6. Output") for s in stages[2:]]
        
        for stage in stages:
            print(f"   {stage}")
        
        print(f"\n📊 Expected processing:")
        print(f"   Video duration: ~{video_path.stat().st_size / (1024*1024):.1f}MB")
        print(f"   Profile speed: {'Very fast' if args.profile == 'draft' else 'Moderate' if args.profile == 'balanced' else 'Slow'}")
        
        # Show what files would be created
        output_dir = Path(args.output)
        print(f"\n📁 Files that would be created in {output_dir}:")
        print(f"   - reels_extraction_YYYYMMDD_HHMMSS_all_segments.csv")
        print(f"   - reels_extraction_YYYYMMDD_HHMMSS_high_value.csv") 
        print(f"   - reels_extraction_YYYYMMDD_HHMMSS_complete.json")
        print(f"   - reels_extraction_YYYYMMDD_HHMMSS_summary.txt")
        print(f"   - performance_report.txt")
        if not args.minimal_mode:
            print(f"   - Various visualization PNG files")
        
        print("\n🔧 To actually run the pipeline:")
        print("   1. Install missing dependencies: pip install librosa matplotlib seaborn")
        print("   2. Fix import issues in the modular system") 
        print("   3. Re-run this command")
        
        print("\n🚀 Alternative: Use the original system:")
        print("   python -m src video.mp4 --profile draft")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n🔧 Missing dependencies. Install with:")
        print("   pip install librosa matplotlib seaborn pyannote.audio")
        return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())