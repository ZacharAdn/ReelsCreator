"""
CLI interface for Content Extractor
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

from .orchestrator.pipeline_orchestrator import PipelineOrchestrator
from .orchestrator.config_manager import ConfigManager
from .shared.models import ProcessingConfig
from .shared.utils import setup_logging
from .shared.exceptions import PipelineException


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
    return logging.getLogger(__name__)


def main():
    """Main CLI function using the orchestrator pipeline"""
    try:
        # Setup logging first
        logger = setup_logging()
        
        # Use ConfigManager to parse arguments and create config
        config = ConfigManager.load_from_args()
        logger.info(f"🔧 Configuration loaded: {config.processing_profile} profile")
        
        # Get video path from arguments  
        import sys
        if len(sys.argv) < 2:
            logger.error("Please provide a video file path")
            sys.exit(1)
        
        video_path = sys.argv[1]
        
        # Initialize orchestrator
        logger.info("🏗️ Initializing pipeline orchestrator...")
        orchestrator = PipelineOrchestrator(config)
        
        # Process video file
        logger.info(f"🎬 Processing video: {video_path}")
        results = orchestrator.process_video(video_path)
        
        # Display results summary
        logger.info("🎉 Processing completed successfully!")
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        
        # Show performance metrics
        performance_report = orchestrator.generate_performance_report()
        print(f"📊 Performance Report:")
        print(performance_report)
        
        # Show stage results if available
        for stage_name, stage_result in results.items():
            if isinstance(stage_result, dict) and 'segments' in stage_result:
                segments = stage_result['segments']
                print(f"   {stage_name}: {len(segments)} segments")
        
        print("✅ Pipeline execution completed!")
        
    except PipelineException as e:
        logger.error(f"❌ Pipeline error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main() 