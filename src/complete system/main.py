"""
Main entry point for the Reels Content Extractor
"""

import sys
import logging
from pathlib import Path

# Add the src directory to Python path for absolute imports
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator.pipeline_orchestrator import PipelineOrchestrator
from orchestrator.config_manager import ConfigManager
from shared.utils import setup_logging
from shared.exceptions import PipelineException


def main():
    """
    Main entry point for the application
    """
    # Setup logging
    logger = setup_logging()
    
    try:
        # Load configuration from command line arguments
        logger.info("Loading configuration...")
        config = ConfigManager.load_from_args()
        logger.info(f"Configuration loaded: {config.processing_profile} profile")
        
        # Initialize orchestrator
        logger.info("Initializing pipeline orchestrator...")
        orchestrator = PipelineOrchestrator(config)
        
        # Process the video
        logger.info(f"Processing video: {config.video_path}")
        results = orchestrator.process_video(config.video_path)
        
        # Save results
        logger.info("Saving results...")
        orchestrator.save_results(results)
        
        # Print performance report
        print("\n" + "="*60)
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(orchestrator.generate_performance_report())
        print("="*60)
        
        return 0
        
    except PipelineException as e:
        logger.error(f"Pipeline error: {str(e)}")
        print(f"\n❌ Pipeline Error: {str(e)}")
        return 1
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        print(f"\n❌ File not found: {str(e)}")
        return 1
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        print("\n⚠️ Processing interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n❌ Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())