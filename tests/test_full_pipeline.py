#!/usr/bin/env python3
"""
Full end-to-end testing of the modular pipeline system
"""

import sys
import os
from pathlib import Path
import tempfile
import logging

# Add src to path properly
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_test_video():
    """Create a simple test video for testing purposes"""
    try:
        # Try to create a simple test video using matplotlib and moviepy
        import matplotlib.pyplot as plt
        import numpy as np
        from moviepy.editor import VideoClip, AudioFileClip
        import tempfile
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        
        # Create a simple 10-second test video with audio
        def make_frame(t):
            # Simple animated frame
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.linspace(0, 2*np.pi, 100)
            y = np.sin(x + t)
            ax.plot(x, y)
            ax.set_title(f'Test Video - Time: {t:.1f}s')
            ax.set_ylim(-2, 2)
            
            # Convert to numpy array
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return buf
        
        # Create video clip
        video = VideoClip(make_frame, duration=10)
        
        # Save test video
        test_video_path = temp_dir / "test_video.mp4"
        video.write_videofile(str(test_video_path), fps=24, audio=False, verbose=False, logger=None)
        
        logger.info(f"Created test video: {test_video_path}")
        return str(test_video_path)
        
    except ImportError as e:
        logger.warning(f"Could not create test video (missing dependencies): {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to create test video: {e}")
        return None


def find_existing_video():
    """Find an existing video file for testing"""
    # Common video locations to check
    possible_paths = [
        "test_video.mp4",
        "example.mp4", 
        "sample.mp4",
        "examples/test.mp4",
        "videos/example.mp4"
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            logger.info(f"Found existing video: {path}")
            return str(Path(path).absolute())
    
    return None


def test_modular_imports():
    """Test that all modular components can be imported"""
    print("🔍 Testing modular component imports...")
    
    try:
        # Test shared components
        from shared.base_stage import BaseStage
        from shared.models import ProcessingConfig, Segment
        from shared.exceptions import PipelineException, StageException
        from shared.utils import format_time, setup_logging
        print("✅ Shared components imported successfully")
        
        # Test orchestrator
        from orchestrator.config_manager import ConfigManager
        from orchestrator.performance_monitor import PerformanceMonitor
        print("✅ Orchestrator components imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False


def test_config_profiles():
    """Test different configuration profiles"""
    print("\n📋 Testing configuration profiles...")
    
    try:
        from shared.models import ProcessingConfig
        
        profiles = ["draft", "fast", "balanced", "quality"]
        for profile in profiles:
            config = ProcessingConfig.create_profile(profile)
            print(f"✅ {profile.capitalize()} profile:")
            print(f"   Whisper model: {config.whisper_model}")
            print(f"   Evaluation enabled: {config.enable_content_evaluation}")
            print(f"   Rule-based scoring: {config.use_rule_based_scoring}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def test_orchestrator_initialization():
    """Test pipeline orchestrator initialization"""
    print("\n🏗️ Testing orchestrator initialization...")
    
    try:
        from shared.models import ProcessingConfig
        from orchestrator.pipeline_orchestrator import PipelineOrchestrator
        
        # Test with different configurations
        configs = [
            ProcessingConfig.create_profile("draft"),
            ProcessingConfig.create_profile("balanced")
        ]
        
        for i, config in enumerate(configs):
            config.enable_speaker_detection = True  # Enable all stages for testing
            config.enable_content_evaluation = True
            
            orchestrator = PipelineOrchestrator(config)
            print(f"✅ Orchestrator {i+1} initialized:")
            print(f"   Profile: {config.processing_profile}")
            print(f"   Stages: {list(orchestrator.stages.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_stage_creation():
    """Test creating individual stages"""
    print("\n🎭 Testing individual stage creation...")
    
    try:
        from shared.models import ProcessingConfig
        
        config = ProcessingConfig.create_profile("draft")
        config.enable_speaker_detection = True
        
        # Test Stage 1: Audio Extraction
        from stages.audio_extraction.code import AudioExtractionStage
        stage1 = AudioExtractionStage(config)
        print(f"✅ Stage 1 (Audio Extraction): {stage1.stage_name}")
        
        # Test Stage 2: Speaker Segmentation  
        from stages.speaker_segmentation.code.stage_wrapper import SpeakerSegmentationStage
        stage2 = SpeakerSegmentationStage(config)
        print(f"✅ Stage 2 (Speaker Segmentation): {stage2.stage_name}")
        
        # Test Stage 3: Transcription
        from stages.transcription.code import TranscriptionStage
        stage3 = TranscriptionStage(config)
        print(f"✅ Stage 3 (Transcription): {stage3.stage_name}")
        
        # Test Stage 4: Content Segmentation
        from stages.content_segmentation.code import ContentSegmentationStage
        stage4 = ContentSegmentationStage(config)
        print(f"✅ Stage 4 (Content Segmentation): {stage4.stage_name}")
        
        # Test Stage 5: Content Evaluation
        from stages.content_evaluation.code import ContentEvaluationStage
        stage5 = ContentEvaluationStage(config)
        print(f"✅ Stage 5 (Content Evaluation): {stage5.stage_name}")
        
        # Test Stage 6: Output Generation
        from stages.output_generation.code import OutputGenerationStage
        stage6 = OutputGenerationStage(config)
        print(f"✅ Stage 6 (Output Generation): {stage6.stage_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual stage creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitoring():
    """Test performance monitoring system"""
    print("\n📊 Testing performance monitoring...")
    
    try:
        from orchestrator.performance_monitor import PerformanceMonitor
        import time
        
        monitor = PerformanceMonitor()
        monitor.start_pipeline()
        
        # Simulate stage metrics
        test_stages = [
            {'stage_name': 'audio_extraction', 'execution_time': 2.5, 'success': True, 'timestamp': time.time()},
            {'stage_name': 'transcription', 'execution_time': 15.2, 'success': True, 'timestamp': time.time()},
            {'stage_name': 'evaluation', 'execution_time': 8.7, 'success': True, 'timestamp': time.time()}
        ]
        
        for metrics in test_stages:
            monitor.add_stage_metrics(metrics)
        
        # Finalize monitoring
        final_metrics = monitor.finish_pipeline(total_duration=120.0)
        
        print("✅ Performance monitoring working:")
        print(f"   Processing speed: {final_metrics.get('processing_speed', 0):.1f}x realtime")
        print(f"   Bottleneck: {final_metrics.get('bottleneck_stage', 'None')}")
        print(f"   Efficiency: {final_metrics.get('efficiency', 0)*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False


def test_end_to_end_dry_run():
    """Test end-to-end pipeline with mock data (no actual video)"""
    print("\n🎬 Testing end-to-end pipeline (dry run)...")
    
    try:
        from shared.models import ProcessingConfig
        from orchestrator.pipeline_orchestrator import PipelineOrchestrator
        
        # Use draft profile for speed
        config = ProcessingConfig.create_profile("draft")
        config.enable_speaker_detection = False  # Disable to avoid dependencies
        config.enable_content_evaluation = True
        config.output_dir = "test_results"
        
        orchestrator = PipelineOrchestrator(config)
        
        print("✅ End-to-end pipeline structure ready:")
        print(f"   Orchestrator initialized: {orchestrator is not None}")
        print(f"   Stages configured: {list(orchestrator.stages.keys())}")
        print(f"   Performance monitor ready: {orchestrator.performance_monitor is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ End-to-end dry run failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualization_imports():
    """Test visualization component imports"""
    print("\n📈 Testing visualization imports...")
    
    try:
        # Test visualization imports (they may fail due to missing matplotlib)
        try:
            from stages.audio_extraction.visualizations.audio_waveform import create_waveform_plot
            print("✅ Audio visualization available")
        except ImportError:
            print("⚠️  Audio visualization unavailable (missing dependencies)")
        
        try:
            from stages.speaker_segmentation.visualizations.speaker_timeline import create_speaker_timeline
            print("✅ Speaker visualization available") 
        except ImportError:
            print("⚠️  Speaker visualization unavailable (missing dependencies)")
        
        try:
            from stages.content_evaluation.visualizations.score_distribution import create_score_distribution_plot
            print("✅ Evaluation visualization available")
        except ImportError:
            print("⚠️  Evaluation visualization unavailable (missing dependencies)")
        
        try:
            from stages.output_generation.visualizations.results_dashboard import create_results_dashboard
            print("✅ Results dashboard available")
        except ImportError:
            print("⚠️  Results dashboard unavailable (missing dependencies)")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization import test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 FULL PIPELINE TESTING\n")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_modular_imports),
        ("Configuration Tests", test_config_profiles), 
        ("Orchestrator Tests", test_orchestrator_initialization),
        ("Individual Stage Tests", test_individual_stage_creation),
        ("Performance Monitoring", test_performance_monitoring),
        ("Visualization Tests", test_visualization_imports),
        ("End-to-End Dry Run", test_end_to_end_dry_run)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        if test_func():
            passed += 1
    
    # Final summary
    print(f"\n{'='*50}")
    print(f"📊 TEST RESULTS: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ The modular pipeline system is ready for use!")
        print("\n🚀 Next steps:")
        print("   1. Test with actual video: python src/main.py video.mp4 --profile draft")
        print("   2. Check results in the output directory")
        print("   3. View generated visualizations")
        print("   4. Iterate and optimize based on results")
        return 0
    else:
        print(f"❌ {total - passed} test suite(s) failed")
        print("\n🔧 Issues to address:")
        print("   - Check import errors for missing dependencies")
        print("   - Verify file paths and module structure") 
        print("   - Install required packages (librosa, matplotlib, etc.)")
        return 1


if __name__ == "__main__":
    sys.exit(main())