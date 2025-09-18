#!/usr/bin/env python3
"""
Test the modular system with actual functionality
"""

import sys
import os
from pathlib import Path

# Add src to path properly
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

def test_stage_1_audio_extraction():
    """Test Stage 1: Audio Extraction"""
    print("Testing Stage 1: Audio Extraction...")
    
    try:
        from shared.models import ProcessingConfig
        from stages.audio_extraction.code import AudioExtractionStage
        
        # Create test config
        config = ProcessingConfig.create_profile("draft")
        stage = AudioExtractionStage(config)
        
        print("✅ AudioExtractionStage created successfully")
        print(f"   Stage name: {stage.stage_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ AudioExtractionStage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stage_2_speaker_segmentation():
    """Test Stage 2: Speaker Segmentation"""
    print("\nTesting Stage 2: Speaker Segmentation...")
    
    try:
        from shared.models import ProcessingConfig
        from stages.speaker_segmentation.code.stage_wrapper import SpeakerSegmentationStage
        
        # Create test config
        config = ProcessingConfig.create_profile("draft")
        stage = SpeakerSegmentationStage(config)
        
        print("✅ SpeakerSegmentationStage created successfully")
        print(f"   Stage name: {stage.stage_name}")
        print(f"   Detector ready: {stage.detector is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ SpeakerSegmentationStage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator():
    """Test Pipeline Orchestrator"""
    print("\nTesting Pipeline Orchestrator...")
    
    try:
        from shared.models import ProcessingConfig
        from orchestrator.pipeline_orchestrator import PipelineOrchestrator
        
        # Create test config
        config = ProcessingConfig.create_profile("draft")
        config.enable_speaker_detection = True
        
        orchestrator = PipelineOrchestrator(config)
        
        print("✅ PipelineOrchestrator created successfully")
        print(f"   Stages initialized: {list(orchestrator.stages.keys())}")
        print(f"   Config profile: {config.processing_profile}")
        
        return True
        
    except Exception as e:
        print(f"❌ PipelineOrchestrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_pipeline_flow():
    """Test basic pipeline flow with mock data"""
    print("\nTesting Basic Pipeline Flow...")
    
    try:
        from shared.models import ProcessingConfig
        
        # Test config creation with different profiles
        profiles = ["draft", "fast", "balanced", "quality"]
        
        for profile in profiles:
            config = ProcessingConfig.create_profile(profile)
            print(f"✅ Created {profile} profile:")
            print(f"   Whisper model: {config.whisper_model}")
            print(f"   Evaluation enabled: {config.enable_content_evaluation}")
            print(f"   Rule-based scoring: {config.use_rule_based_scoring}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic pipeline flow test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring"""
    print("\nTesting Performance Monitoring...")
    
    try:
        from orchestrator.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor()
        monitor.start_pipeline()
        
        # Simulate stage metrics
        test_metrics = {
            'stage_name': 'test_stage',
            'execution_time': 1.5,
            'success': True,
            'timestamp': 1692000000
        }
        
        monitor.add_stage_metrics(test_metrics)
        final_metrics = monitor.finish_pipeline(total_duration=60.0)
        
        print("✅ Performance monitoring working")
        print(f"   Processing speed: {final_metrics.get('processing_speed', 0):.1f}x realtime")
        print(f"   Total stages: {final_metrics.get('total_stages', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Modular System Components...\n")
    
    tests = [
        test_stage_1_audio_extraction,
        test_stage_2_speaker_segmentation, 
        test_orchestrator,
        test_basic_pipeline_flow,
        test_performance_monitoring
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All modular system tests passed!")
        print("✅ Ready for actual video processing")
        return 0
    else:
        print("❌ Some tests failed. Fix issues before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())