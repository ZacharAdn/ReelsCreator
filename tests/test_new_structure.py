#!/usr/bin/env python3
"""
Quick test to verify the new structure works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that we can import the main components"""
    print("Testing imports...")
    
    try:
        # Test shared components
        from shared.base_stage import BaseStage
        from shared.models import ProcessingConfig
        from shared.exceptions import PipelineException
        print("✅ Shared components imported successfully")
        
        # Test orchestrator
        from orchestrator.pipeline_orchestrator import PipelineOrchestrator
        from orchestrator.config_manager import ConfigManager
        from orchestrator.performance_monitor import PerformanceMonitor
        print("✅ Orchestrator components imported successfully")
        
        # Test audio extraction stage
        from stages.audio_extraction.code import AudioExtractionStage
        print("✅ Audio extraction stage imported successfully")
        
        # Test speaker segmentation stage
        from stages.speaker_segmentation.code.stage_wrapper import SpeakerSegmentationStage
        print("✅ Speaker segmentation stage imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_creation():
    """Test creating a configuration"""
    print("\nTesting configuration creation...")
    
    try:
        from shared.models import ProcessingConfig
        
        # Test profile creation
        config = ProcessingConfig.create_profile("draft")
        print(f"✅ Created draft profile: {config.processing_profile}")
        
        config = ProcessingConfig.create_profile("balanced")
        print(f"✅ Created balanced profile: {config.processing_profile}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False

def test_orchestrator_creation():
    """Test creating an orchestrator"""
    print("\nTesting orchestrator creation...")
    
    try:
        from orchestrator.pipeline_orchestrator import PipelineOrchestrator
        from shared.models import ProcessingConfig
        
        config = ProcessingConfig.create_profile("draft")
        orchestrator = PipelineOrchestrator(config)
        
        print("✅ Orchestrator created successfully")
        print(f"   Stages initialized: {list(orchestrator.stages.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Testing new modular structure...\n")
    
    tests = [
        test_imports,
        test_config_creation,
        test_orchestrator_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The new structure is working.")
        return 0
    else:
        print("❌ Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())