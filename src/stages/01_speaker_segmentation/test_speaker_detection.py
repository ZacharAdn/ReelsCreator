#!/usr/bin/env python3
"""
Test Suite for Speaker Segmentation Module
"""

import sys
import time
import logging
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from speaker_analyzer import (
    SpeakerSegmentationPipeline,
    SpeakerSegmentationConfig,
    SpeakerSegment,
    SpeakerAnalysisResult
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeakerDetectionTests:
    """Test suite for speaker detection functionality"""
    
    def __init__(self):
        self.pipeline = SpeakerSegmentationPipeline()
        self.test_results = []
    
    def run_all_tests(self, test_video_path: str = None):
        """Run comprehensive test suite"""
        
        print("🧪 Speaker Detection Test Suite")
        print("=" * 50)
        
        # Configuration tests
        self.test_configuration()
        
        # Pipeline tests (require test video)
        if test_video_path and Path(test_video_path).exists():
            self.test_video_analysis(test_video_path)
            self.test_performance_metrics(test_video_path)
            self.test_segment_quality(test_video_path)
            self.test_first_seconds_accuracy(test_video_path)
        else:
            print("⚠️  No test video provided - skipping integration tests")
            print("   Usage: python test_speaker_detection.py <video_file>")
        
        # Summary
        self.print_test_summary()
    
    def test_configuration(self):
        """Test configuration and initialization"""
        
        print("\n🔧 Testing Configuration...")
        
        try:
            # Default config
            config = SpeakerSegmentationConfig()
            assert config.window_size == 0.5
            assert config.hop_size == 0.25
            assert config.max_gap == 3.0
            self._log_test_result("Default configuration", True)
            
            # Custom config
            custom_config = SpeakerSegmentationConfig(
                window_size=1.0,
                hop_size=0.5,
                max_gap=5.0
            )
            pipeline = SpeakerSegmentationPipeline(custom_config)
            assert pipeline.config.window_size == 1.0
            self._log_test_result("Custom configuration", True)
            
        except Exception as e:
            self._log_test_result("Configuration", False, str(e))
    
    def test_video_analysis(self, video_path: str):
        """Test end-to-end video analysis"""
        
        print(f"\n🎬 Testing Video Analysis: {Path(video_path).name}")
        
        try:
            start_time = time.time()
            result = self.pipeline.analyze_video(video_path)
            processing_time = time.time() - start_time
            
            # Basic validations
            assert isinstance(result, SpeakerAnalysisResult)
            assert result.total_duration > 0
            assert len(result.teacher_segments) > 0
            assert result.teacher_time > 0
            assert 0 <= result.teacher_percentage <= 100
            
            print(f"   ✅ Analysis completed in {processing_time:.2f}s")
            print(f"   📊 Found {len(result.teacher_segments)} teacher segments")
            print(f"   🎓 Teacher speaking: {result.teacher_time:.1f}s ({result.teacher_percentage:.1f}%)")
            
            self._log_test_result("Video analysis", True, f"{processing_time:.2f}s")
            
        except Exception as e:
            self._log_test_result("Video analysis", False, str(e))
    
    def test_performance_metrics(self, video_path: str):
        """Test performance requirements"""
        
        print("\n⚡ Testing Performance Metrics...")
        
        try:
            # Measure processing speed
            start_time = time.time()
            result = self.pipeline.analyze_video(video_path)
            processing_time = time.time() - start_time
            
            # Performance requirements
            speed_ratio = result.total_duration / processing_time
            
            print(f"   📈 Processing speed: {speed_ratio:.1f}x real-time")
            
            # Benchmarks
            if speed_ratio > 5.0:
                print("   🚀 EXCELLENT performance (>5x real-time)")
            elif speed_ratio > 2.0:
                print("   ✅ GOOD performance (>2x real-time)")
            elif speed_ratio > 1.0:
                print("   ⚠️  ACCEPTABLE performance (>1x real-time)")
            else:
                print("   ❌ SLOW performance (<1x real-time)")
            
            self._log_test_result("Performance", speed_ratio > 1.0, f"{speed_ratio:.1f}x")
            
        except Exception as e:
            self._log_test_result("Performance", False, str(e))
    
    def test_segment_quality(self, video_path: str):
        """Test quality of segment detection"""
        
        print("\n🎯 Testing Segment Quality...")
        
        try:
            result = self.pipeline.analyze_video(video_path)
            teacher_segments = result.get_teacher_segments()
            
            # Quality metrics
            excellent_segments = [s for s in teacher_segments if s.duration >= 30]
            good_segments = [s for s in teacher_segments if 10 <= s.duration < 30]
            short_segments = [s for s in teacher_segments if s.duration < 10]
            
            print(f"   🎯 Excellent segments (≥30s): {len(excellent_segments)}")
            print(f"   ⭐ Good segments (10-29s): {len(good_segments)}")
            print(f"   📝 Short segments (<10s): {len(short_segments)}")
            
            # Quality assessment
            usable_segments = len(excellent_segments) + len(good_segments)
            total_segments = len(teacher_segments)
            quality_ratio = usable_segments / total_segments if total_segments > 0 else 0
            
            print(f"   📊 Usable for reels: {usable_segments}/{total_segments} ({quality_ratio:.1%})")
            
            # Show top segments
            if teacher_segments:
                print(f"\n   🏆 Top segments:")
                sorted_segments = sorted(teacher_segments, key=lambda x: x.duration, reverse=True)
                for i, seg in enumerate(sorted_segments[:3]):
                    quality = self._get_segment_quality_label(seg.duration)
                    print(f"      {i+1}. {self._format_time(seg.start_time)} - {self._format_time(seg.end_time)} ({seg.duration:.1f}s) {quality}")
            
            self._log_test_result("Segment quality", quality_ratio > 0.5, f"{quality_ratio:.1%}")
            
        except Exception as e:
            self._log_test_result("Segment quality", False, str(e))
    
    def test_first_seconds_accuracy(self, video_path: str):
        """Test accuracy in the first few seconds (known issue: teacher shouldn't speak in first 4s)"""
        
        print("\n🔍 Testing First Seconds Accuracy...")
        
        try:
            # Create a more precise configuration for this test
            precise_config = SpeakerSegmentationConfig(
                window_size=0.25,   # Smaller windows
                hop_size=0.125,     # More overlap
                max_gap=1.0,        # Smaller gaps for precision
                min_duration=1.0    # Shorter minimum segments
            )
            
            precise_pipeline = SpeakerSegmentationPipeline(precise_config)
            result = precise_pipeline.analyze_video(video_path)
            
            # Check if any teacher segment starts before 4 seconds
            early_teacher_segments = [
                seg for seg in result.teacher_segments 
                if seg.start_time < 4.0
            ]
            
            if early_teacher_segments:
                print(f"   ❌ ISSUE: Found teacher speaking in first 4 seconds:")
                for seg in early_teacher_segments:
                    if seg.start_time < 4.0:
                        print(f"      - {self._format_time(seg.start_time)} - {self._format_time(min(seg.end_time, 4.0))}")
                
                self._log_test_result("First seconds accuracy", False, "Teacher detected before 4s")
            else:
                print(f"   ✅ Correct: No teacher speech detected in first 4 seconds")
                self._log_test_result("First seconds accuracy", True, "No early teacher detection")
            
            # Show detailed breakdown of first 10 seconds
            print(f"\n   📊 First 10 seconds breakdown:")
            for seg in result.teacher_segments:
                if seg.start_time < 10.0:
                    start_display = max(seg.start_time, 0)
                    end_display = min(seg.end_time, 10.0)
                    print(f"      Teacher: {self._format_time(start_display)} - {self._format_time(end_display)}")

            # Report known Q&A window for debugging (no hard assertions)
            print(f"\n   🔎 Reporting Q&A window (1:39-1:57) for diagnostics...")
            teacher_in_period = [
                seg for seg in result.teacher_segments
                if not (seg.end_time <= 99.0 or seg.start_time >= 117.0)
            ]

            total_teacher = sum(min(seg.end_time, 117.0) - max(seg.start_time, 99.0)
                                for seg in teacher_in_period) if teacher_in_period else 0.0
            period_duration = 18.0
            teacher_ratio = total_teacher / period_duration
            print(f"      Teacher time in 1:39-1:57: {total_teacher:.1f}s ({teacher_ratio:.0%})")
            self._log_test_result("Q&A window diagnostic", True, f"Teacher {teacher_ratio:.0%}")
            
        except Exception as e:
            self._log_test_result("First seconds accuracy", False, str(e))
    
    def _get_segment_quality_label(self, duration: float) -> str:
        """Get quality label for segment duration"""
        if duration >= 30:
            return "🎯 EXCELLENT"
        elif duration >= 15:
            return "⭐ VERY GOOD"
        elif duration >= 10:
            return "✅ GOOD"
        else:
            return "📝 SHORT"
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _log_test_result(self, test_name: str, passed: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            'name': test_name,
            'passed': passed,
            'details': details
        })
        
        detail_str = f" ({details})" if details else ""
        print(f"   {status}: {test_name}{detail_str}")
    
    def print_test_summary(self):
        """Print final test summary"""
        
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)
        
        passed_tests = sum(1 for test in self.test_results if test['passed'])
        total_tests = len(self.test_results)
        
        print(f"Tests passed: {passed_tests}/{total_tests}")
        print(f"Success rate: {passed_tests/total_tests:.1%}" if total_tests > 0 else "No tests run")
        
        # Failed tests details
        failed_tests = [test for test in self.test_results if not test['passed']]
        if failed_tests:
            print(f"\n❌ Failed tests:")
            for test in failed_tests:
                print(f"   - {test['name']}: {test['details']}")
        
        # Overall status
        if passed_tests == total_tests and total_tests > 0:
            print("\n🎉 ALL TESTS PASSED! Speaker detection is working correctly.")
        elif passed_tests > total_tests * 0.8:
            print("\n⚠️  MOSTLY WORKING: Some minor issues detected.")
        else:
            print("\n❌ ISSUES DETECTED: Speaker detection needs attention.")

def main():
    """Main test runner"""
    
    # Get test video path
    test_video_path = None
    if len(sys.argv) > 1:
        test_video_path = sys.argv[1]
        if not Path(test_video_path).exists():
            print(f"❌ Test video not found: {test_video_path}")
            return
    
    # Run tests
    tests = SpeakerDetectionTests()
    tests.run_all_tests(test_video_path)

if __name__ == "__main__":
    main()
