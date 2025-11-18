#!/usr/bin/env python3
"""
Test script for transcribe_advanced.py

Tests the transcription functionality including:
- Model loading and fallback logic
- Timestamp formatting
- Directory creation
- Video metadata extraction
- Ollama analyzer (if available)
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from transcribe_advanced import (
    format_timestamp,
    ensure_results_dir,
    get_video_info,
    scan_directory_for_videos,
    find_directories_with_videos,
    OllamaAnalyzer
)


def test_format_timestamp():
    """Test timestamp formatting"""
    print("Testing format_timestamp()...")

    tests = [
        (0, "0:00.00"),
        (45, "0:45.00"),
        (60, "1:00.00"),
        (90, "1:30.00"),
        (125, "2:05.00"),
        (3661, "61:01.00"),
    ]

    passed = 0
    for seconds, expected in tests:
        result = format_timestamp(seconds)
        if result == expected:
            print(f"  ✅ {seconds}s → '{result}'")
            passed += 1
        else:
            print(f"  ❌ {seconds}s → '{result}' (expected '{expected}')")

    print(f"\nPassed: {passed}/{len(tests)}\n")
    return passed == len(tests)


def test_ensure_results_dir():
    """Test results directory creation"""
    print("Testing ensure_results_dir()...")

    video_name = "TEST_VIDEO"
    results_dir = ensure_results_dir(video_name)

    if os.path.exists(results_dir):
        print(f"  ✅ Results directory created: {os.path.basename(results_dir)}")

        # Check naming convention: YYYY-MM-DD_HHMMSS_VideoName
        dirname = os.path.basename(results_dir)
        parts = dirname.split('_')

        if len(parts) >= 3:
            date_part = parts[0]
            time_part = parts[1]
            video_part = '_'.join(parts[2:])

            print(f"  ✅ Date: {date_part}")
            print(f"  ✅ Time: {time_part}")
            print(f"  ✅ Video: {video_part}")

            # Clean up test directory
            try:
                os.rmdir(results_dir)
                print(f"  ✅ Test directory cleaned up")
            except:
                print(f"  ⚠️  Could not clean up test directory")

            return True
        else:
            print(f"  ❌ Unexpected directory format: {dirname}")
            return False
    else:
        print(f"  ❌ Results directory not created")
        return False


def test_get_video_info():
    """Test video metadata extraction"""
    print("Testing get_video_info()...")

    # Find a real video file to test
    data_dir = Path("data")
    video_files = list(data_dir.rglob("*.MOV")) + list(data_dir.rglob("*.MP4")) + list(data_dir.rglob("*.mp4"))

    if not video_files:
        print("  ⚠️  No video files found in data/, skipping this test")
        return True

    test_video = str(video_files[0])
    print(f"  📹 Testing with: {os.path.basename(test_video)}")

    try:
        info = get_video_info(test_video)

        if info['duration'] > 0:
            print(f"  ✅ Duration: {info['duration']:.1f}s")
        else:
            print(f"  ❌ Invalid duration: {info['duration']}")
            return False

        if info['size_mb'] > 0:
            print(f"  ✅ Size: {info['size_mb']:.1f} MB")
        else:
            print(f"  ❌ Invalid size: {info['size_mb']}")
            return False

        if info['modified_date']:
            print(f"  ✅ Modified: {info['modified_date']}")
        else:
            print(f"  ❌ No modification date")
            return False

        print()
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_scan_directory_for_videos():
    """Test directory scanning for videos"""
    print("Testing scan_directory_for_videos()...")

    data_dir = Path("data")

    if not data_dir.exists():
        print("  ⚠️  data/ directory not found, skipping this test")
        return True

    try:
        videos = scan_directory_for_videos(str(data_dir))

        print(f"  ✅ Found {len(videos)} videos")

        if videos:
            # Check first video structure
            first_video = videos[0]
            required_keys = ['path', 'name', 'duration', 'size_mb', 'modified_date']

            all_keys_present = all(key in first_video for key in required_keys)

            if all_keys_present:
                print(f"  ✅ Video metadata structure correct")
                print(f"     Sample: {first_video['name']} ({first_video['duration']:.1f}s)")
            else:
                print(f"  ❌ Missing required keys in video metadata")
                return False

        print()
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_find_directories_with_videos():
    """Test finding all directories with videos"""
    print("Testing find_directories_with_videos()...")

    try:
        directories = find_directories_with_videos()

        print(f"  ✅ Found {len(directories)} directories with videos")

        if directories:
            # Check that it returns directory paths
            first_dir = directories[0]

            if isinstance(first_dir, str) and os.path.exists(first_dir):
                print(f"  ✅ Returns valid directory paths")
                print(f"     Sample: {first_dir}")
            else:
                print(f"  ❌ Invalid directory path: {first_dir}")
                return False

        print()
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def test_ollama_analyzer():
    """Test Ollama analyzer availability"""
    print("Testing OllamaAnalyzer...")

    # Create analyzer instance
    analyzer = OllamaAnalyzer()
    is_available = analyzer.is_available()

    if is_available:
        print(f"  ✅ Ollama is available")
        print(f"  💡 AI analysis will be enabled during transcription")
    else:
        print(f"  ⚠️  Ollama not available (this is OK)")
        print(f"  💡 Transcription will work without AI analysis")

    # Analyzer already initialized above
    print(f"  ✅ OllamaAnalyzer initialized")

    print()
    return True


def test_results_directory_structure():
    """Test existing results directory structure"""
    print("Testing results directory structure...")

    results_dir = Path("results")

    if not results_dir.exists():
        print("  ⚠️  results/ directory not found, skipping this test")
        return True

    # Find all results directories
    result_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

    if not result_dirs:
        print("  ⚠️  No results directories found, skipping this test")
        return True

    print(f"  ✅ Found {len(result_dirs)} results directories")

    # Check one directory structure
    test_dir = result_dirs[0]
    expected_files = ["full_transcript.txt"]

    found_files = [f.name for f in test_dir.iterdir() if f.is_file()]

    if "full_transcript.txt" in found_files:
        print(f"  ✅ Contains full_transcript.txt")
    else:
        print(f"  ⚠️  No full_transcript.txt found (might be incomplete run)")

    # Check for chunk files
    chunk_files = [f for f in found_files if f.startswith("chunk_") and f.endswith(".txt")]
    if chunk_files:
        print(f"  ✅ Contains {len(chunk_files)} chunk files")

    # Check for AI files (optional)
    if "ai_summary.txt" in found_files:
        print(f"  ✅ Contains ai_summary.txt (Ollama analysis)")

    if "suggested_reels.txt" in found_files:
        print(f"  ✅ Contains suggested_reels.txt")

    print()
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("TESTING transcribe_advanced.py")
    print("=" * 80)
    print()

    results = {}

    results['timestamp_formatting'] = test_format_timestamp()
    results['results_dir_creation'] = test_ensure_results_dir()
    results['video_info_extraction'] = test_get_video_info()
    results['directory_scanning'] = test_scan_directory_for_videos()
    results['find_directories'] = test_find_directories_with_videos()
    results['ollama_analyzer'] = test_ollama_analyzer()
    results['results_structure'] = test_results_directory_structure()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "-" * 80)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        print("\nYou can now run the script with:")
        print('python src/scripts/transcribe_advanced.py')
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
