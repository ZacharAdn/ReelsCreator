#!/usr/bin/env python3
"""
Test script for cut_video_segments.py

Tests the exact scenario provided by the user:
Input: data/IMG_4225.MP4, 1:00.26-1:07.16, 1:27.64-1:31.72, 1:42.30-1:49.04, 2:00.08-2:06.68
Output: generated_data/IMG_4225_REEL.MP4
"""

import os
import sys
from cut_video_segments import (
    parse_timestamp,
    parse_time_range,
    parse_ranges,
    format_time,
    ensure_output_dir
)

def test_parse_timestamp():
    """Test timestamp parsing with various formats"""
    print("Testing parse_timestamp()...")

    tests = [
        ("1:00.26", 60.26),
        ("1:07.16", 67.16),
        ("1:27.64", 87.64),
        ("1:31.72", 91.72),
        ("1:42.30", 102.30),
        ("1:49.04", 109.04),
        ("2:00.08", 120.08),
        ("2:06.68", 126.68),
        ("45.50", 45.50),
        ("1:23", 83.0),
    ]

    passed = 0
    for ts_str, expected in tests:
        result = parse_timestamp(ts_str)
        if abs(result - expected) < 0.01:  # Allow small floating point differences
            print(f"  ✅ '{ts_str}' → {result}s (expected {expected}s)")
            passed += 1
        else:
            print(f"  ❌ '{ts_str}' → {result}s (expected {expected}s)")

    print(f"\nPassed: {passed}/{len(tests)}\n")
    return passed == len(tests)

def test_parse_time_range():
    """Test time range parsing"""
    print("Testing parse_time_range()...")

    tests = [
        ("1:00.26-1:07.16", (60.26, 67.16)),
        ("1:27.64-1:31.72", (87.64, 91.72)),
        ("1:42.30-1:49.04", (102.30, 109.04)),
        ("2:00.08-2:06.68", (120.08, 126.68)),
    ]

    passed = 0
    for range_str, (expected_start, expected_end) in tests:
        start, end = parse_time_range(range_str)
        if (abs(start - expected_start) < 0.01 and
            abs(end - expected_end) < 0.01):
            print(f"  ✅ '{range_str}' → ({start}s, {end}s)")
            passed += 1
        else:
            print(f"  ❌ '{range_str}' → ({start}s, {end}s) (expected ({expected_start}s, {expected_end}s))")

    print(f"\nPassed: {passed}/{len(tests)}\n")
    return passed == len(tests)

def test_parse_ranges():
    """Test parsing multiple ranges (user's exact input)"""
    print("Testing parse_ranges() with user's example...")

    # User's exact input
    ranges_str = "1:00.26-1:07.16, 1:27.64-1:31.72, 1:42.30-1:49.04, 2:00.08-2:06.68"

    expected = [
        (60.26, 67.16),
        (87.64, 91.72),
        (102.30, 109.04),
        (120.08, 126.68),
    ]

    result = parse_ranges(ranges_str)

    print(f"  Input: '{ranges_str}'")
    print(f"  Expected {len(expected)} ranges")
    print(f"  Got {len(result)} ranges")

    if len(result) != len(expected):
        print("  ❌ Range count mismatch!")
        return False

    passed = True
    for i, ((r_start, r_end), (e_start, e_end)) in enumerate(zip(result, expected), 1):
        if (abs(r_start - e_start) < 0.01 and abs(r_end - e_end) < 0.01):
            print(f"  ✅ Range {i}: {format_time(r_start)} - {format_time(r_end)}")
        else:
            print(f"  ❌ Range {i}: Got ({r_start}, {r_end}), expected ({e_start}, {e_end})")
            passed = False

    # Calculate total duration
    total_duration = sum(end - start for start, end in result)
    print(f"\n  Total duration of all segments: {total_duration:.2f}s ({total_duration/60:.1f} minutes)")

    return passed

def test_format_time():
    """Test time formatting"""
    print("\nTesting format_time()...")

    tests = [
        (60.26, "1:00.26"),
        (67.16, "1:07.16"),
        (87.64, "1:27.64"),
        (91.72, "1:31.72"),
        (102.30, "1:42.30"),
        (109.04, "1:49.04"),
        (120.08, "2:00.08"),
        (126.68, "2:06.68"),
    ]

    passed = 0
    for seconds, expected in tests:
        result = format_time(seconds)
        if result == expected:
            print(f"  ✅ {seconds}s → '{result}'")
            passed += 1
        else:
            print(f"  ❌ {seconds}s → '{result}' (expected '{expected}')")

    print(f"\nPassed: {passed}/{len(tests)}\n")
    return passed == len(tests)

def test_ensure_output_dir():
    """Test output directory creation"""
    print("Testing ensure_output_dir()...")

    output_dir = ensure_output_dir()

    if os.path.exists(output_dir):
        print(f"  ✅ Output directory exists: {output_dir}")
        if os.path.isdir(output_dir):
            print(f"  ✅ Path is a directory")
            return True
        else:
            print(f"  ❌ Path is not a directory")
            return False
    else:
        print(f"  ❌ Output directory does not exist: {output_dir}")
        return False

def test_video_exists():
    """Check if the test video exists"""
    print("\nChecking test video availability...")

    video_path = "data/IMG_4225.MP4"

    if os.path.exists(video_path):
        size_mb = os.path.getsize(video_path) / (1024 * 1024)
        print(f"  ✅ Test video found: {video_path}")
        print(f"  📹 Size: {size_mb:.1f} MB")
        return True
    else:
        print(f"  ❌ Test video not found: {video_path}")
        print(f"  Note: This test requires the video file to be present")
        return False

def test_expected_output():
    """Check if output file would be created correctly"""
    print("\nTesting expected output path...")

    video_path = "data/IMG_4225.MP4"
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = ensure_output_dir()
    expected_output = os.path.join(output_dir, f"{video_name}_REEL.MP4")

    print(f"  Input video: {video_path}")
    print(f"  Expected output: {expected_output}")

    if expected_output.endswith("IMG_4225_REEL.MP4"):
        print(f"  ✅ Output filename matches expected format")
        return True
    else:
        print(f"  ❌ Output filename doesn't match expected format")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("TESTING cut_video_segments.py")
    print("=" * 60)
    print()

    results = {}

    results['timestamp_parsing'] = test_parse_timestamp()
    results['time_range_parsing'] = test_parse_time_range()
    results['ranges_parsing'] = test_parse_ranges()
    results['time_formatting'] = test_format_time()
    results['output_dir'] = test_ensure_output_dir()
    results['video_exists'] = test_video_exists()
    results['expected_output'] = test_expected_output()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "-" * 60)
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
        print("\nYou can now run the actual script with:")
        print('python "src/quick scripts/cut_video_segments.py" \\')
        print('  --video data/IMG_4225.MP4 \\')
        print('  --ranges "1:00.26-1:07.16, 1:27.64-1:31.72, 1:42.30-1:49.04, 2:00.08-2:06.68"')
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
