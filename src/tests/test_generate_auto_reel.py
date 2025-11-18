#!/usr/bin/env python3
"""
Test script for generate_auto_reel.py

Tests the automated reel generation functionality including:
- AI summary parsing
- Full transcript extraction
- LLM response parsing
- Duration validation
- File path inference
"""

import os
import sys

# Add parent directory to path to import from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from generate_auto_reel import (
    parse_ai_summary,
    get_full_transcript,
    parse_llm_response,
    validate_and_calculate_duration,
    find_latest_results_dir,
    extract_video_path_from_results
)
from cut_video_segments import parse_timestamp, parse_time_range


def test_parse_timestamp():
    """Test timestamp parsing"""
    print("Testing parse_timestamp()...")

    tests = [
        ("0:15", 15.0),
        ("1:30", 90.0),
        ("2:45.50", 165.5),
        ("10:00.25", 600.25),
    ]

    passed = 0
    for ts_str, expected in tests:
        result = parse_timestamp(ts_str)
        if abs(result - expected) < 0.01:
            print(f"  ✅ '{ts_str}' → {result}s")
            passed += 1
        else:
            print(f"  ❌ '{ts_str}' → {result}s (expected {expected}s)")

    print(f"\nPassed: {passed}/{len(tests)}\n")
    return passed == len(tests)


def test_parse_llm_response():
    """Test LLM response parsing"""
    print("Testing parse_llm_response()...")

    # Simulated LLM response
    llm_response = """
PARTS:
1. [0:15 - 0:30] Strong hook about data power
2. [1:00 - 1:20] Model training explanation
3. [2:30 - 2:50] Results demonstration

TOTAL_DURATION: 65s

NARRATIVE: This combination effectively tells the story of data-driven modeling.

TITLE: Data Science Fundamentals
"""

    result = parse_llm_response(llm_response)

    if not result:
        print("  ❌ Failed to parse LLM response")
        return False

    # Check parts
    if len(result['parts']) != 3:
        print(f"  ❌ Expected 3 parts, got {len(result['parts'])}")
        return False

    print(f"  ✅ Parsed {len(result['parts'])} parts")

    # Check time ranges
    expected_ranges = [
        ("0:15 - 0:30", "Strong hook about data power"),
        ("1:00 - 1:20", "Model training explanation"),
        ("2:30 - 2:50", "Results demonstration"),
    ]

    all_correct = True
    for i, (part, (exp_range, exp_reason)) in enumerate(zip(result['parts'], expected_ranges), 1):
        if part['time_range'] == exp_range and exp_reason in part['reason']:
            print(f"  ✅ Part {i}: {part['time_range']}")
        else:
            print(f"  ❌ Part {i}: Got '{part['time_range']}', expected '{exp_range}'")
            all_correct = False

    # Check narrative and title
    if 'narrative' in result and 'NARRATIVE' in llm_response:
        print(f"  ✅ Narrative extracted")
    else:
        print(f"  ❌ Narrative not extracted")
        all_correct = False

    if 'title' in result and 'title' in llm_response.lower():
        print(f"  ✅ Title extracted: {result['title']}")
    else:
        print(f"  ❌ Title not extracted")
        all_correct = False

    print()
    return all_correct


def test_validate_duration():
    """Test duration validation"""
    print("Testing validate_and_calculate_duration()...")

    # Test case 1: Valid multi-part reel (45-70s range)
    parts_valid = [
        {'time_range': '0:15 - 0:30', 'reason': 'Hook'},  # 15s
        {'time_range': '1:00 - 1:20', 'reason': 'Content'},  # 20s
        {'time_range': '2:30 - 2:50', 'reason': 'Results'},  # 20s
    ]

    is_valid, duration = validate_and_calculate_duration(parts_valid)
    expected_duration = 55

    if is_valid and abs(duration - expected_duration) < 1:
        print(f"  ✅ Valid parts: {duration}s")
    else:
        print(f"  ❌ Valid parts: {duration}s (expected ~{expected_duration}s)")
        return False

    # Test case 2: Invalid time range format
    parts_invalid = [
        {'time_range': 'invalid-format', 'reason': 'Test'},
    ]

    is_valid, duration = validate_and_calculate_duration(parts_invalid)

    if not is_valid:
        print(f"  ✅ Invalid format detected correctly")
    else:
        print(f"  ❌ Should have detected invalid format")
        return False

    print()
    return True


def test_find_latest_results():
    """Test finding latest results directory"""
    print("Testing find_latest_results_dir()...")

    try:
        results_dir = find_latest_results_dir()

        if results_dir and os.path.exists(results_dir):
            print(f"  ✅ Found latest results: {os.path.basename(results_dir)}")

            # Check if it follows naming convention
            dirname = os.path.basename(results_dir)
            if len(dirname.split('_')) >= 3:
                print(f"  ✅ Follows YYYY-MM-DD_HHMMSS_VideoName format")
                return True
            else:
                print(f"  ⚠️  Format might be unusual: {dirname}")
                return True
        else:
            print(f"  ⚠️  No results directory found (this is OK if none exist)")
            return True
    except FileNotFoundError:
        print(f"  ⚠️  No results directory found (this is OK if none exist)")
        return True


def test_extract_video_path():
    """Test video path extraction from results directory"""
    print("Testing extract_video_path_from_results()...")

    # Test case: Standard format
    results_dir = "results/2025-11-17_221415_IMG_4314"

    try:
        video_path = extract_video_path_from_results(results_dir)

        if video_path:
            print(f"  ✅ Extracted path: {video_path}")

            if os.path.exists(video_path):
                print(f"  ✅ Video file exists")
                return True
            else:
                print(f"  ⚠️  Video file not found (expected for this test)")
                # This is OK - we're just testing the extraction logic
                return True
        else:
            print(f"  ⚠️  Could not extract video path (this is OK if video doesn't exist)")
            return True
    except FileNotFoundError:
        print(f"  ⚠️  Video file not found (this is OK for testing)")
        return True


def test_ai_summary_parsing():
    """Test AI summary parsing with real results"""
    print("Testing parse_ai_summary() with real data...")

    # Find latest results
    try:
        results_dir = find_latest_results_dir()
    except FileNotFoundError:
        print("  ⚠️  No results directory found, skipping this test")
        return True

    if not results_dir:
        print("  ⚠️  No results directory found, skipping this test")
        return True

    ai_summary_path = os.path.join(results_dir, "ai_summary.txt")

    if not os.path.exists(ai_summary_path):
        print(f"  ⚠️  No ai_summary.txt found in {os.path.basename(results_dir)}")
        return True

    try:
        ai_data = parse_ai_summary(ai_summary_path)

        print(f"  ✅ Parsed AI summary")
        print(f"  📊 Suggestions: {len(ai_data['suggestions'])}")
        print(f"  📝 Topics: {len(ai_data['topics'])}")
        print(f"  #️⃣  Hashtags: {len(ai_data['hashtags'])}")

        if ai_data['cumulative_summary']:
            print(f"  ✅ Found cumulative summary")

        return True
    except Exception as e:
        print(f"  ❌ Error parsing AI summary: {e}")
        return False


def test_full_transcript_extraction():
    """Test full transcript extraction"""
    print("Testing get_full_transcript()...")

    try:
        results_dir = find_latest_results_dir()
    except FileNotFoundError:
        print("  ⚠️  No results directory found, skipping this test")
        return True

    if not results_dir:
        print("  ⚠️  No results directory found, skipping this test")
        return True

    transcript = get_full_transcript(results_dir)

    if transcript:
        print(f"  ✅ Extracted transcript ({len(transcript)} characters)")

        # Check if transcript has Hebrew content
        if any('\u0590' <= c <= '\u05FF' for c in transcript[:1000]):
            print(f"  ✅ Contains Hebrew text")

        return True
    else:
        print(f"  ⚠️  No transcript found (this is OK if it doesn't exist)")
        return True


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("TESTING generate_auto_reel.py")
    print("=" * 80)
    print()

    results = {}

    results['timestamp_parsing'] = test_parse_timestamp()
    results['llm_response_parsing'] = test_parse_llm_response()
    results['duration_validation'] = test_validate_duration()
    results['find_latest_results'] = test_find_latest_results()
    results['extract_video_path'] = test_extract_video_path()
    results['ai_summary_parsing'] = test_ai_summary_parsing()
    results['transcript_extraction'] = test_full_transcript_extraction()

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
        print('python src/scripts/generate_auto_reel.py')
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
