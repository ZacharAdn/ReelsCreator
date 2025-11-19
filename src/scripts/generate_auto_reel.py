#!/usr/bin/env python3
"""
Automated Reel Generator - Intelligently creates optimal 45-70 second reels

This script analyzes transcription results and uses AI to select the best
non-contiguous video segments to create engaging short-form content.

Features:
- Parses AI-generated reel suggestions from transcription
- Uses Ollama LLM to intelligently combine multiple segments
- Supports non-contiguous multi-part reels (2-4 segments)
- Enforces 45-70 second duration constraint
- Automatically cuts and concatenates using FFmpeg

Usage:
    # Standalone mode
    python src/scripts/generate_auto_reel.py \\
      --results-dir results/2025-01-17_221415_IMG_4314 \\
      --video data/IMG_4314.MP4

    # Interactive mode (auto-detect latest results)
    python src/scripts/generate_auto_reel.py

Output:
    generated_data/VideoName_AUTO_REEL.MP4
"""

import os
import sys
import argparse
import re
from typing import List, Dict, Tuple
from pathlib import Path

# Import functions from cut_video_segments.py
from cut_video_segments import (
    parse_time_range,
    parse_timestamp,
    format_time,
    ensure_output_dir,
    get_unique_output_path,
    cut_segments_ffmpeg,
    cut_segments_moviepy
)


def parse_ai_summary(summary_path: str) -> Dict:
    """
    Extract reel suggestions and full transcript from ai_summary.txt

    Args:
        summary_path: Path to ai_summary.txt file

    Returns:
        Dictionary with 'suggestions', 'full_transcript', 'topics', 'hashtags'
    """
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"AI summary not found: {summary_path}")

    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()

    result = {
        'suggestions': [],
        'full_transcript': '',
        'topics': [],
        'hashtags': [],
        'cumulative_summary': ''
    }

    # Parse reel suggestions
    in_reel_section = False
    for line in content.split('\n'):
        line_stripped = line.strip()

        # Find reel suggestions section
        if 'REEL' in line_stripped.upper() and 'SUGGESTION' in line_stripped.upper():
            in_reel_section = True
            continue

        # Stop at next major section
        if in_reel_section and line_stripped.startswith('═'):
            in_reel_section = False
            continue

        # Parse reel entries
        if in_reel_section and line_stripped:
            timestamp_match = re.search(r'(\d+:\d+\.?\d*\s*-\s*\d+:\d+\.?\d*)', line_stripped)
            if timestamp_match:
                timestamp = timestamp_match.group(1).strip()
                reason = line_stripped.split(timestamp, 1)[-1].strip()
                if reason.startswith('-'):
                    reason = reason[1:].strip()

                result['suggestions'].append({
                    'time_range': timestamp,
                    'reason': reason if reason else 'Engaging content'
                })

    # Extract cumulative summary
    if 'Overall Summary:' in content:
        summary_start = content.find('Overall Summary:')
        summary_section = content[summary_start:summary_start+500]
        lines = summary_section.split('\n')
        if len(lines) > 1:
            result['cumulative_summary'] = lines[1].strip()

    # Extract topics
    in_topics = False
    for line in content.split('\n'):
        line_stripped = line.strip()
        if 'All Topics:' in line or 'TOPICS:' in line.upper():
            in_topics = True
            continue
        if in_topics and (line_stripped.startswith('═') or line_stripped.startswith('Suggested Hashtags')):
            in_topics = False
            continue
        if in_topics and line_stripped.startswith('•'):
            result['topics'].append(line_stripped[1:].strip())

    # Extract hashtags
    hashtag_match = re.search(r'#\w+', content)
    if hashtag_match:
        # Find all hashtags in the line
        hashtag_line_start = content.rfind('\n', 0, hashtag_match.start())
        hashtag_line_end = content.find('\n', hashtag_match.start())
        hashtag_line = content[hashtag_line_start:hashtag_line_end]
        result['hashtags'] = re.findall(r'#\w+', hashtag_line)

    return result


def get_chunk_summaries_from_ai_analysis(results_dir: str) -> str:
    """
    Extract per-chunk summaries with timestamps from ai_summary.txt

    This provides the LLM with a structured view of ALL video content,
    not just truncated raw transcript.

    Args:
        results_dir: Results directory containing ai_summary.txt

    Returns:
        Formatted string with all chunk summaries and timestamps
    """
    ai_summary_path = os.path.join(results_dir, "ai_summary.txt")

    if not os.path.exists(ai_summary_path):
        return ""

    with open(ai_summary_path, 'r', encoding='utf-8') as f:
        content = f.read()

    summaries = []

    # Parse all chunk summaries
    # Pattern: CHUNK X ANALYSIS ... Time: MM:SS.ss - MM:SS.ss ... Summary: ...
    chunk_pattern = re.compile(
        r'CHUNK\s+(\d+)\s+ANALYSIS.*?Time:\s*([\d:.]+\s*-\s*[\d:.]+).*?Summary:\s*(.*?)(?=Topics:|$)',
        re.DOTALL | re.IGNORECASE
    )

    matches = chunk_pattern.findall(content)

    for chunk_num, time_range, summary in matches:
        # Clean up the summary
        summary_clean = summary.strip()
        # Limit each summary to prevent overflow
        if len(summary_clean) > 300:
            summary_clean = summary_clean[:297] + "..."

        summaries.append({
            'chunk': int(chunk_num),
            'time': time_range.strip(),
            'summary': summary_clean
        })

    # Sort by chunk number
    summaries.sort(key=lambda x: x['chunk'])

    if not summaries:
        return ""

    # Format for LLM consumption
    result = "VIDEO CONTENT BY TIMESTAMP (AI-analyzed summaries):\n\n"
    for s in summaries:
        result += f"[{s['time']}] Chunk {s['chunk']}:\n"
        result += f"{s['summary']}\n\n"

    return result


def get_full_transcript(results_dir: str) -> str:
    """
    Read the full transcript from full_transcript.txt

    Args:
        results_dir: Results directory

    Returns:
        Full transcript text
    """
    transcript_path = os.path.join(results_dir, "full_transcript.txt")

    if not os.path.exists(transcript_path):
        return ""

    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract just the transcript part (after the header)
    if "Full transcript so far:" in content:
        transcript_start = content.find("Full transcript so far:")
        transcript_text = content[transcript_start:].split('\n', 2)
        if len(transcript_text) > 2:
            return transcript_text[2].strip()

    return content


def analyze_with_llm(chunk_summaries: str, suggestions: List[Dict], cumulative_summary: str = "", min_duration: int = 45, max_duration: int = 70) -> Dict:
    """
    Use Ollama to select the best multi-part reel from video content

    Args:
        chunk_summaries: Formatted chunk summaries with timestamps from AI analysis
        suggestions: List of reel suggestions from AI analysis (if any)
        cumulative_summary: Overall video summary
        min_duration: Minimum reel duration in seconds
        max_duration: Maximum reel duration in seconds

    Returns:
        Dictionary with 'parts', 'total_duration', 'narrative', 'title'
    """
    try:
        import requests

        # Format suggestions for the prompt (if available)
        if suggestions and len(suggestions) > 0:
            suggestions_text = "\n".join([
                f"{i+1}. {s['time_range']} - {s['reason']}"
                for i, s in enumerate(suggestions)
            ])
            suggestions_section = f"\nPRE-DEFINED SUGGESTIONS (use as guidance):\n{suggestions_text}\n"
        else:
            suggestions_section = ""

        # Include cumulative summary if available
        if cumulative_summary:
            summary_section = f"\nOVERALL VIDEO THEME:\n{cumulative_summary}\n"
        else:
            summary_section = ""

        # Build the enhanced prompt with chunk summaries (not truncated transcript)
        prompt = f"""You are an expert content strategist for short-form video (Reels/TikTok/Shorts).

GOAL: Create a SHORT reel of exactly {min_duration}-{max_duration} seconds total.

Each chunk below is ~2 minutes long. You must select SHORT PORTIONS (15-30 seconds each) from within these chunks, NOT entire chunks.
{summary_section}
{chunk_summaries}
{suggestions_section}
CRITICAL DURATION CONSTRAINT:
- Target: {min_duration}-{max_duration} seconds TOTAL (NOT minutes!)
- Each part should be 15-30 seconds
- Example: 3 parts of 20 seconds each = 60 seconds total
- DO NOT select entire 2-minute chunks!

SELECTION CRITERIA:
1. STANDALONE VALUE - Must make sense without watching the full video
2. ENGAGING CONTENT - Educational, entertaining, or insightful
3. COMPLETENESS - Tells a complete micro-story

EXAMPLE OUTPUT:
PARTS:
1. [8:15 - 8:35] Explains the key concept clearly (20s)
2. [12:40 - 13:00] Gives practical example (20s)
3. [18:30 - 18:50] Summarizes the insight (20s)

TOTAL_DURATION: 60s

YOUR OUTPUT (use this exact format):

PARTS:
1. [MM:SS - MM:SS] Reason (duration in seconds)
2. [MM:SS - MM:SS] Reason (if needed)
3. [MM:SS - MM:SS] Reason (if needed)

TOTAL_DURATION: Xs

NARRATIVE: Why these specific short segments work well together.

TITLE: Engaging title
"""

        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "aya-expanse:8b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 600
                }
            },
            timeout=120
        )

        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code}")

        llm_response = response.json().get('response', '')

        # Parse the response
        result = parse_llm_response(llm_response)

        return result

    except Exception as e:
        print(f"⚠️  LLM analysis failed: {e}")
        return None


def parse_llm_response(llm_response: str) -> Dict:
    """
    Parse the LLM's response into structured data

    Args:
        llm_response: Raw text from Ollama

    Returns:
        Dictionary with 'parts', 'total_duration', 'narrative', 'title'
    """
    result = {
        'parts': [],
        'total_duration': 0,
        'narrative': '',
        'title': ''
    }

    lines = llm_response.split('\n')
    current_section = None

    for line in lines:
        line_stripped = line.strip()

        # Detect sections
        if 'PARTS:' in line_stripped.upper():
            current_section = 'parts'
            continue
        elif 'TOTAL_DURATION:' in line_stripped.upper():
            # Extract duration
            duration_match = re.search(r'(\d+)s', line_stripped)
            if duration_match:
                result['total_duration'] = int(duration_match.group(1))
            current_section = None
            continue
        elif 'NARRATIVE:' in line_stripped.upper():
            current_section = 'narrative'
            continue
        elif 'TITLE:' in line_stripped.upper():
            current_section = 'title'
            continue

        # Parse content
        if not line_stripped:
            continue

        if current_section == 'parts':
            # Parse part: "1. [MM:SS - MM:SS] reason" or "[MM:SS - MM:SS] reason"
            timestamp_match = re.search(r'\[?(\d+:\d+\.?\d*\s*-\s*\d+:\d+\.?\d*)\]?', line_stripped)
            if timestamp_match:
                timestamp = timestamp_match.group(1).strip()
                reason = line_stripped.split(timestamp, 1)[-1].strip()
                if reason.startswith(']'):
                    reason = reason[1:].strip()
                if reason.startswith('-'):
                    reason = reason[1:].strip()

                result['parts'].append({
                    'time_range': timestamp,
                    'reason': reason if reason else 'Selected segment'
                })

        elif current_section == 'narrative':
            result['narrative'] += line_stripped + ' '

        elif current_section == 'title':
            result['title'] += line_stripped + ' '

    # Clean up
    result['narrative'] = result['narrative'].strip()
    result['title'] = result['title'].strip()

    return result


def validate_and_calculate_duration(parts: List[Dict]) -> Tuple[bool, int]:
    """
    Validate reel parts and calculate total duration

    Args:
        parts: List of parts with 'time_range'

    Returns:
        Tuple of (is_valid, total_duration_seconds)
    """
    total_duration = 0

    try:
        for part in parts:
            time_range = part['time_range']
            # Parse the time range
            start, end = parse_time_range(time_range)
            duration = end - start
            total_duration += duration

        return True, int(total_duration)

    except Exception as e:
        print(f"⚠️  Duration validation failed: {e}")
        return False, 0


def validate_llm_selection(llm_result: Dict, min_duration: int = 45, max_duration: int = 70) -> Tuple[bool, str]:
    """
    Validate LLM selection before cutting video

    Checks:
    1. Has valid parts
    2. All parts have parseable timestamps
    3. Duration is within range

    Args:
        llm_result: Result from analyze_with_llm()
        min_duration: Minimum acceptable duration
        max_duration: Maximum acceptable duration

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check for valid parts
    if not llm_result or 'parts' not in llm_result:
        return False, "No parts in LLM result"

    parts = llm_result['parts']
    if not parts or len(parts) == 0:
        return False, "Empty parts list"

    # Validate each part has parseable timestamp
    for i, part in enumerate(parts, 1):
        if 'time_range' not in part:
            return False, f"Part {i} missing time_range"

        try:
            start, end = parse_time_range(part['time_range'])
            if start >= end:
                return False, f"Part {i} has invalid range: start ({start}) >= end ({end})"
            if end - start < 5:
                return False, f"Part {i} is too short: {end - start:.1f}s (minimum 5s)"
        except Exception as e:
            return False, f"Part {i} has unparseable timestamp '{part['time_range']}': {e}"

    # Check total duration
    is_valid, total_duration = validate_and_calculate_duration(parts)
    if not is_valid:
        return False, "Could not calculate total duration"

    if total_duration < min_duration:
        return False, f"Total duration {total_duration}s is below minimum {min_duration}s"

    if total_duration > max_duration + 10:  # Allow 10s grace period
        return False, f"Total duration {total_duration}s exceeds maximum {max_duration}s by too much"

    return True, "Valid selection"


def generate_reel(video_path: str, parts: List[Dict], output_path: str, use_ffmpeg: bool = True):
    """
    Generate reel by cutting and concatenating video segments

    Args:
        video_path: Path to input video
        parts: List of parts with 'time_range'
        output_path: Path for output reel
        use_ffmpeg: Use FFmpeg (faster) vs MoviePy
    """
    print(f"\n🎬 Generating reel from {len(parts)} part(s)...")

    # Convert parts to time_ranges
    time_ranges = []
    for i, part in enumerate(parts, 1):
        try:
            start, end = parse_time_range(part['time_range'])
            time_ranges.append((start, end))
            print(f"  Part {i}: {format_time(start)} - {format_time(end)} ({end-start:.1f}s)")
        except Exception as e:
            raise ValueError(f"Invalid time range in part {i}: {part['time_range']} - {e}")

    # Generate reel
    if use_ffmpeg:
        print("\n🔧 Using FFmpeg for fast processing...")
        cut_segments_ffmpeg(video_path, time_ranges, output_path)
    else:
        print("\n🔧 Using MoviePy for processing...")
        cut_segments_moviepy(video_path, time_ranges, output_path)


def find_latest_results_dir() -> str:
    """
    Find the most recent results directory

    Returns:
        Path to latest results directory
    """
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    results_dir = PROJECT_ROOT / "results"

    if not results_dir.exists():
        raise FileNotFoundError("No results directory found")

    # Get all subdirectories
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]

    if not subdirs:
        raise FileNotFoundError("No transcription results found in results/")

    # Sort by modification time and return the latest
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)

    return str(latest)


def extract_video_path_from_results(results_dir: str) -> str:
    """
    Extract original video path from results metadata

    Args:
        results_dir: Results directory path

    Returns:
        Path to original video file
    """
    # Check chunk metadata files for video path
    for filename in os.listdir(results_dir):
        if filename.startswith("chunk_") and filename.endswith("_metadata.txt"):
            metadata_path = os.path.join(results_dir, filename)
            with open(metadata_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for video path (not implemented yet in transcribe_advanced.py)
            # For now, try to infer from directory name
            break

    # Infer from directory name (format: YYYY-MM-DD_HHMMSS_VideoName)
    dir_name = os.path.basename(results_dir)
    parts = dir_name.split('_')

    if len(parts) >= 3:
        video_name = '_'.join(parts[2:])  # Everything after date and time

        # Search for video in common locations
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        search_paths = [
            PROJECT_ROOT / "data" / f"{video_name}.MP4",
            PROJECT_ROOT / "data" / f"{video_name}.MOV",
            PROJECT_ROOT / "data" / f"{video_name}.mp4",
            PROJECT_ROOT / "data" / f"{video_name}.mov",
        ]

        for path in search_paths:
            if path.exists():
                return str(path)

    raise FileNotFoundError(f"Could not find original video for results: {results_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Automatically generate optimal 45-70 second reels from transcription results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standalone mode
  python src/scripts/generate_auto_reel.py \\
    --results-dir results/2025-01-17_221415_IMG_4314 \\
    --video data/IMG_4314.MP4

  # Interactive mode (auto-detect latest)
  python src/scripts/generate_auto_reel.py

  # With custom duration range
  python src/scripts/generate_auto_reel.py \\
    --results-dir results/2025-01-17_221415_IMG_4314 \\
    --video data/IMG_4314.MP4 \\
    --min-duration 50 --max-duration 60
        """
    )

    parser.add_argument(
        '--results-dir', '-r',
        help='Path to transcription results directory',
        type=str
    )

    parser.add_argument(
        '--video', '-v',
        help='Path to original video file',
        type=str
    )

    parser.add_argument(
        '--min-duration',
        help='Minimum reel duration in seconds (default: 45)',
        type=int,
        default=45
    )

    parser.add_argument(
        '--max-duration',
        help='Maximum reel duration in seconds (default: 70)',
        type=int,
        default=70
    )

    parser.add_argument(
        '--use-moviepy',
        help='Use MoviePy instead of FFmpeg (slower)',
        action='store_true'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("AUTOMATED REEL GENERATOR")
    print("=" * 80)

    # Determine results directory
    if args.results_dir:
        results_dir = args.results_dir
    else:
        print("\n🔍 Finding latest transcription results...")
        try:
            results_dir = find_latest_results_dir()
            print(f"✅ Found: {results_dir}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)

    # Determine video path
    if args.video:
        video_path = args.video
    else:
        print("\n🔍 Looking for original video...")
        try:
            video_path = extract_video_path_from_results(results_dir)
            print(f"✅ Found: {video_path}")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            print("\n💡 Please specify video path with --video")
            sys.exit(1)

    # Parse AI summary
    print("\n📄 Parsing AI analysis...")
    ai_summary_path = os.path.join(results_dir, "ai_summary.txt")

    try:
        ai_data = parse_ai_summary(ai_summary_path)
        print(f"✅ Found {len(ai_data['suggestions'])} reel suggestions")

        if len(ai_data['suggestions']) == 0:
            print("⚠️  No pre-defined reel suggestions found")
            print("💡 Will analyze full transcript directly with LLM")

    except FileNotFoundError:
        print("❌ AI summary not found (ai_summary.txt)")
        print("💡 Run transcription with Ollama enabled first")
        sys.exit(1)

    # Get chunk summaries (these have timestamps and are already LLM-analyzed)
    print("\n📖 Extracting chunk summaries with timestamps...")
    chunk_summaries = get_chunk_summaries_from_ai_analysis(results_dir)
    if chunk_summaries:
        # Count chunks
        chunk_count = chunk_summaries.count("Chunk ")
        print(f"✅ Loaded {chunk_count} chunk summaries with timestamps")
    else:
        print("⚠️  Could not load chunk summaries, will use suggestions only")

    # Analyze with LLM
    print(f"\n🤖 Analyzing with Ollama to find best {args.min_duration}-{args.max_duration}s reel...")
    print("⏳ This may take 30-60 seconds...")

    # Try LLM analysis with validation and retry
    MAX_ATTEMPTS = 3
    llm_result = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if attempt > 1:
            print(f"\n🔄 Retry attempt {attempt}/{MAX_ATTEMPTS}...")

        llm_result = analyze_with_llm(
            chunk_summaries,
            ai_data['suggestions'],
            ai_data.get('cumulative_summary', ''),
            args.min_duration,
            args.max_duration
        )

        if not llm_result or not llm_result.get('parts'):
            print(f"⚠️  Attempt {attempt}: LLM returned no parts")
            continue

        # Validate the selection
        is_valid, error_msg = validate_llm_selection(
            llm_result,
            args.min_duration,
            args.max_duration
        )

        if is_valid:
            print(f"✅ Selection validated on attempt {attempt}")
            break
        else:
            print(f"⚠️  Attempt {attempt} validation failed: {error_msg}")
            llm_result = None

    # If all attempts failed, try fallback
    if not llm_result or not llm_result.get('parts'):
        print("\n❌ All LLM attempts failed")

        # Fallback to first suggestion (if available)
        if ai_data['suggestions'] and len(ai_data['suggestions']) > 0:
            print("💡 Falling back to first suggestion...")
            llm_result = {
                'parts': [ai_data['suggestions'][0]],
                'narrative': 'Using first AI suggestion',
                'title': 'Auto-generated Reel'
            }
        else:
            print("❌ No suggestions available and LLM failed")
            print("💡 Please ensure Ollama is running: ollama serve")
            sys.exit(1)

    # Display result
    print("\n" + "=" * 80)
    print("🎯 SELECTED REEL")
    print("=" * 80)

    print(f"\n📹 Title: {llm_result['title']}")
    print(f"📝 Narrative: {llm_result['narrative']}\n")

    print(f"🎬 Parts ({len(llm_result['parts'])}):")
    for i, part in enumerate(llm_result['parts'], 1):
        print(f"  {i}. {part['time_range']} - {part['reason']}")

    # Validate duration
    is_valid, actual_duration = validate_and_calculate_duration(llm_result['parts'])

    if is_valid:
        print(f"\n⏱️  Total Duration: {actual_duration}s")

        if args.min_duration <= actual_duration <= args.max_duration:
            print(f"✅ Within target range ({args.min_duration}-{args.max_duration}s)")
        else:
            print(f"⚠️  Outside target range ({args.min_duration}-{args.max_duration}s), but proceeding...")
    else:
        print("❌ Duration validation failed")
        sys.exit(1)

    # Generate output path
    output_dir = ensure_output_dir()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_AUTO_REEL.MP4")

    # Check if file exists and create unique name
    counter = 1
    while os.path.exists(output_path):
        counter += 1
        output_path = os.path.join(output_dir, f"{video_name}_AUTO_REEL_{counter}.MP4")

    # Generate reel
    try:
        generate_reel(
            video_path,
            llm_result['parts'],
            output_path,
            use_ffmpeg=not args.use_moviepy
        )

        print(f"\n✅ Reel generated successfully!")
        print(f"📁 Output: {output_path}")

        # Save metadata
        metadata_path = output_path.replace('.MP4', '_metadata.txt')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("AUTO-GENERATED REEL METADATA\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Source Video: {video_path}\n")
            f.write(f"Transcription Results: {results_dir}\n")
            f.write(f"Duration: {actual_duration}s\n")
            f.write(f"Parts: {len(llm_result['parts'])}\n\n")
            f.write(f"Title: {llm_result['title']}\n")
            f.write(f"Narrative: {llm_result['narrative']}\n\n")
            f.write("SEGMENTS:\n")
            for i, part in enumerate(llm_result['parts'], 1):
                f.write(f"{i}. {part['time_range']} - {part['reason']}\n")

        print(f"📄 Metadata: {metadata_path}")

    except Exception as e:
        print(f"\n❌ Reel generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
