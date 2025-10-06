#!/usr/bin/env python3
"""
Video Segment Cutter - Extract and concatenate specific time ranges from videos

Usage:
    # Interactive mode
    python cut_video_segments.py

    # Command-line mode
    python cut_video_segments.py --video data/IMG_4225.MP4 --ranges "1:00.26-1:07.16,1:27.64-1:31.72"

    # With custom output
    python cut_video_segments.py --video data/IMG_4225.MP4 --ranges "1:00-2:00" --output my_reel.mp4

    # Using FFmpeg (faster, no re-encoding)
    python cut_video_segments.py --video data/IMG_4225.MP4 --ranges "1:00-2:00" --use-ffmpeg

Input format:
    Time ranges: MM:SS.MS-MM:SS.MS or M:SS.MS-M:SS.MS
    Example: "1:00.26-1:07.16, 1:27.64-1:31.72, 1:42.30-1:49.04"

Output:
    generated_data/VideoName_REEL.MP4
"""

import os
import sys
import argparse
import re
from typing import List, Tuple
from moviepy.editor import VideoFileClip, concatenate_videoclips
import subprocess

def parse_timestamp(timestamp: str) -> float:
    """
    Convert timestamp string to seconds

    Supports formats:
        - MM:SS.MS (e.g., "01:23.45" → 83.45)
        - M:SS.MS (e.g., "1:23.45" → 83.45)
        - SS.MS (e.g., "23.45" → 23.45)
        - MM:SS (e.g., "01:23" → 83.0)

    Args:
        timestamp: Time string in format MM:SS.MS or variations

    Returns:
        Time in seconds as float
    """
    timestamp = timestamp.strip()

    # Pattern: [M]M:SS[.MS] or SS[.MS]
    if ':' in timestamp:
        parts = timestamp.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    else:
        # Just seconds
        return float(timestamp)

def parse_time_range(range_str: str) -> Tuple[float, float]:
    """
    Parse a time range string into start and end seconds

    Args:
        range_str: String like "1:00.26-1:07.16" or "60.26-67.16"

    Returns:
        Tuple of (start_seconds, end_seconds)
    """
    range_str = range_str.strip()

    # Split by dash, handling possible spaces
    parts = re.split(r'\s*-\s*', range_str)

    if len(parts) != 2:
        raise ValueError(f"Invalid range format: '{range_str}'. Expected format: 'start-end'")

    start = parse_timestamp(parts[0])
    end = parse_timestamp(parts[1])

    if start >= end:
        raise ValueError(f"Start time ({start}s) must be before end time ({end}s)")

    return (start, end)

def parse_ranges(ranges_str: str) -> List[Tuple[float, float]]:
    """
    Parse multiple time ranges from a comma-separated string

    Args:
        ranges_str: String like "1:00.26-1:07.16, 1:27.64-1:31.72, 1:42.30-1:49.04"

    Returns:
        List of (start, end) tuples in seconds
    """
    ranges = []
    for range_str in ranges_str.split(','):
        range_str = range_str.strip()
        if range_str:
            ranges.append(parse_time_range(range_str))

    return ranges

def format_time(seconds: float) -> str:
    """
    Convert seconds to MM:SS.MS format for display

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"

def ensure_output_dir():
    """
    Create generated_data directory if it doesn't exist

    Returns:
        Path to generated_data directory
    """
    output_dir = os.path.abspath("generated_data")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"📁 Created output directory: {output_dir}")

    return output_dir

def cut_segments_moviepy(video_path: str, time_ranges: List[Tuple[float, float]], output_path: str):
    """
    Cut and concatenate video segments using MoviePy

    Args:
        video_path: Path to input video
        time_ranges: List of (start, end) tuples in seconds
        output_path: Path for output video
    """
    print(f"\n🎬 Loading video: {video_path}")
    video = VideoFileClip(video_path)
    video_duration = video.duration

    print(f"📹 Video duration: {format_time(video_duration)}")
    print(f"✂️  Extracting {len(time_ranges)} segments...\n")

    # Validate and extract segments
    clips = []
    for i, (start, end) in enumerate(time_ranges, 1):
        if start < 0 or end > video_duration:
            raise ValueError(
                f"Segment {i} ({format_time(start)}-{format_time(end)}) "
                f"is outside video duration ({format_time(video_duration)})"
            )

        print(f"  Segment {i}: {format_time(start)} - {format_time(end)} ({end-start:.1f}s)")

        subclip = video.subclip(start, end)
        clips.append(subclip)

    # Concatenate all segments
    print(f"\n🔗 Concatenating {len(clips)} segments...")
    final_video = concatenate_videoclips(clips, method="compose")

    # Calculate total duration
    total_duration = sum(end - start for start, end in time_ranges)
    print(f"📊 Total output duration: {format_time(total_duration)}")

    # Write output
    print(f"\n💾 Writing output to: {output_path}")
    print("⏳ This may take a while...")

    final_video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True,
        verbose=False,
        logger=None
    )

    # Cleanup
    video.close()
    final_video.close()

    print(f"\n✅ Done! Output saved to: {output_path}")

def cut_segments_ffmpeg(video_path: str, time_ranges: List[Tuple[float, float]], output_path: str):
    """
    Cut and concatenate video segments using FFmpeg (faster, no re-encoding)

    Args:
        video_path: Path to input video
        time_ranges: List of (start, end) tuples in seconds
        output_path: Path for output video
    """
    print(f"\n🎬 Using FFmpeg for fast processing: {video_path}")
    print(f"✂️  Extracting {len(time_ranges)} segments...\n")

    # Build filter_complex command
    video_filters = []
    audio_filters = []

    for i, (start, end) in enumerate(time_ranges):
        print(f"  Segment {i+1}: {format_time(start)} - {format_time(end)} ({end-start:.1f}s)")

        # Video trim
        video_filters.append(
            f"[0:v]trim={start}:{end},setpts=PTS-STARTPTS[v{i}]"
        )

        # Audio trim
        audio_filters.append(
            f"[0:a]atrim={start}:{end},asetpts=PTS-STARTPTS[a{i}]"
        )

    # Concatenate all segments
    v_inputs = ''.join(f"[v{i}]" for i in range(len(time_ranges)))
    a_inputs = ''.join(f"[a{i}]" for i in range(len(time_ranges)))

    concat_filter = f"{v_inputs}{a_inputs}concat=n={len(time_ranges)}:v=1:a=1[outv][outa]"

    # Combine all filters
    filter_complex = ';'.join(video_filters + audio_filters + [concat_filter])

    # Build FFmpeg command
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-map', '[outa]',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-y',  # Overwrite output file
        output_path
    ]

    print(f"\n🔗 Concatenating segments with FFmpeg...")
    print("⏳ This may take a while...")

    # Run FFmpeg
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\n❌ FFmpeg error:\n{result.stderr}")
        raise RuntimeError("FFmpeg processing failed")

    print(f"\n✅ Done! Output saved to: {output_path}")

def interactive_mode():
    """
    Run the script in interactive mode, prompting for inputs
    """
    print("=" * 60)
    print("VIDEO SEGMENT CUTTER - Interactive Mode")
    print("=" * 60)

    # Get video path
    print("\nEnter video path (e.g., data/IMG_4225.MP4):")
    video_path = input("> ").strip()

    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        sys.exit(1)

    # Get time ranges
    print("\nEnter time ranges (format: MM:SS.MS-MM:SS.MS, separated by commas):")
    print("Example: 1:00.26-1:07.16, 1:27.64-1:31.72, 1:42.30-1:49.04")
    ranges_str = input("> ").strip()

    try:
        time_ranges = parse_ranges(ranges_str)
    except ValueError as e:
        print(f"❌ Error parsing ranges: {e}")
        sys.exit(1)

    # Generate output path
    output_dir = ensure_output_dir()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_REEL.MP4")

    # Ask for FFmpeg option
    print("\nUse FFmpeg for faster processing? (y/n, default: n):")
    use_ffmpeg = input("> ").strip().lower() == 'y'

    # Process video
    if use_ffmpeg:
        cut_segments_ffmpeg(video_path, time_ranges, output_path)
    else:
        cut_segments_moviepy(video_path, time_ranges, output_path)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Cut and concatenate video segments from time ranges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python cut_video_segments.py

  # Command-line mode
  python cut_video_segments.py --video data/IMG_4225.MP4 --ranges "1:00.26-1:07.16,1:27.64-1:31.72"

  # Custom output
  python cut_video_segments.py --video data/video.mp4 --ranges "0:30-1:00" --output my_reel.mp4

  # Use FFmpeg (faster)
  python cut_video_segments.py --video data/video.mp4 --ranges "1:00-2:00" --use-ffmpeg
        """
    )

    parser.add_argument(
        '--video', '-v',
        help='Path to input video file',
        type=str
    )

    parser.add_argument(
        '--ranges', '-r',
        help='Time ranges to extract (format: MM:SS.MS-MM:SS.MS, comma-separated)',
        type=str
    )

    parser.add_argument(
        '--output', '-o',
        help='Output file path (default: generated_data/VideoName_REEL.MP4)',
        type=str
    )

    parser.add_argument(
        '--use-ffmpeg',
        help='Use FFmpeg instead of MoviePy (faster, no re-encoding)',
        action='store_true'
    )

    args = parser.parse_args()

    # If no arguments provided, run interactive mode
    if not args.video and not args.ranges:
        interactive_mode()
        return

    # Validate required arguments
    if not args.video or not args.ranges:
        parser.error("Both --video and --ranges are required (or use interactive mode)")

    # Check video exists
    if not os.path.exists(args.video):
        print(f"❌ Error: Video file not found: {args.video}")
        sys.exit(1)

    # Parse time ranges
    try:
        time_ranges = parse_ranges(args.ranges)
    except ValueError as e:
        print(f"❌ Error parsing ranges: {e}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
        # Create parent directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = ensure_output_dir()
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_path = os.path.join(output_dir, f"{video_name}_REEL.MP4")

    # Process video
    try:
        if args.use_ffmpeg:
            cut_segments_ffmpeg(args.video, time_ranges, output_path)
        else:
            cut_segments_moviepy(args.video, time_ranges, output_path)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
