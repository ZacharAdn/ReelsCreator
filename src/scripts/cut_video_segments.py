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
import shutil

def check_ffmpeg_installed():
    """Check if FFmpeg is installed and accessible"""
    if shutil.which('ffmpeg') is None:
        print("❌ FFmpeg is not installed or not in PATH")
        print("\nPlease install FFmpeg:")
        print("  macOS: brew install ffmpeg")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        sys.exit(1)

def check_ffprobe_installed():
    """Check if ffprobe is installed (needed for rotation detection)"""
    if shutil.which('ffprobe') is None:
        print("⚠️  Warning: ffprobe not found (rotation detection disabled)")

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

def get_unique_output_path(output_dir: str, video_name: str) -> str:
    """
    Generate unique output path, adding _2, _3, etc. if file exists

    Args:
        output_dir: Directory for output file
        video_name: Base video name (without extension)

    Returns:
        Unique output file path
    """
    base_output_path = os.path.join(output_dir, f"{video_name}_REEL.MP4")

    # If file doesn't exist, return the base path
    if not os.path.exists(base_output_path):
        return base_output_path

    # File exists, find next available number
    counter = 2
    while True:
        new_output_path = os.path.join(output_dir, f"{video_name}_REEL_{counter}.MP4")
        if not os.path.exists(new_output_path):
            return new_output_path
        counter += 1

def get_video_rotation(video_path: str) -> int:
    """
    Get rotation angle from video metadata

    Args:
        video_path: Path to video file

    Returns:
        Rotation angle in degrees (0, 90, 180, 270)
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream_side_data=rotation',
             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            rotation = int(float(result.stdout.strip()))
            # Normalize to positive angle
            return abs(rotation) % 360
    except:
        pass

    return 0

def cut_segments_moviepy(video_path: str, time_ranges: List[Tuple[float, float]], output_path: str):
    """
    Cut and concatenate video segments using MoviePy

    Args:
        video_path: Path to input video
        time_ranges: List of (start, end) tuples in seconds
        output_path: Path for output video
    """
    print(f"\n🎬 Loading video: {video_path}")

    # Check for rotation metadata
    rotation = get_video_rotation(video_path)
    if rotation != 0:
        print(f"⚠️  Warning: Video has rotation metadata ({rotation}°)")
        print(f"💡 Recommendation: Use FFmpeg mode (--use-ffmpeg) to preserve rotation")
        print(f"   Continuing with MoviePy, but rotation may not be preserved...\n")

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
    final_video = concatenate_videoclips(clips)

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
        logger=None,
        preset='medium',
        ffmpeg_params=['-pix_fmt', 'yuv420p']
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

    # Concatenate all segments (video and audio must be interleaved)
    concat_inputs = ''.join(f"[v{i}][a{i}]" for i in range(len(time_ranges)))
    concat_filter = f"{concat_inputs}concat=n={len(time_ranges)}:v=1:a=1[outv][outa]"

    # Combine all filters
    filter_complex = ';'.join(video_filters + audio_filters + [concat_filter])

    # Build FFmpeg command (with metadata copy to preserve rotation)
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-filter_complex', filter_complex,
        '-map', '[outv]',
        '-map', '[outa]',
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-map_metadata', '0',  # Copy metadata from input
        '-movflags', '+faststart',  # Enable fast start for web playback
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

def get_video_info(video_path: str) -> dict:
    """
    Get video metadata using MoviePy

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with duration, size, and modified date
    """
    from datetime import datetime

    video = VideoFileClip(video_path)
    duration = video.duration
    video.close()

    file_stat = os.stat(video_path)
    file_size_mb = file_stat.st_size / (1024 * 1024)
    modified_date = datetime.fromtimestamp(file_stat.st_mtime).strftime("%Y-%m-%d")

    return {
        'duration': duration,
        'size_mb': file_size_mb,
        'modified_date': modified_date
    }

def scan_directory_for_videos(directory: str) -> List[dict]:
    """
    Scan directory for video files and get their metadata

    Args:
        directory: Directory to scan

    Returns:
        List of video info dictionaries
    """
    video_extensions = ('.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI', '.mkv', '.MKV')
    videos = []

    if not os.path.exists(directory):
        return []

    for filename in os.listdir(directory):
        if filename.endswith(video_extensions):
            video_path = os.path.join(directory, filename)
            try:
                info = get_video_info(video_path)
                videos.append({
                    'path': video_path,
                    'name': filename,
                    'duration': info['duration'],
                    'size_mb': info['size_mb'],
                    'modified_date': info['modified_date']
                })
            except Exception as e:
                print(f"⚠️  Warning: Could not read {filename}: {e}")

    return videos

def find_directories_with_videos(base_path: str = ".") -> List[str]:
    """
    Find all directories in the project that contain video files

    Args:
        base_path: Base path to search from (default: current directory)

    Returns:
        List of directory paths containing videos
    """
    video_extensions = ('.mp4', '.MP4', '.mov', '.MOV', '.avi', '.AVI', '.mkv', '.MKV')
    directories_with_videos = []

    # Walk through all directories
    for root, dirs, files in os.walk(base_path):
        # Skip hidden directories and common excludes
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', 'reels_extractor_env', '__pycache__']]

        # Check if this directory contains any video files
        has_videos = any(filename.endswith(video_extensions) for filename in files)

        if has_videos:
            directories_with_videos.append(root)

    return sorted(directories_with_videos)

def display_video_list(videos: List[dict]):
    """
    Display formatted list of videos with metadata

    Args:
        videos: List of video info dictionaries
    """
    print("\n" + "=" * 80)
    print("📹 Available Videos")
    print("=" * 80)

    for i, video in enumerate(videos, 1):
        duration_str = format_time(video['duration'])
        print(f"\n{i}. {video['name']}")
        print(f"   Duration: {duration_str} | Size: {video['size_mb']:.1f}MB | Date: {video['modified_date']}")

    print("\n" + "=" * 80)

def interactive_mode():
    """
    Run the script in interactive mode, prompting for inputs
    """
    print("=" * 80)
    print("VIDEO SEGMENT CUTTER - Interactive Mode")
    print("=" * 80)

    # Find all directories with videos
    print("\n🔍 Scanning project for video files...")
    video_dirs = find_directories_with_videos(".")

    if not video_dirs:
        print("\n❌ No directories with video files found in project")
        sys.exit(1)

    # Display directory options
    print("\n" + "=" * 80)
    print("📁 Directories with videos:")
    print("=" * 80)
    for i, directory in enumerate(video_dirs, 1):
        # Count videos in this directory
        videos_in_dir = scan_directory_for_videos(directory)
        print(f"\n{i}. {directory}")
        print(f"   ({len(videos_in_dir)} video{'s' if len(videos_in_dir) != 1 else ''})")
    print("\n" + "=" * 80)

    # Get directory selection
    print("\nSelect directory number (or press Enter to cancel):")
    selection = input("> ").strip()

    if not selection:
        print("Cancelled.")
        sys.exit(0)

    try:
        dir_index = int(selection) - 1
        if dir_index < 0 or dir_index >= len(video_dirs):
            print(f"❌ Invalid number. Choose between 1 and {len(video_dirs)}")
            sys.exit(1)
    except ValueError:
        print("❌ Please enter a valid number")
        sys.exit(1)

    selected_dir = video_dirs[dir_index]
    print(f"\n✅ Selected directory: {selected_dir}")

    # Scan selected directory for videos
    videos = scan_directory_for_videos(selected_dir)

    if not videos:
        print(f"\n❌ No video files found in {selected_dir}")
        sys.exit(1)

    # Display video list
    display_video_list(videos)

    # Get video selection
    print("\nSelect video number (or press Enter to cancel):")
    selection = input("> ").strip()

    if not selection:
        print("Cancelled.")
        sys.exit(0)

    try:
        video_index = int(selection) - 1
        if video_index < 0 or video_index >= len(videos):
            print(f"❌ Invalid number. Choose between 1 and {len(videos)}")
            sys.exit(1)
    except ValueError:
        print("❌ Please enter a valid number")
        sys.exit(1)

    video_path = videos[video_index]['path']
    print(f"\n✅ Selected: {videos[video_index]['name']}")
    print(f"📊 Duration: {format_time(videos[video_index]['duration'])} | Size: {videos[video_index]['size_mb']:.1f}MB")

    # Get time ranges interactively
    print("\n" + "=" * 80)
    print("Enter time ranges to extract")
    print("=" * 80)
    print("Option 1: Paste all ranges at once (one per line, then type 'done')")
    print("Option 2: Enter ranges one by one (press Enter after each)")
    print("\nFormat: MM:SS.MS-MM:SS.MS or MM:SS.MS - MM:SS.MS")
    print("Example: 3:45.94-3:53.82 or 3:45.94 - 3:53.82")
    print("=" * 80)

    time_ranges = []
    segment_num = 1
    bulk_mode = False
    bulk_input_lines = []

    while True:
        if not bulk_mode:
            print(f"\nRange #{segment_num} (or press Enter to finish):")
        range_input = input("> ").strip()

        # Check if user wants to finish bulk input
        if bulk_mode and range_input.lower() == 'done':
            break

        # Empty input handling
        if not range_input:
            if len(time_ranges) == 0 and not bulk_mode:
                print("❌ Please enter at least one range")
                continue
            elif bulk_mode:
                # Empty line in bulk mode, continue collecting
                continue
            else:
                # User pressed Enter to finish
                break

        # Try to parse as a time range
        try:
            time_range = parse_time_range(range_input)
            time_ranges.append(time_range)
            start_str = format_time(time_range[0])
            end_str = format_time(time_range[1])
            duration = time_range[1] - time_range[0]
            print(f"   ✓ Added: {start_str} - {end_str} ({duration:.1f}s)")
            segment_num += 1

            # If this is the first valid range and there are more lines coming,
            # we might be in bulk paste mode - activate it
            if segment_num == 2 and not bulk_mode:
                bulk_mode = True
                print("\n💡 Bulk mode activated! Paste remaining ranges (type 'done' when finished)")

        except ValueError as e:
            # If it's the first input and parsing failed, might be starting bulk mode
            if segment_num == 1 and not bulk_mode:
                print(f"   ⚠️  Invalid format. Please use: MM:SS.MS-MM:SS.MS")
            else:
                print(f"   ❌ Error: {e}")
                print("   Skipping this line...")

    # Summary
    total_duration = sum(end - start for start, end in time_ranges)
    print(f"\n📊 Total: {len(time_ranges)} ranges | Total duration: {format_time(total_duration)}")

    # Generate unique output path
    output_dir = ensure_output_dir()
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = get_unique_output_path(output_dir, video_name)

    # Check for rotation and recommend FFmpeg
    rotation = get_video_rotation(video_path)
    if rotation != 0:
        print(f"\n⚠️  Video has rotation metadata ({rotation}°)")
        print("💡 FFmpeg mode is recommended to preserve rotation correctly")
        print("\nUse FFmpeg? (y/n, default: y):")
        use_ffmpeg = input("> ").strip().lower() != 'n'
    else:
        print("\nUse FFmpeg for faster processing? (y/n, default: n):")
        use_ffmpeg = input("> ").strip().lower() == 'y'

    # Process video
    if use_ffmpeg:
        cut_segments_ffmpeg(video_path, time_ranges, output_path)
    else:
        cut_segments_moviepy(video_path, time_ranges, output_path)

def main():
    """Main entry point"""
    # Check if FFmpeg is installed
    check_ffmpeg_installed()
    check_ffprobe_installed()

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
        output_path = get_unique_output_path(output_dir, video_name)

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
