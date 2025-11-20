#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch transcription script - automatically processes all videos in a directory
"""

import os
import sys
import subprocess

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

def get_videos_in_directory(directory):
    """Get all video files in a directory"""
    video_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV')
    videos = []

    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return []

    for file in os.listdir(directory):
        if file.endswith(video_extensions):
            videos.append(os.path.join(directory, file))

    return sorted(videos)

def check_already_transcribed(video_path, results_dir):
    """Check if a video has already been transcribed"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_name_clean = video_name.replace(" ", "_").replace(".", "-")

    if not os.path.exists(results_dir):
        return False

    # Look for any directory containing this video name and a final summary file
    for dirname in os.listdir(results_dir):
        if video_name_clean in dirname:
            summary_file = os.path.join(results_dir, dirname, f"{video_name_clean}_final_summary.txt")
            if os.path.exists(summary_file):
                return True

    return False

def main():
    # Configuration
    VIDEO_DIR = "nice data"
    SCRIPT_PATH = os.path.join("src", "quick scripts", "transcribe_advanced.py")
    PYTHON_PATH = os.path.join("reels_extractor_env", "Scripts", "python.exe")
    RESULTS_DIR = "results"

    # Get all videos
    videos = get_videos_in_directory(VIDEO_DIR)

    if not videos:
        print(f"❌ No videos found in '{VIDEO_DIR}'")
        return

    print(f"📹 Found {len(videos)} videos in '{VIDEO_DIR}'")
    print("="*80)

    # Process each video
    for i, video_path in enumerate(videos, 1):
        video_name = os.path.basename(video_path)
        print(f"\n[{i}/{len(videos)}] Processing: {video_name}")

        # Check if already transcribed
        if check_already_transcribed(video_path, RESULTS_DIR):
            print(f"✅ Already transcribed - skipping")
            continue

        print(f"🎬 Starting transcription...")

        # Create input file for the script (to simulate user selecting video)
        # We'll modify the approach: pass video path as environment variable
        env = os.environ.copy()
        env['AUTO_VIDEO_PATH'] = video_path

        try:
            # Run transcription script
            result = subprocess.run(
                [PYTHON_PATH, SCRIPT_PATH],
                env=env,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                print(f"✅ Completed: {video_name}")
            else:
                print(f"❌ Failed: {video_name}")
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"❌ Error processing {video_name}: {e}")

    print("\n" + "="*80)
    print("🎉 Batch processing complete!")

if __name__ == "__main__":
    main()
