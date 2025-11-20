#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED Hebrew-optimized transcription script using the best models from the full pipeline

Features:
- Hebrew-optimized models with automatic fallback
- Multi-chunk processing for long videos with proper timestamp adjustment
- Progress tracking and detailed output
- Multiple model support (Hugging Face wav2vec2 → Whisper large-v3-turbo → Whisper large)

Configuration:
- CHUNK_SIZE_MINUTES: Set the size of processing chunks (default: 2 minutes)
  - For longer videos, adjust this value at the top of the script
  - Smaller chunks (1-2 minutes) = More frequent progress updates
  - Larger chunks (5+ minutes) = Potentially faster overall processing
"""

import whisper
import os
import sys
from moviepy import VideoFileClip
import tempfile
from datetime import datetime
import time
import math
from transformers import pipeline
from huggingface_hub import hf_hub_download
from typing import List, Dict

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Configuration
CHUNK_SIZE_MINUTES = 2  # Easily change this value to adjust chunk size
CHUNK_SIZE_SECONDS = CHUNK_SIZE_MINUTES * 60  # Convert to seconds

def load_optimal_model():
    """
    Load the best available model with Hebrew optimization from Hugging Face
    """
    try:
        print("🇮🇱 Loading Hebrew-optimized model from Hugging Face...")
        transcriber = pipeline(
            "automatic-speech-recognition",
            model="imvladikon/wav2vec2-large-xlsr-53-hebrew",
            device="auto"
        )
        print("✅ Hebrew-optimized model loaded successfully!")
        return transcriber, "huggingface"
    except Exception as e:
        print(f"⚠️  Hebrew model failed ({e}), trying Whisper large-v3-turbo...")
        
        # Fallback to Whisper
        try:
            print("🚀 Loading large-v3-turbo...")
            model = whisper.load_model("large-v3-turbo")
            print("✅ Large-v3-turbo model loaded!")
            return model, "whisper"
        except Exception as e:
            print(f"⚠️  Turbo model failed, using large...")
            model = whisper.load_model("large")
            return model, "whisper"

def process_chunk(model, model_type, audio_path, start_time=0, duration=None):
    """
    Process a chunk of audio
    """
    chunk_start = time.time()
    
    if model_type == "huggingface":
        # Process with Hugging Face model
        result = model(audio_path, chunk_length_s=30)

        # Convert to whisper-like format
        segments = []
        text = ""
        current_time = start_time

        for segment in result["chunks"]:
            cleaned_text = clean_rtl_markers(segment)
            segment_dict = {
                'start': current_time,
                'end': current_time + 30,  # Approximate, as HF doesn't provide exact timestamps
                'text': cleaned_text
            }
            segments.append(segment_dict)
            text += cleaned_text + " "
            current_time += 30

        result = {
            'language': 'he',
            'duration': duration if duration else (current_time - start_time),
            'text': text.strip(),
            'segments': segments
        }
    else:
        # Standard whisper doesn't support start_time/duration, process full file
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            language="he"
        )

        # Clean RTL markers from Whisper output
        result['text'] = clean_rtl_markers(result['text'])
        for segment in result['segments']:
            segment['text'] = clean_rtl_markers(segment['text'])

        # If we need chunking for standard whisper, filter segments by time
        if start_time > 0 or duration is not None:
            end_time = start_time + duration if duration else float('inf')
            filtered_segments = []
            filtered_text = ""

            for segment in result['segments']:
                if segment['start'] >= start_time and segment['start'] < end_time:
                    # Adjust segment times relative to chunk start
                    adjusted_segment = segment.copy()
                    adjusted_segment['start'] = segment['start'] - start_time
                    adjusted_segment['end'] = segment['end'] - start_time
                    filtered_segments.append(adjusted_segment)
                    filtered_text += segment['text']

            result['segments'] = filtered_segments
            result['text'] = filtered_text
    
    chunk_time = time.time() - chunk_start
    return result, chunk_time

def ensure_results_dir(video_path):
    """
    Create timestamped results subdirectory with video filename
    Returns the path to the timestamped subdirectory

    Args:
        video_path: Path to the video file being processed
    """
    # Get base results directory (relative to project root)
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_results_dir = os.path.join(script_dir, "results")

    # Create base directory if needed
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)
        print(f"📁 Created base results directory: {base_results_dir}")

    # Extract video filename without extension and sanitize it
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Replace spaces and special chars with underscores
    video_name_clean = video_name.replace(" ", "_").replace(".", "-")

    # Create timestamped subdirectory with date first, then video filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dir_name = f"{timestamp}_{video_name_clean}"
    timestamped_dir = os.path.join(base_results_dir, dir_name)

    # Check if directory already exists (in case of very fast repeated runs)
    if os.path.exists(timestamped_dir):
        # Add milliseconds to make it unique
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        dir_name = f"{timestamp}_{video_name_clean}"
        timestamped_dir = os.path.join(base_results_dir, dir_name)

    os.makedirs(timestamped_dir)
    print(f"📁 Created timestamped output directory: {timestamped_dir}")

    return timestamped_dir

def clean_rtl_markers(text):
    """
    Remove RTL (Right-to-Left) control characters from text
    These are invisible Unicode characters that Whisper adds to Hebrew text
    """
    rtl_chars = [
        '\u202B',  # RIGHT-TO-LEFT EMBEDDING
        '\u202A',  # LEFT-TO-RIGHT EMBEDDING
        '\u202C',  # POP DIRECTIONAL FORMATTING
        '\u200F',  # RIGHT-TO-LEFT MARK
        '\u200E',  # LEFT-TO-RIGHT MARK
    ]
    cleaned = text
    for char in rtl_chars:
        cleaned = cleaned.replace(char, '')
    return cleaned

def format_timestamp(seconds):
    """
    Convert seconds to MM:SS format
    """
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:05.2f}"

def write_chunk_output(output_dir, chunk_num, chunk_result, chunk_time, chunk_start, chunk_duration, cumulative_text):
    """
    Write chunk output to files in real-time (after each 2-minute chunk)

    Args:
        output_dir: Directory to write outputs to
        chunk_num: Chunk number (1-indexed)
        chunk_result: Result dictionary from process_chunk
        chunk_time: Processing time for this chunk
        chunk_start: Start time in seconds
        chunk_duration: Duration of chunk in seconds
        cumulative_text: All text transcribed so far (including this chunk)
    """
    # Write individual chunk transcript
    chunk_file = os.path.join(output_dir, f"chunk_{chunk_num:02d}.txt")
    with open(chunk_file, 'w', encoding='utf-8') as f:
        f.write(f"CHUNK {chunk_num} TRANSCRIPT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Time range: {format_timestamp(chunk_start)} - {format_timestamp(chunk_start + chunk_duration)}\n")
        f.write(f"Processing time: {chunk_time:.1f} seconds ({chunk_time/60:.1f} minutes)\n\n")
        f.write("Transcript:\n")
        f.write("-" * 40 + "\n")
        f.write(chunk_result['text'] + "\n\n")
        f.write("Segments with timestamps:\n")
        f.write("-" * 40 + "\n")
        for segment in chunk_result['segments']:
            # Timestamps in chunk_result are already adjusted for chunk start
            start_time = segment['start'] + chunk_start
            end_time = segment['end'] + chunk_start
            text = segment['text'].strip()
            f.write(f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}] {text}\n")

    # Write chunk metadata
    metadata_file = os.path.join(output_dir, f"chunk_{chunk_num:02d}_metadata.txt")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(f"CHUNK {chunk_num} METADATA\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Chunk number: {chunk_num}\n")
        f.write(f"Start time: {format_timestamp(chunk_start)}\n")
        f.write(f"Duration: {chunk_duration:.1f} seconds ({chunk_duration/60:.1f} minutes)\n")
        f.write(f"Processing time: {chunk_time:.1f} seconds ({chunk_time/60:.1f} minutes)\n")
        f.write(f"Language: {chunk_result['language']}\n")
        f.write(f"Number of segments: {len(chunk_result['segments'])}\n")

    # Update cumulative full transcript
    full_transcript_file = os.path.join(output_dir, "full_transcript.txt")
    with open(full_transcript_file, 'w', encoding='utf-8') as f:
        f.write("CUMULATIVE FULL TRANSCRIPT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Last updated: Chunk {chunk_num}\n")
        f.write(f"Updated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Full transcript so far:\n")
        f.write("-" * 40 + "\n")
        f.write(cumulative_text)

    print(f"💾 Chunk {chunk_num} output saved to: {output_dir}")

def get_video_info(video_path: str) -> Dict:
    """
    Get video metadata using MoviePy

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with duration, size, and modified date
    """
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

def scan_directory_for_videos(directory: str) -> List[Dict]:
    """
    Scan a directory for video files and get their metadata

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

def display_video_list(videos: List[Dict]):
    """
    Display formatted list of videos with metadata

    Args:
        videos: List of video info dictionaries
    """
    print("\n" + "=" * 80)
    print("📹 Available Videos")
    print("=" * 80)

    for i, video in enumerate(videos, 1):
        duration_str = format_timestamp(video['duration'])
        print(f"\n{i}. {video['name']}")
        print(f"   Duration: {duration_str} | Size: {video['size_mb']:.1f}MB | Date: {video['modified_date']}")

    print("\n" + "=" * 80)

def interactive_mode():
    """
    Run interactive mode to select directory and video

    Returns:
        Path to selected video file
    """
    print("=" * 80)
    print("HEBREW VIDEO TRANSCRIPTION - Interactive Mode")
    print("=" * 80)

    # Find all directories with videos
    print("\n🔍 Scanning project for video files...")
    video_dirs = find_directories_with_videos(".")

    if not video_dirs:
        print("\n❌ No directories with video files found in project")
        return None

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
        return None

    try:
        dir_index = int(selection) - 1
        if dir_index < 0 or dir_index >= len(video_dirs):
            print(f"❌ Invalid number. Choose between 1 and {len(video_dirs)}")
            return None
    except ValueError:
        print("❌ Please enter a valid number")
        return None

    selected_dir = video_dirs[dir_index]
    print(f"\n✅ Selected directory: {selected_dir}")

    # Scan selected directory for videos
    videos = scan_directory_for_videos(selected_dir)

    if not videos:
        print(f"\n❌ No video files found in {selected_dir}")
        return None

    # Display video list
    display_video_list(videos)

    # Get video selection
    print("\nSelect video number (or press Enter to cancel):")
    selection = input("> ").strip()

    if not selection:
        print("Cancelled.")
        return None

    try:
        video_index = int(selection) - 1
        if video_index < 0 or video_index >= len(videos):
            print(f"❌ Invalid number. Choose between 1 and {len(videos)}")
            return None
    except ValueError:
        print("❌ Please enter a valid number")
        return None

    video_path = videos[video_index]['path']
    print(f"\n✅ Selected: {videos[video_index]['name']}")
    print(f"📊 Duration: {format_timestamp(videos[video_index]['duration'])} | Size: {videos[video_index]['size_mb']:.1f}MB")

    return video_path

def transcribe_video(video_path):
    """
    Extract transcription from video using configurable chunk sizes
    """
    print(f"Processing video: {video_path}")
    start_time_total = time.time()

    # Create timestamped output directory early (so chunk outputs can be written during processing)
    output_dir = ensure_results_dir(video_path)

    # Extract audio using moviepy
    print("Extracting audio from video...")
    video = VideoFileClip(video_path)
    video_duration = video.duration

    # Create temporary wav file
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio_path = temp_audio.name
    temp_audio.close()
    
    try:
        # Extract audio to temporary file
        video.audio.write_audiofile(temp_audio_path, logger=None)
        video.close()
        
        # Load optimal model with Hebrew support
        model, model_type = load_optimal_model()
        
        # Initialize results
        all_results = []
        total_text = ""
        
        # Check if video is longer than chunk size
        if video_duration > CHUNK_SIZE_SECONDS:
            num_chunks = math.ceil(video_duration / CHUNK_SIZE_SECONDS)
            print(f"\n🎬 Video is {video_duration/60:.1f} minutes long")
            print(f"📋 Processing in {num_chunks} chunks of {CHUNK_SIZE_MINUTES} minutes each...")
            
            for i in range(num_chunks):
                chunk_start = i * CHUNK_SIZE_SECONDS
                chunk_duration = min(CHUNK_SIZE_SECONDS, video_duration - chunk_start)

                print(f"\n🔄 Processing chunk {i+1}/{num_chunks}")
                print(f"⏱️  Time range: {format_timestamp(chunk_start)} - {format_timestamp(chunk_start + chunk_duration)}")

                result, chunk_time = process_chunk(model, model_type, temp_audio_path,
                                                 start_time=chunk_start, duration=chunk_duration)

                print(f"✅ Chunk {i+1} completed in {chunk_time:.1f} seconds")
                print("\nTranscript for this chunk:")
                print("-" * 40)
                print(result['text'])
                print("-" * 40)

                all_results.append((result, chunk_time))
                total_text += result['text'] + "\n\n"

                # Write chunk output in real-time (after every 2-minute chunk)
                write_chunk_output(
                    output_dir=output_dir,
                    chunk_num=i+1,
                    chunk_result=result,
                    chunk_time=chunk_time,
                    chunk_start=chunk_start,
                    chunk_duration=chunk_duration,
                    cumulative_text=total_text
                )
            
            # Combine results
            final_result = {
                'language': all_results[0][0]['language'],
                'duration': video_duration,
                'text': total_text,
                'segments': []
            }
            
            # Fix: Adjust timestamps for each chunk by adding the chunk's start time
            for i, (result, _) in enumerate(all_results):
                chunk_start_time = i * CHUNK_SIZE_SECONDS
                for segment in result['segments']:
                    adjusted_segment = segment.copy()
                    adjusted_segment['start'] += chunk_start_time
                    adjusted_segment['end'] += chunk_start_time
                    final_result['segments'].append(adjusted_segment)
            
            result = final_result
        else:
            print("\n🎤 Starting transcription...")
            result, chunk_time = process_chunk(model, model_type, temp_audio_path)
            print(f"✅ Transcription completed in {chunk_time:.1f} seconds")
        
        # Print results
        print("\n" + "="*60)
        print("TRANSCRIPTION RESULTS")
        print("="*60)
        
        print(f"Language detected: {result['language']}")
        duration = result.get('duration', 'N/A')
        print(f"Video Length: {format_timestamp(duration) if isinstance(duration, (int, float)) else duration}")
        
        print("\nFull transcript:")
        print("-" * 40)
        print(result['text'])
        
        print("\nSegments with timestamps:")
        print("-" * 40)
        for i, segment in enumerate(result['segments']):
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()
            print(f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}] {text}")
        
        # Save final summary to file with model info in the same timestamped directory
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        model_suffix = "hebrew" if model_type == "huggingface" else "whisper"
        output_filename = f"{video_name}_final_summary.txt"
        output_file = os.path.join(output_dir, output_filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("FINAL TRANSCRIPTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Model used: {model_suffix}\n")
            f.write(f"Language detected: {result['language']}\n")
            f.write(f"Video Length: {format_timestamp(duration) if isinstance(duration, (int, float)) else duration}\n")
            f.write(f"Output directory: {output_dir}\n\n")
            
            f.write("Full transcript:\n")
            f.write("-" * 40 + "\n")
            f.write(result['text'] + "\n\n")
            
            f.write("Segments with timestamps:\n")
            f.write("-" * 40 + "\n")
            for segment in result['segments']:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()
                f.write(f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}] {text}\n")
        
        # Calculate and print total processing time
        total_time = time.time() - start_time_total
        processing_minutes = total_time / 60
        print(f"\n⏱️  Total processing time: {total_time:.1f} seconds ({processing_minutes:.2f} minutes)")
        print(f"📁 All results saved to directory: {output_dir}")
        print(f"📄 Final summary: {output_file}")
        
        # Add processing time to the output file
        processing_minutes = total_time / 60
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\nPROCESSING INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total processing time: {total_time:.1f} seconds ({processing_minutes:.2f} minutes)\n")
            if video_duration > CHUNK_SIZE_SECONDS:  # If processed in chunks
                f.write(f"Processed in {math.ceil(video_duration/CHUNK_SIZE_SECONDS)} chunks of {CHUNK_SIZE_MINUTES} minutes each\n")
                for i, (_, chunk_time) in enumerate(all_results, 1):
                    f.write(f"Chunk {i} processing time: {chunk_time:.1f} seconds ({(chunk_time/60):.2f} minutes)\n")
        
        # Basic speaker information (Whisper doesn't provide advanced speaker detection)
        print("\nSPEAKER DETECTION NOTES:")
        print("- Whisper provides basic transcription but not advanced speaker diarization")
        print("- For advanced speaker detection, additional libraries like pyannote.audio are needed")
        print("- The segments above are based on natural speech pauses, not speaker changes")
        
        return result
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

if __name__ == "__main__":
    # Check for auto mode (batch processing)
    auto_video_path = os.environ.get('AUTO_VIDEO_PATH')

    if auto_video_path:
        # Auto mode - use provided video path
        video_file = auto_video_path
        print(f"🎬 Auto mode: Processing {os.path.basename(video_file)}")
    else:
        # Run in interactive mode
        video_file = interactive_mode()

        if not video_file:
            print("\n❌ No video selected")
            exit(1)

    # Process the selected video
    try:
        print("\n" + "=" * 80)
        print("Starting transcription...")
        print("=" * 80 + "\n")

        result = transcribe_video(video_file)

        print("\n🎉 Hebrew-optimized transcription completed successfully!")
        print("💡 This script uses the best Hebrew models:")
        print("   - wav2vec2-large-xlsr-53-hebrew (Hugging Face)")
        print("   - large-v3-turbo (Whisper fallback)")
        print("   - Automatic model fallback with proper error handling")

    except Exception as e:
        print(f"\n❌ Error during transcription: {str(e)}")
        exit(1)