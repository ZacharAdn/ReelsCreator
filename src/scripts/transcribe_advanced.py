#!/usr/bin/env python3
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
from moviepy.editor import VideoFileClip
import tempfile
from datetime import datetime
import time
import math
from transformers import pipeline
from huggingface_hub import hf_hub_download
from typing import List, Dict
from pathlib import Path
import re

# Configuration
CHUNK_SIZE_MINUTES = 2  # Easily change this value to adjust chunk size
CHUNK_SIZE_SECONDS = CHUNK_SIZE_MINUTES * 60  # Convert to seconds

# RTL marker pattern (compiled once for efficiency)
RTL_PATTERN = re.compile(r'[\u202B\u202A\u202C\u200F\u200E]')

def load_optimal_model():
    """
    Load the best available model with Hebrew optimization from Hugging Face
    """
    try:
        print("🇮🇱 Loading Hebrew-optimized model from Hugging Face...")

        # Detect available device
        import torch
        if torch.cuda.is_available():
            device = 0  # First CUDA GPU
        elif torch.backends.mps.is_available():
            device = "mps"  # Apple Silicon
        else:
            device = -1  # CPU

        transcriber = pipeline(
            "automatic-speech-recognition",
            model="imvladikon/wav2vec2-large-xlsr-53-hebrew",
            device=device
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
        result = model(audio_path, chunk_length_s=30, return_timestamps='word')

        # Convert to whisper-like format
        segments = []

        # Use the full text directly from the result
        full_text = clean_rtl_markers(result.get('text', ''))

        # Process chunks with actual timestamps
        for chunk in result.get("chunks", []):
            chunk_text = clean_rtl_markers(chunk['text'])
            timestamp = chunk.get('timestamp', (0, 0))

            # Handle timestamp tuple (start, end)
            if isinstance(timestamp, tuple) and len(timestamp) == 2:
                chunk_start_time, chunk_end_time = timestamp
            else:
                # Fallback if timestamp format is unexpected
                chunk_start_time = start_time
                chunk_end_time = start_time + 30

            segment_dict = {
                'start': start_time + chunk_start_time,
                'end': start_time + chunk_end_time,
                'text': chunk_text
            }
            segments.append(segment_dict)

        result = {
            'language': 'he',
            'duration': duration if duration else (segments[-1]['end'] - start_time if segments else 0),
            'text': full_text,
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
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    base_results_dir = str(PROJECT_ROOT / "results")

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

def clean_rtl_markers(text: str) -> str:
    """
    Remove RTL (Right-to-Left) control characters and padding tokens from Hebrew text.

    These invisible Unicode characters are added by Whisper:
    - U+202B: RIGHT-TO-LEFT EMBEDDING
    - U+202A: LEFT-TO-RIGHT EMBEDDING
    - U+202C: POP DIRECTIONAL FORMATTING
    - U+200F: RIGHT-TO-LEFT MARK
    - U+200E: LEFT-TO-RIGHT MARK

    Also removes [PAD] tokens from Hugging Face models.

    Args:
        text: Text containing RTL markers and/or padding tokens

    Returns:
        Text with RTL markers and padding tokens removed
    """
    # Remove RTL markers
    text = RTL_PATTERN.sub('', text)

    # Remove [PAD] tokens from Hugging Face models
    text = text.replace('[PAD]', '')

    return text

def format_timestamp(seconds):
    """
    Convert seconds to MM:SS format
    """
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:05.2f}"

def write_chunk_output(output_dir, chunk_num, chunk_result, chunk_time, chunk_start, chunk_duration, cumulative_text, num_chunks=None):
    """
    Write chunk output to files in real-time (after each 2-minute chunk)

    Also performs real-time AI analysis if Ollama is available

    Args:
        output_dir: Directory to write outputs to
        chunk_num: Chunk number (1-indexed)
        chunk_result: Result dictionary from process_chunk
        chunk_time: Processing time for this chunk
        chunk_start: Start time in seconds
        chunk_duration: Duration of chunk in seconds
        cumulative_text: All text transcribed so far (including this chunk)
        num_chunks: Total number of chunks (optional, for AI analysis progress)
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

    # ═══════════════════════════════════════════════════════════
    # REAL-TIME AI ANALYSIS (Optional, per chunk)
    # ═══════════════════════════════════════════════════════════
    if num_chunks:
        try:
            analyzer = OllamaAnalyzer(model="aya-expanse:8b")

            if analyzer.is_available():
                print(f"\n🤖 Running real-time AI analysis...")

                analysis_start = time.time()
                analysis = analyze_chunk_realtime(
                    chunk_text=chunk_result['text'],
                    cumulative_text=cumulative_text,
                    chunk_num=chunk_num,
                    total_chunks=num_chunks,
                    chunk_start=chunk_start
                )
                analysis_time = time.time() - analysis_start

                if analysis:
                    # Save progressive summary
                    update_progressive_summary(
                        output_dir=output_dir,
                        analysis=analysis,
                        chunk_num=chunk_num,
                        chunk_start=chunk_start,
                        chunk_duration=chunk_duration
                    )

                    # Display results
                    print(f"✅ AI Analysis Complete! ({analysis_time:.1f} seconds)")

                    # Show chunk analysis
                    if analysis.get('chunk_analysis') and analysis['chunk_analysis'].get('summary'):
                        print(f"\n📝 CHUNK {chunk_num} SUMMARY:")
                        summary_preview = analysis['chunk_analysis']['summary'][:150]
                        if len(analysis['chunk_analysis']['summary']) > 150:
                            summary_preview += "..."
                        print(f"   {summary_preview}")

                    # Show cumulative summary
                    if analysis.get('cumulative_analysis') and analysis['cumulative_analysis'].get('summary'):
                        print(f"\n📚 CUMULATIVE SUMMARY (Chunks 1-{chunk_num}):")
                        cum_summary = analysis['cumulative_analysis']['summary'][:150]
                        if len(analysis['cumulative_analysis']['summary']) > 150:
                            cum_summary += "..."
                        print(f"   {cum_summary}")

                    # Show reel suggestion if any
                    if (analysis.get('chunk_analysis') and
                        analysis['chunk_analysis'].get('reel_segments') and
                        len(analysis['chunk_analysis']['reel_segments']) > 0):
                        seg = analysis['chunk_analysis']['reel_segments'][0]
                        print(f"\n🎬 NEW REEL SUGGESTION:")
                        print(f"   {seg['time_range']} - {seg['reason'][:60]}...")

                    # Show progress
                    print(f"\n📊 PROGRESS: {chunk_num}/{num_chunks} chunks analyzed")
                    print(f"📄 Updated: {output_dir}/ai_summary.txt")
                else:
                    print(f"⚠️  AI analysis returned no results (may be too short)")

        except Exception:
            # Silent failure - don't disrupt transcription
            pass

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
        video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
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
                    cumulative_text=total_text,
                    num_chunks=num_chunks  # Pass total chunks for AI progress tracking
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
        video_name = os.path.splitext(os.path.basename(video_path))[0]
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
        
        # Basic speaker information (Whisper doesn't provide advanced speaker diarization)
        print("\nSPEAKER DETECTION NOTES:")
        print("- Whisper provides basic transcription but not advanced speaker diarization")
        print("- For advanced speaker detection, additional libraries like pyannote.audio are needed")
        print("- The segments above are based on natural speech pauses, not speaker changes")

        # NOTE: AI analysis now happens per-chunk in write_chunk_output()
        # No need for end-of-video analysis anymore

        return result
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)

# ============================================================================
# OLLAMA AI ANALYSIS (OPTIONAL)
# ============================================================================

class OllamaAnalyzer:
    """
    Optional AI-powered content analysis using local Ollama LLM

    Features:
    - Auto-generate video summaries
    - Extract main topics and keywords
    - Suggest optimal reel segments
    - Generate hashtags for social media

    Requires:
    - Ollama installed (https://ollama.ai)
    - Model downloaded: ollama pull aya-expanse:8b
    """

    def __init__(self, model="aya-expanse:8b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def is_available(self):
        """
        Check if Ollama is running and the model is available

        Returns:
            bool: True if Ollama is ready, False otherwise
        """
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)

            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                # Check if our model (or any variant) is available
                return any(self.model.split(':')[0] in m for m in model_names)
        except:
            pass

        return False

    def analyze(self, transcript: str, segments: List[Dict], timeout: int = 90):
        """
        Generate comprehensive AI analysis of the transcript

        Args:
            transcript: Full video transcript text
            segments: List of timestamped segments
            timeout: Maximum time to wait for response (seconds)

        Returns:
            dict: Analysis results with summary, topics, hashtags, reel suggestions
            None: If analysis fails or Ollama unavailable
        """
        try:
            import requests

            # Build the analysis prompt
            prompt = self._build_analysis_prompt(transcript, segments)

            # Call Ollama API
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower = more deterministic
                        "num_predict": 800    # Limit response length for speed
                    }
                },
                timeout=timeout
            )

            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', '')

                # Parse the LLM response into structured data
                return self._parse_response(llm_response, segments)

        except Exception as e:
            # Silent failure - don't disrupt transcription workflow
            pass

        return None

    def _build_analysis_prompt(self, transcript: str, segments: List[Dict]) -> str:
        """
        Build the prompt for Ollama to analyze the transcript

        Uses English instructions for better model performance, but handles
        Hebrew/English bilingual content appropriately.
        """
        # Limit transcript length for faster processing
        # For a 5-minute video, this is typically ~2000-3000 chars
        limited_transcript = transcript[:3000]
        if len(transcript) > 3000:
            limited_transcript += "\n\n... (content continues) ..."

        # Build prompt with clear structure
        prompt = f"""Analyze this video transcript and provide:

1. SUMMARY: A concise 3-5 sentence summary of the video content
2. TOPICS: List 3-5 main topics covered (as bullet points starting with -)
3. HASHTAGS: Suggest 5-7 relevant hashtags for social media (mix of broad and specific)
4. REEL SEGMENTS: Identify 2-3 engaging moments suitable for short-form content (15-60 seconds)

For REEL SEGMENTS, identify timestamps that would make good standalone clips for Instagram Reels/TikTok/YouTube Shorts. Look for:
- Engaging hooks or interesting statements
- Self-contained explanations
- Entertaining or educational moments
- Content that makes sense without full context

Transcript:
{limited_transcript}

Format your response EXACTLY like this:

SUMMARY:
[Your 3-5 sentence summary here]

TOPICS:
- Topic 1
- Topic 2
- Topic 3

HASHTAGS:
#hashtag1 #hashtag2 #hashtag3 #hashtag4 #hashtag5

REEL SEGMENTS:
[MM:SS - MM:SS] Brief reason why this is engaging
[MM:SS - MM:SS] Brief reason why this is engaging
"""

        return prompt

    def _parse_response(self, llm_response: str, segments: List[Dict]) -> Dict:
        """
        Parse the LLM's text response into structured data

        Args:
            llm_response: Raw text from Ollama
            segments: Original segments for timestamp validation

        Returns:
            dict: Structured analysis results
        """
        result = {
            'summary': '',
            'topics': [],
            'hashtags': [],
            'reel_segments': []
        }

        lines = llm_response.split('\n')
        current_section = None

        for line in lines:
            line_stripped = line.strip()

            # Detect section headers
            if 'SUMMARY:' in line_stripped.upper():
                current_section = 'summary'
                continue
            elif 'TOPICS:' in line_stripped.upper():
                current_section = 'topics'
                continue
            elif 'HASHTAGS:' in line_stripped.upper():
                current_section = 'hashtags'
                continue
            elif 'REEL' in line_stripped.upper() and 'SEGMENT' in line_stripped.upper():
                current_section = 'reels'
                continue

            # Parse content based on current section
            if not line_stripped or not current_section:
                continue

            if current_section == 'summary':
                result['summary'] += line_stripped + ' '

            elif current_section == 'topics':
                # Extract topics (lines starting with - or •)
                if line_stripped.startswith('-') or line_stripped.startswith('•'):
                    topic = line_stripped[1:].strip()
                    if topic:
                        result['topics'].append(topic)

            elif current_section == 'hashtags':
                # Extract hashtags
                hashtags = [word.strip() for word in line_stripped.split() if word.startswith('#')]
                result['hashtags'].extend(hashtags)

            elif current_section == 'reels':
                # Parse reel segments: [MM:SS - MM:SS] reason
                if '[' in line_stripped and ']' in line_stripped:
                    try:
                        # Extract timestamp range
                        timestamp_part = line_stripped[line_stripped.find('[')+1:line_stripped.find(']')]
                        reason_part = line_stripped[line_stripped.find(']')+1:].strip()

                        if '-' in timestamp_part:
                            start_str, end_str = timestamp_part.split('-')
                            result['reel_segments'].append({
                                'time_range': timestamp_part.strip(),
                                'reason': reason_part
                            })
                    except:
                        # Skip malformed entries
                        pass

        # Clean up summary
        result['summary'] = result['summary'].strip()

        return result


def save_ollama_analysis(output_dir: str, analysis: Dict, video_name: str):
    """
    Save AI analysis results to readable text files

    Args:
        output_dir: Directory to save files
        analysis: Analysis results from OllamaAnalyzer
        video_name: Base name of the video file

    Returns:
        tuple: Paths to (summary_file, reels_file)
    """
    # Save comprehensive summary file
    summary_file = os.path.join(output_dir, "ai_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("AI CONTENT ANALYSIS\n")
        f.write("Generated with: Ollama (aya-expanse:8b)\n")
        f.write("═" * 70 + "\n\n")

        f.write("SUMMARY\n")
        f.write("─" * 70 + "\n")
        f.write(analysis['summary'] + "\n\n")

        f.write("MAIN TOPICS\n")
        f.write("─" * 70 + "\n")
        for topic in analysis['topics']:
            f.write(f"• {topic}\n")
        f.write("\n")

        f.write("SUGGESTED HASHTAGS\n")
        f.write("─" * 70 + "\n")
        f.write(" ".join(analysis['hashtags']) + "\n\n")

        f.write("💡 TIP: Use these hashtags when posting your content to social media\n")

    # Save reel suggestions file with actionable commands
    reels_file = os.path.join(output_dir, "suggested_reels.txt")
    with open(reels_file, 'w', encoding='utf-8') as f:
        f.write("SUGGESTED REEL SEGMENTS\n")
        f.write("AI-identified engaging moments for short-form content\n")
        f.write("═" * 70 + "\n\n")

        if analysis['reel_segments']:
            for i, segment in enumerate(analysis['reel_segments'], 1):
                f.write(f"SEGMENT {i}\n")
                f.write(f"Time: {segment['time_range']}\n")
                f.write(f"Why: {segment['reason']}\n")
                f.write(f"\n")
                f.write(f"To extract this segment, run:\n")
                f.write(f"  python src/scripts/cut_video_segments.py \\\n")
                f.write(f"    --video [path/to/{video_name}] \\\n")
                f.write(f"    --ranges \"{segment['time_range'].replace(' - ', '-')}\"\n")
                f.write("\n" + "─" * 70 + "\n\n")
        else:
            f.write("No specific reel segments identified.\n")
            f.write("Try cutting interesting moments manually.\n")

    return summary_file, reels_file


def analyze_chunk_realtime(chunk_text: str, cumulative_text: str, chunk_num: int,
                           total_chunks: int, chunk_start: float) -> Dict:
    """
    Run real-time AI analysis on a single chunk and cumulative transcript

    Performs two analyses:
    1. Individual chunk analysis (quick, focused on this chunk only)
    2. Cumulative analysis (all chunks processed so far)

    Args:
        chunk_text: Transcript text for this chunk only
        cumulative_text: All transcript text accumulated so far
        chunk_num: Current chunk number (1-indexed)
        total_chunks: Total number of chunks expected
        chunk_start: Start time of this chunk in seconds

    Returns:
        dict with 'chunk_analysis' and 'cumulative_analysis', or None if Ollama unavailable
    """
    try:
        analyzer = OllamaAnalyzer(model="aya-expanse:8b")

        if not analyzer.is_available():
            return None

        import requests

        result = {
            'chunk_analysis': None,
            'cumulative_analysis': None,
            'chunk_num': chunk_num,
            'total_chunks': total_chunks
        }

        # ===================================================================
        # ANALYSIS 1: Individual Chunk (Fast, ~5-10 seconds)
        # ===================================================================
        chunk_prompt = f"""Analyze this 2-minute video segment and provide:

1. SUMMARY: A brief 2-3 sentence summary of THIS segment only
2. TOPICS: List 2-3 main topics in THIS segment (bullet points starting with -)
3. REEL: Suggest 1 engaging moment from THIS segment for short-form content

Segment transcript:
{chunk_text[:1500]}

Format EXACTLY like this:

SUMMARY:
[Your 2-3 sentence summary]

TOPICS:
- Topic 1
- Topic 2

REEL:
[timestamp] Brief reason why engaging
"""

        try:
            response = requests.post(
                f"{analyzer.base_url}/api/generate",
                json={
                    "model": analyzer.model,
                    "prompt": chunk_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 400  # Short response for speed
                    }
                },
                timeout=120  # Increased timeout - chunk analysis can take 60s+
            )

            if response.status_code == 200:
                llm_response = response.json().get('response', '')
                result['chunk_analysis'] = analyzer._parse_response(llm_response, [])

        except Exception as e:
            # Silent failure on chunk analysis (continues without AI summary)
            pass

        # ===================================================================
        # ANALYSIS 2: Cumulative (Slower, ~10-20 seconds)
        # ===================================================================
        cumulative_prompt = f"""Analyze this video transcript (chunks 1-{chunk_num} of {total_chunks}) and provide:

1. SUMMARY: A comprehensive 3-5 sentence summary of EVERYTHING so far
2. TOPICS: List all main topics covered so far (bullet points starting with -)
3. HASHTAGS: Suggest 5-7 relevant hashtags for social media
4. TOP REELS: Identify the 2-3 BEST moments for short-form content from ALL chunks

Cumulative transcript (all chunks so far):
{cumulative_text[:3000]}
{f'... (content continues, {len(cumulative_text)} total chars)' if len(cumulative_text) > 3000 else ''}

Format EXACTLY like this:

SUMMARY:
[Your 3-5 sentence summary]

TOPICS:
- Topic 1
- Topic 2
- Topic 3

HASHTAGS:
#hashtag1 #hashtag2 #hashtag3

TOP REELS:
[timestamp] Reason
[timestamp] Reason
"""

        try:
            response = requests.post(
                f"{analyzer.base_url}/api/generate",
                json={
                    "model": analyzer.model,
                    "prompt": cumulative_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 800  # Longer response for cumulative
                    }
                },
                timeout=120  # Increased timeout - cumulative can be slow
            )

            if response.status_code == 200:
                llm_response = response.json().get('response', '')
                result['cumulative_analysis'] = analyzer._parse_response(llm_response, [])

        except Exception as e:
            # Silent failure on cumulative analysis (continues without AI summary)
            pass

        return result

    except Exception:
        return None  # Complete silent failure if anything goes wrong


def update_progressive_summary(output_dir: str, analysis: Dict, chunk_num: int,
                               chunk_start: float, chunk_duration: float):
    """
    Update the progressive ai_summary.txt file with new chunk analysis

    File structure:
    - Latest chunk analysis at top
    - Cumulative analysis in middle
    - Previous chunks archived at bottom

    Args:
        output_dir: Directory to save the file
        analysis: Analysis results from analyze_chunk_realtime()
        chunk_num: Current chunk number
        chunk_start: Start time of chunk in seconds
        chunk_duration: Duration of chunk in seconds
    """
    if not analysis:
        return

    summary_file = os.path.join(output_dir, "ai_summary.txt")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Read existing content to archive previous chunks
    archived_chunks = ""
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Extract previous chunk analyses (everything after "PREVIOUS CHUNKS")
                if "PREVIOUS CHUNKS" in content:
                    parts = content.split("PREVIOUS CHUNKS")
                    if len(parts) > 1:
                        archived_chunks = parts[1].strip()
                # Also archive the current chunk analysis before overwriting
                if f"CHUNK {chunk_num - 1} ANALYSIS" in content:
                    # Find the previous chunk section
                    lines = content.split('\n')
                    chunk_section = []
                    in_chunk_section = False
                    for line in lines:
                        if f"CHUNK {chunk_num - 1} ANALYSIS" in line:
                            in_chunk_section = True
                        elif "CUMULATIVE ANALYSIS" in line or "═══════" in line:
                            if in_chunk_section:
                                break
                        if in_chunk_section:
                            chunk_section.append(line)
                    if chunk_section:
                        archived_chunks = '\n'.join(chunk_section) + "\n\n" + archived_chunks
        except:
            pass  # If reading fails, just start fresh

    # Write updated summary
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("AI CONTENT ANALYSIS (Real-Time)\n")
        f.write(f"Updated after: Chunk {chunk_num} of {analysis['total_chunks']}\n")
        f.write(f"Last update: {timestamp}\n")
        f.write(f"Generated with: Ollama (aya-expanse:8b)\n")
        f.write("═" * 70 + "\n\n")

        # ============================================================
        # CURRENT CHUNK ANALYSIS (Individual)
        # ============================================================
        if analysis.get('chunk_analysis'):
            chunk_anal = analysis['chunk_analysis']
            f.write(f"CHUNK {chunk_num} ANALYSIS (Individual)\n")
            f.write(f"Time: {format_timestamp(chunk_start)} - {format_timestamp(chunk_start + chunk_duration)}\n")
            f.write("─" * 70 + "\n")

            if chunk_anal.get('summary'):
                f.write(f"Summary: {chunk_anal['summary']}\n\n")

            if chunk_anal.get('topics'):
                f.write("Topics:\n")
                for topic in chunk_anal['topics']:
                    f.write(f"  • {topic}\n")
                f.write("\n")

            if chunk_anal.get('reel_segments'):
                f.write("Reel Suggestion:\n")
                for seg in chunk_anal['reel_segments'][:1]:  # Just first one
                    f.write(f"  {seg['time_range']} - {seg['reason']}\n")
                f.write("\n")

        f.write("═" * 70 + "\n\n")

        # ============================================================
        # CUMULATIVE ANALYSIS (All chunks so far)
        # ============================================================
        if analysis.get('cumulative_analysis'):
            cum_anal = analysis['cumulative_analysis']
            f.write(f"CUMULATIVE ANALYSIS (Chunks 1-{chunk_num})\n")
            f.write("─" * 70 + "\n")

            if cum_anal.get('summary'):
                f.write(f"Overall Summary:\n{cum_anal['summary']}\n\n")

            if cum_anal.get('topics'):
                f.write("All Topics:\n")
                for topic in cum_anal['topics']:
                    f.write(f"  • {topic}\n")
                f.write("\n")

            if cum_anal.get('reel_segments'):
                f.write(f"Best Reel Suggestions (Top {len(cum_anal['reel_segments'])}):\n")
                for i, seg in enumerate(cum_anal['reel_segments'], 1):
                    f.write(f"  {i}. {seg['time_range']} - {seg['reason']}\n")
                f.write("\n")

            if cum_anal.get('hashtags'):
                f.write("Suggested Hashtags:\n")
                f.write("  " + " ".join(cum_anal['hashtags']) + "\n\n")

        f.write("═" * 70 + "\n\n")

        # ============================================================
        # PREVIOUS CHUNKS (Archived)
        # ============================================================
        if archived_chunks:
            f.write("PREVIOUS CHUNKS\n")
            f.write("─" * 70 + "\n")
            f.write(archived_chunks)


if __name__ == "__main__":
    # Run in interactive mode
    video_path = interactive_mode()

    if not video_path:
        print("\n❌ No video selected")
        exit(1)

    # Process the selected video
    try:
        print("\n" + "=" * 80)
        print("Starting transcription...")
        print("=" * 80 + "\n")

        result = transcribe_video(video_path)

        print("\n🎉 Hebrew-optimized transcription completed successfully!")
        print("💡 This script uses the best Hebrew models:")
        print("   - wav2vec2-large-xlsr-53-hebrew (Hugging Face)")
        print("   - large-v3-turbo (Whisper fallback)")
        print("   - Automatic model fallback with proper error handling")

    except Exception as e:
        print(f"\n❌ Error during transcription: {str(e)}")
        exit(1)