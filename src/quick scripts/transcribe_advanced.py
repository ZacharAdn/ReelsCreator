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
            segment_dict = {
                'start': current_time,
                'end': current_time + 30,  # Approximate, as HF doesn't provide exact timestamps
                'text': segment
            }
            segments.append(segment_dict)
            text += segment + " "
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

def ensure_results_dir():
    """
    Create results directory if it doesn't exist
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"📁 Created results directory: {results_dir}")
    return results_dir

def format_timestamp(seconds):
    """
    Convert seconds to MM:SS format
    """
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:05.2f}"

def transcribe_video(video_path):
    """
    Extract transcription from video using configurable chunk sizes
    """
    print(f"Processing video: {video_path}")
    start_time_total = time.time()
    
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
        
        # Save to file with model info in results directory
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        model_suffix = "hebrew" if model_type == "huggingface" else "whisper"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{video_name}_transcription_{model_suffix}_{timestamp}.txt"
        results_dir = ensure_results_dir()
        output_file = os.path.join(results_dir, output_filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("TRANSCRIPTION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Language detected: {result['language']}\n")
            f.write(f"Video Length: {format_timestamp(duration) if isinstance(duration, (int, float)) else duration}\n\n")
            
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
        print(f"📁 Results saved to: {output_file}")
        
        # Add processing time to the output file
        processing_minutes = total_time / 60
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write("\nPROCESSING INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total processing time: {total_time:.1f} seconds ({processing_minutes:.2f} minutes)\n")
            if video_duration > CHUNK_SIZE_SECONDS:  # If processed in chunks
                f.write(f"Processed in {math.ceil(video_duration/CHUNK_SIZE_SECONDS)} chunks of {CHUNK_SIZE_MINUTES} minutes each\n")
                for i, (_, chunk_time) in enumerate(all_results, 1):
                    f.write(f"Chunk {i} processing time: {chunk_time:.1f} seconds\n")
        
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
    # Try multiple video files available in data directory
    video_files = [
       "data/IMG_4225.MP4",  # Recommended for Hebrew testing
     #     "data/IMG_4262.MOV", 
        # "data/IMG_4216.MOV",
    #      "data/IMG_4222.MP4"   # Original target
    ]
    
    video_file = None
    for candidate in video_files:
        if os.path.exists(candidate):
            video_file = candidate
            print(f"📁 Found video file: {video_file}")
            break
    
    if not video_file:
        print(f"❌ Error: No video files found in data directory")
        print(f"Expected files: {', '.join(video_files)}")
        exit(1)
    
    # Process the video
    try:
        result = transcribe_video(video_file)
        print("\n🎉 Hebrew-optimized transcription completed successfully!")
        print("💡 This script now uses the best Hebrew models:")
        print("   - wav2vec2-large-xlsr-53-hebrew (Hugging Face)")
        print("   - large-v3-turbo (Whisper fallback)")
        print("   - Automatic model fallback with proper error handling")
        
    except Exception as e:
        print(f"\n❌ Error during transcription: {str(e)}")
        exit(1)