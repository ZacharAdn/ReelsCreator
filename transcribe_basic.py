#!/usr/bin/env python3
"""
Basic transcription script for IMG_4222.MP4 using moviepy for audio extraction
"""

import whisper
import os
from moviepy import VideoFileClip
import tempfile

def transcribe_video(video_path):
    """
    Extract transcription from video
    """
    print(f"Processing video: {video_path}")
    
    # Extract audio using moviepy
    print("Extracting audio from video...")
    video = VideoFileClip(video_path)
    
    # Create temporary wav file
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio_path = temp_audio.name
    temp_audio.close()
    
    try:
        # Extract audio to temporary file
        video.audio.write_audiofile(temp_audio_path, verbose=False, logger=None)
        video.close()
        
        # Load Whisper model
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        
        # Transcribe with word-level timestamps
        print("Transcribing audio...")
        result = model.transcribe(
            temp_audio_path,
            word_timestamps=True,
            language=None  # Auto-detect language
        )
        
        # Print results
        print("\n" + "="*60)
        print("TRANSCRIPTION RESULTS")
        print("="*60)
        
        print(f"Language detected: {result['language']}")
        duration = result.get('duration', 'N/A')
        print(f"Duration: {duration} seconds")
        
        print("\nFull transcript:")
        print("-" * 40)
        print(result['text'])
        
        print("\nSegments with timestamps:")
        print("-" * 40)
        for i, segment in enumerate(result['segments']):
            start_time = segment['start']
            end_time = segment['end']
            text = segment['text'].strip()
            print(f"[{start_time:7.2f}s - {end_time:7.2f}s] {text}")
        
        # Save to file
        output_file = "IMG_4222_transcription.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("TRANSCRIPTION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Video: {video_path}\n")
            f.write(f"Language detected: {result['language']}\n")
            f.write(f"Duration: {duration} seconds\n\n")
            
            f.write("Full transcript:\n")
            f.write("-" * 40 + "\n")
            f.write(result['text'] + "\n\n")
            
            f.write("Segments with timestamps:\n")
            f.write("-" * 40 + "\n")
            for segment in result['segments']:
                start_time = segment['start']
                end_time = segment['end']
                text = segment['text'].strip()
                f.write(f"[{start_time:7.2f}s - {end_time:7.2f}s] {text}\n")
        
        print(f"\nResults saved to: {output_file}")
        
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
    video_file = "data/IMG_4222.MP4"
    
    if not os.path.exists(video_file):
        print(f"Error: Video file not found: {video_file}")
        exit(1)
    
    # Process the video
    try:
        result = transcribe_video(video_file)
        print("\nTranscription completed successfully!")
        
    except Exception as e:
        print(f"\nError during transcription: {str(e)}")
        exit(1)