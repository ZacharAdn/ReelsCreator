#!/usr/bin/env python3
"""
Simple transcription script to extract transcription and speaker detection from IMG_4222.MP4
"""

import whisper
import os

def transcribe_video(video_path):
    """
    Extract transcription with speaker detection from video
    """
    print(f"Processing video: {video_path}")
    
    # Load Whisper model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    # Transcribe with word-level timestamps
    print("Transcribing audio...")
    result = model.transcribe(
        video_path,
        word_timestamps=True,
        language="auto"
    )
    
    # Print results
    print("\n" + "="*60)
    print("TRANSCRIPTION RESULTS")
    print("="*60)
    
    print(f"Language detected: {result['language']}")
    print(f"Duration: {result.get('duration', 'N/A')} seconds")
    
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
    output_file = "transcription_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("TRANSCRIPTION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Language detected: {result['language']}\n")
        f.write(f"Duration: {result.get('duration', 'N/A')} seconds\n\n")
        
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
    return result

if __name__ == "__main__":
    video_file = "data/IMG_4222.MP4"
    
    if not os.path.exists(video_file):
        print(f"Error: Video file not found: {video_file}")
        exit(1)
    
    # Process the video
    try:
        result = transcribe_video(video_file)
        print("\n✅ Transcription completed successfully!")
        
        # Basic speaker detection info (Whisper doesn't do advanced speaker diarization)
        print("\n📝 Note: This uses basic Whisper transcription.")
        print("For advanced speaker detection, additional libraries like pyannote.audio would be needed.")
        
    except Exception as e:
        print(f"\n❌ Error during transcription: {e}")
        exit(1)