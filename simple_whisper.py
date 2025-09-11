#!/usr/bin/env python3
"""
Simple transcription using openai-whisper for IMG_4222.MP4
"""

import whisper
import os
import sys

def transcribe_video():
    video_file = "data/IMG_4222.MP4"
    
    if not os.path.exists(video_file):
        print(f"Error: Video file not found: {video_file}")
        return False
    
    print(f"Transcribing: {video_file}")
    print("Loading whisper model...")
    
    try:
        # Load the model
        model = whisper.load_model("base")
        
        print("Starting transcription (this may take a few minutes)...")
        
        # Transcribe the video
        result = model.transcribe(video_file)
        
        # Save to file
        output_file = "IMG_4222_transcript.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("TRANSCRIPTION RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Video: {video_file}\n")
            f.write(f"Language: {result.get('language', 'Unknown')}\n\n")
            f.write("Full Transcript:\n")
            f.write("-"*30 + "\n")
            f.write(result['text'] + "\n\n")
            
            if 'segments' in result:
                f.write("Segments with Timestamps:\n")
                f.write("-"*30 + "\n")
                for segment in result['segments']:
                    start = segment['start']
                    end = segment['end']
                    text = segment['text'].strip()
                    f.write(f"[{start:6.1f}s - {end:6.1f}s] {text}\n")
        
        # Display results
        print("\n" + "="*50)
        print("TRANSCRIPTION COMPLETED")
        print("="*50)
        print(f"Language detected: {result.get('language', 'Unknown')}")
        print(f"Full transcript saved to: {output_file}")
        
        print("\nFull transcript:")
        print("-"*30)
        print(result['text'])
        
        return True
        
    except Exception as e:
        print(f"Error during transcription: {e}")
        return False

if __name__ == "__main__":
    success = transcribe_video()
    if not success:
        sys.exit(1)