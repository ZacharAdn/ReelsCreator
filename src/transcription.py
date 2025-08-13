"""
Transcription module using Whisper
"""

import logging
import whisper
from typing import List, Dict, Any
from pathlib import Path
import time

from .models import Segment
import math

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Handles audio transcription using Whisper"""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize Whisper transcriber
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        logger.info(f"Initializing Whisper model: {model_name}")
    
    def load_model(self):
        """Load Whisper model"""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        self.load_model()
        
        logger.info(f"Transcribing: {audio_path}")
        start_time = time.time()
        
        try:
            result = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=True
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def extract_segments(self, transcription_result: Dict[str, Any]) -> List[Segment]:
        """
        Extract segments from transcription result
        
        Args:
            transcription_result: Result from Whisper transcription
            
        Returns:
            List of Segment objects
        """
        segments = []
        
        for segment_data in transcription_result.get("segments", []):
            # Convert Whisper avg_logprob (typically negative) to a pseudo-confidence in [0,1]
            raw_logprob = float(segment_data.get("avg_logprob", -2.0))
            # Clamp to a sensible range to avoid exp underflow, then exponentiate
            raw_logprob = max(-10.0, min(0.0, raw_logprob))
            confidence_prob = math.exp(raw_logprob)

            segment = Segment(
                start_time=segment_data["start"],
                end_time=segment_data["end"],
                text=segment_data["text"].strip(),
                confidence=confidence_prob
            )
            segments.append(segment)
        
        logger.info(f"Extracted {len(segments)} segments from transcription")
        return segments
    
    def process_audio_file(self, audio_path: str) -> List[Segment]:
        """
        Complete audio processing pipeline
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of Segment objects
        """
        # Validate file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Transcribe
        result = self.transcribe(audio_path)
        
        # Extract segments
        segments = self.extract_segments(result)
        
        return segments 