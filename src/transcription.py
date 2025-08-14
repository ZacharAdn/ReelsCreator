"""
Transcription module using Whisper with enhanced multilingual support
"""

import logging
import torch
from typing import List, Dict, Any
from pathlib import Path
import time

from .models import Segment
import math

# Try whisper-timestamped first, fallback to openai-whisper
try:
    import whisper_timestamped as whisper
    WHISPER_TIMESTAMPED = True
    logger = logging.getLogger(__name__)
    logger.info("Using whisper-timestamped for enhanced multilingual support")
except ImportError:
    import whisper
    WHISPER_TIMESTAMPED = False
    logger = logging.getLogger(__name__)
    logger.warning("Using standard openai-whisper (consider installing whisper-timestamped for better multilingual support)")

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Handles audio transcription using Whisper with multilingual support"""
    
    def __init__(self, model_name: str = "base", primary_language: str = "he"):
        """
        Initialize Whisper transcriber
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            primary_language: Primary language for transcription
        """
        self.model_name = model_name
        self.primary_language = primary_language
        self.model = None
        self.device = self._setup_device()
        logger.info(f"Initializing Whisper model: {model_name} for language: {primary_language}")
    
    def _setup_device(self) -> torch.device:
        """Setup M1 Mac optimized device"""
        if torch.backends.mps.is_available():
            logger.info("Using M1 GPU (MPS) for transcription")
            return torch.device("mps")
        else:
            logger.info("Using CPU for transcription")
            return torch.device("cpu")
    
    def load_model(self):
        """Load Whisper model"""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file with multilingual support
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        self.load_model()
        
        logger.info(f"Transcribing: {audio_path}")
        start_time = time.time()
        
        try:
            if WHISPER_TIMESTAMPED:
                # Enhanced multilingual transcription with whisper-timestamped
                result = whisper.transcribe(
                    self.model,
                    audio_path,
                    language=self.primary_language,
                    verbose=True,
                    detect_disfluencies=True  # Better for mixed languages
                )
            else:
                # Standard transcription with openai-whisper
                result = self.model.transcribe(
                    audio_path,
                    language=self.primary_language,
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