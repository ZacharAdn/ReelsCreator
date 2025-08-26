"""
Transcription module using Whisper with enhanced multilingual support
"""

import logging
import torch
from typing import List, Dict, Any
from pathlib import Path
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.models import Segment
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
    
    def __init__(self, model_name: str = "base", primary_language: str = "he", smart_model_selection: bool = True):
        """
        Initialize Whisper transcriber
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large, or "auto" for smart selection)
            primary_language: Primary language for transcription
            smart_model_selection: Enable automatic model selection based on audio length
        """
        self.model_name = model_name
        self.primary_language = primary_language
        self.smart_model_selection = smart_model_selection
        self.model = None
        self.actual_model_used = None  # Track which model was actually used
        self.device = self._setup_device()
        logger.info(f"Initializing Whisper transcriber: {model_name} for language: {primary_language}")
        if smart_model_selection and model_name == "auto":
            logger.info("Smart model selection enabled - will choose optimal model based on audio duration")
    
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
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            duration = len(y) / sr
            return duration
        except ImportError:
            # Fallback using moviepy if librosa not available
            try:
                from moviepy.editor import AudioFileClip
                with AudioFileClip(audio_path) as audio:
                    return audio.duration
            except ImportError:
                logger.warning("Cannot determine audio duration - librosa and moviepy not available")
                return 600.0  # Default to 10 minutes if we can't determine duration
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}")
            return 600.0  # Default fallback
    
    def select_optimal_model(self, audio_duration: float) -> str:
        """
        Select optimal Whisper model based on audio duration
        
        Args:
            audio_duration: Duration of audio in seconds
            
        Returns:
            Optimal model name
        """
        duration_minutes = audio_duration / 60.0
        
        if duration_minutes < 3:  # Very short videos
            model = "tiny"
            reason = f"short video ({duration_minutes:.1f}m) - prioritizing speed"
        elif duration_minutes < 10:  # Short videos
            model = "base"
            reason = f"medium video ({duration_minutes:.1f}m) - balanced speed/accuracy"
        elif duration_minutes < 30:  # Medium videos
            model = "small"
            reason = f"longer video ({duration_minutes:.1f}m) - prioritizing accuracy"
        else:  # Long videos
            model = "base"  # Use base for very long videos to avoid memory issues
            reason = f"very long video ({duration_minutes:.1f}m) - balanced for memory efficiency"
        
        logger.info(f"🎯 Selected Whisper model '{model}' for {reason}")
        return model
    
    def process_audio_file(self, audio_path: str) -> List[Segment]:
        """
        Complete audio processing pipeline with smart model selection
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of Segment objects
        """
        # Validate file exists
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Smart model selection if enabled
        if self.smart_model_selection and (self.model_name == "auto" or self.model is None):
            audio_duration = self.get_audio_duration(audio_path)
            optimal_model = self.select_optimal_model(audio_duration)
            
            # Update model if different from current
            if self.actual_model_used != optimal_model:
                logger.info(f"Switching from {self.actual_model_used or 'None'} to {optimal_model} model")
                self.actual_model_used = optimal_model
                self.model = None  # Force reload with new model
                # Temporarily update model_name for loading
                original_model_name = self.model_name
                self.model_name = optimal_model
                self.load_model()
                self.model_name = original_model_name  # Restore original
        
        # Transcribe
        result = self.transcribe(audio_path)
        
        # Extract segments
        segments = self.extract_segments(result)
        
        return segments 