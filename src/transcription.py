"""
Transcription module using Whisper
"""

import logging
import whisper
from typing import List, Dict, Any
from pathlib import Path
import time
import os
import wave
import numpy as np

from .models import Segment

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
    
    def _ensure_ffmpeg_available(self) -> None:
        """Ensure ffmpeg binary is available on PATH using imageio-ffmpeg if needed"""
        try:
            import shutil
            if shutil.which("ffmpeg"):
                return
            import imageio_ffmpeg as iio
            ffmpeg_exe = iio.get_ffmpeg_exe()
            ffmpeg_dir = os.path.dirname(ffmpeg_exe)
            os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_exe
            os.environ["FFMPEG_BINARY"] = ffmpeg_exe
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            logger.info(f"Configured ffmpeg from imageio-ffmpeg: {ffmpeg_exe}")
        except Exception as e:
            logger.warning(f"Could not auto-configure ffmpeg: {e}")
    
    def load_model(self):
        """Load Whisper model"""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
    
    def _read_wav_as_float32(self, wav_path: str) -> np.ndarray:
        """Read a PCM WAV file and return float32 mono waveform in range [-1, 1]"""
        with wave.open(wav_path, 'rb') as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            frames = wf.readframes(num_frames)
        
        if sample_width != 2:
            raise ValueError(f"Unsupported WAV sample width: {sample_width*8} bits. Expected 16-bit PCM.")
        
        audio = np.frombuffer(frames, dtype=np.int16)
        if num_channels > 1:
            audio = audio.reshape(-1, num_channels).mean(axis=1).astype(np.int16)
        
        # Normalize to float32 -1..1
        audio_float = (audio.astype(np.float32) / 32768.0).clip(-1.0, 1.0)
        
        # Whisper expects 16kHz; our extraction enforces 16kHz
        if sample_rate != 16000:
            logger.warning(f"WAV sample rate is {sample_rate}, expected 16000. Results may be degraded.")
        
        return audio_float
    
    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        self._ensure_ffmpeg_available()
        self.load_model()
        
        logger.info(f"Transcribing: {audio_path}")
        start_time = time.time()
        
        try:
            # Prefer direct WAV read to avoid external ffmpeg calls
            if str(audio_path).lower().endswith('.wav') and Path(audio_path).exists():
                audio_array = self._read_wav_as_float32(audio_path)
                result = self.model.transcribe(
                    audio_array,
                    word_timestamps=True,
                    verbose=False,
                    fp16=False,
                )
            else:
                result = self.model.transcribe(
                    audio_path,
                    word_timestamps=True,
                    verbose=False,
                    fp16=False,
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
            segment = Segment(
                start_time=segment_data["start"],
                end_time=segment_data["end"],
                text=segment_data["text"].strip(),
                confidence=segment_data.get("avg_logprob", 0.0)
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