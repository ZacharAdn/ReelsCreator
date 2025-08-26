"""
Speaker diarization module for identifying and filtering speakers
"""

import logging
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """Handles speaker identification and diarization"""
    
    def __init__(self, batch_size: int = 8):
        """
        Initialize speaker diarizer
        
        Args:
            batch_size: Batch size for M1 GPU optimization
        """
        self.batch_size = batch_size
        self.pipeline = None
        self.device = self._setup_device()
        logger.info(f"Speaker diarizer initialized on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup M1 Mac optimized device"""
        if torch.backends.mps.is_available():
            logger.info("Using M1 GPU (MPS) for speaker diarization")
            return torch.device("mps")
        else:
            logger.info("Using CPU for speaker diarization")
            return torch.device("cpu")
    
    def load_model(self):
        """Load pyannote speaker diarization model"""
        if self.pipeline is None:
            try:
                from pyannote.audio import Pipeline
                
                logger.info("Loading speaker diarization pipeline...")
                # Use pyannote speaker diarization pipeline
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=None  # Public model
                )
                
                # M1 Mac optimization
                if self.device.type == "mps":
                    self.pipeline.to(self.device)
                    # Clear MPS cache
                    torch.mps.empty_cache()
                
                logger.info("Speaker diarization pipeline loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load speaker diarization model: {e}")
                logger.warning("Speaker diarization will be disabled")
                self.pipeline = None
    
    def analyze_speakers(self, audio_path: str) -> Dict[str, any]:
        """
        Analyze speakers in audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with speaker analysis results
        """
        if self.pipeline is None:
            self.load_model()
        
        if self.pipeline is None:
            logger.warning("Speaker diarization unavailable, returning empty results")
            return {"speakers": {}, "primary_speaker": None}
        
        try:
            logger.info(f"Analyzing speakers in: {audio_path}")
            
            # Run diarization
            diarization = self.pipeline(audio_path)
            
            # Process results
            speakers = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speakers:
                    speakers[speaker] = {
                        "total_time": 0.0,
                        "segments": []
                    }
                
                speakers[speaker]["total_time"] += turn.duration
                speakers[speaker]["segments"].append({
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.duration
                })
            
            # Find primary speaker (most speaking time)
            primary_speaker = None
            max_time = 0.0
            
            for speaker_id, data in speakers.items():
                if data["total_time"] > max_time:
                    max_time = data["total_time"]
                    primary_speaker = speaker_id
            
            logger.info(f"Found {len(speakers)} speakers, primary: {primary_speaker}")
            
            return {
                "speakers": speakers,
                "primary_speaker": primary_speaker,
                "total_speakers": len(speakers)
            }
            
        except Exception as e:
            logger.error(f"Speaker analysis failed: {e}")
            return {"speakers": {}, "primary_speaker": None}
    
    def filter_segments_by_speaker(self, segments: List, speaker_analysis: Dict, 
                                 primary_only: bool = True) -> List:
        """
        Filter segments by speaker identity
        
        Args:
            segments: List of segments to filter
            speaker_analysis: Results from analyze_speakers
            primary_only: If True, keep only primary speaker segments
            
        Returns:
            Filtered list of segments
        """
        if not speaker_analysis.get("speakers") or not primary_only:
            return segments
        
        primary_speaker = speaker_analysis.get("primary_speaker")
        if not primary_speaker:
            return segments
        
        primary_segments = speaker_analysis["speakers"][primary_speaker]["segments"]
        filtered_segments = []
        
        for segment in segments:
            # Check if segment overlaps with primary speaker
            segment_start = segment.start_time
            segment_end = segment.end_time
            
            for speaker_seg in primary_segments:
                # Calculate overlap
                overlap_start = max(segment_start, speaker_seg["start"])
                overlap_end = min(segment_end, speaker_seg["end"])
                
                if overlap_end > overlap_start:
                    # Significant overlap found
                    overlap_ratio = (overlap_end - overlap_start) / (segment_end - segment_start)
                    
                    if overlap_ratio > 0.5:  # At least 50% overlap
                        # Add speaker metadata
                        segment.speaker_id = primary_speaker
                        segment.speaker_confidence = overlap_ratio
                        filtered_segments.append(segment)
                        break
        
        logger.info(f"Filtered {len(segments)} segments to {len(filtered_segments)} primary speaker segments")
        return filtered_segments
    
    def get_speaker_info(self, segment_start: float, segment_end: float, 
                        speaker_analysis: Dict) -> Tuple[Optional[str], Optional[float]]:
        """
        Get speaker information for a specific time segment
        
        Args:
            segment_start: Start time of segment
            segment_end: End time of segment
            speaker_analysis: Results from analyze_speakers
            
        Returns:
            Tuple of (speaker_id, confidence)
        """
        if not speaker_analysis.get("speakers"):
            return None, None
        
        best_speaker = None
        best_overlap = 0.0
        
        for speaker_id, data in speaker_analysis["speakers"].items():
            for speaker_seg in data["segments"]:
                # Calculate overlap
                overlap_start = max(segment_start, speaker_seg["start"])
                overlap_end = min(segment_end, speaker_seg["end"])
                
                if overlap_end > overlap_start:
                    overlap_ratio = (overlap_end - overlap_start) / (segment_end - segment_start)
                    
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_speaker = speaker_id
        
        return best_speaker, best_overlap if best_overlap > 0.1 else None
