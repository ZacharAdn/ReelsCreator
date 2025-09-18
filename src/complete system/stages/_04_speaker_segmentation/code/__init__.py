"""
Stage 1: Speaker Segmentation Module

High-precision speaker diarization for educational content.
Identifies and segments teacher vs student speech for optimal content extraction.
"""

from .speaker_analyzer import (
    SpeakerSegmentationPipeline,
    SpeakerSegmentationConfig,
    SpeakerSegment,
    SpeakerAnalysisResult
)

__version__ = "1.0.0"
__author__ = "Reels Extractor Team"

__all__ = [
    "SpeakerSegmentationPipeline",
    "SpeakerSegmentationConfig", 
    "SpeakerSegment",
    "SpeakerAnalysisResult"
]
