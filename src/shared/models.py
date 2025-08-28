"""
Data models for Content Extractor
"""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import timedelta
import json


@dataclass
class Segment:
    """Represents a transcribed segment with metadata"""
    start_time: float
    end_time: float
    text: str
    confidence: float
    embedding: Optional[List[float]] = None
    value_score: Optional[float] = None
    reasoning: Optional[str] = None
    # Speaker and language support
    speaker_id: Optional[str] = None
    speaker_confidence: Optional[float] = None
    primary_language: Optional[str] = None
    technical_terms: Optional[List[str]] = None
    
    def duration(self) -> float:
        """Calculate segment duration in seconds"""
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert segment to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Segment':
        """Create segment from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Segment':
        """Create segment from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class ProcessingConfig:
    """Configuration for content processing"""
    segment_duration: int = 90  # Updated to 90 seconds for longer segments
    overlap_duration: int = 20  # Updated to 20 seconds overlap
    min_score_threshold: float = 0.7
    # Transcription model settings
    whisper_model: str = "base"  # Backward compatibility
    transcription_model: str = "auto"  # New unified model field
    force_transcription_model: bool = False  # Disable smart selection
    embedding_model: str = "all-MiniLM-L6-v2"
    include_embeddings_in_json: bool = False
    keep_audio: bool = False
    embedding_batch_size: int = 32
    # Speaker and language processing
    enable_speaker_detection: bool = False
    primary_speaker_only: bool = False
    speaker_batch_size: int = 8
    preserve_technical_terms: bool = True
    primary_language: str = "he"  # Hebrew
    technical_language: str = "en"  # English for technical terms
    # Performance optimization options
    enable_similarity_analysis: bool = False  # Skip if not needed for speed
    enable_technical_terms: bool = True       # Language specific processing
    minimal_mode: bool = False               # Skip non-essential processing
    evaluation_batch_size: int = 5           # LLM batch processing size
    evaluation_model: str = "microsoft/Phi-3-mini-4k-instruct"  # LLM model for evaluation
    processing_profile: str = "balanced"     # draft/fast/balanced/quality
    # Evaluation control
    enable_content_evaluation: bool = True   # Disable for maximum speed
    use_rule_based_scoring: bool = False     # Fast rule-based scoring instead of LLM
    confidence_based_evaluation: bool = False # Only evaluate uncertain segments
    # Device control
    force_cpu: bool = False                  # Force CPU processing, disable MPS/CUDA
    # Stage output control
    save_stage_outputs: bool = False         # Save intermediate outputs from each stage
    stage_output_dir: str = "stage_outputs"  # Directory for stage outputs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfig':
        """Create config from dictionary"""
        return cls(**data)
    
    @classmethod
    def create_profile(cls, profile: str = "balanced") -> 'ProcessingConfig':
        """
        Create optimized processing profiles
        
        Args:
            profile: Processing profile type
                - "draft": Maximum speed processing (80% faster, no LLM evaluation)
                - "fast": Fast processing with rule-based scoring (60% faster)
                - "balanced": Default balanced processing
                - "quality": High-quality processing (20% slower)
        
        Returns:
            ProcessingConfig optimized for the profile
        """
        if profile == "draft":
            return cls(
                whisper_model="tiny",
                transcription_model="tiny",
                min_score_threshold=0.5,  # Lower threshold since no LLM evaluation
                embedding_batch_size=64,
                evaluation_batch_size=8,
                enable_similarity_analysis=False,
                enable_technical_terms=False,
                enable_speaker_detection=False,  # Disable speaker detection
                minimal_mode=True,
                enable_content_evaluation=False,  # Disable LLM evaluation
                use_rule_based_scoring=True,
                processing_profile="draft"
            )
        elif profile == "fast":
            return cls(
                whisper_model="tiny",
                transcription_model="tiny",
                min_score_threshold=0.6,
                embedding_batch_size=64,
                evaluation_batch_size=8,
                enable_similarity_analysis=False,
                enable_technical_terms=False,
                enable_speaker_detection=False,
                minimal_mode=True,
                enable_content_evaluation=True,
                use_rule_based_scoring=True,  # Use rule-based instead of LLM
                confidence_based_evaluation=True,
                processing_profile="fast"
            )
        elif profile == "quality":
            return cls(
                whisper_model="auto",  # Use smart model selection
                transcription_model="auto",  # Use smart model selection
                min_score_threshold=0.8,
                embedding_batch_size=16,
                evaluation_batch_size=3,
                enable_similarity_analysis=True,
                enable_technical_terms=True,
                minimal_mode=False,
                enable_content_evaluation=True,
                use_rule_based_scoring=False,
                processing_profile="quality"
            )
        else:  # balanced
            return cls(
                whisper_model="auto",  # Use smart model selection
                transcription_model="auto",  # Use smart model selection
                min_score_threshold=0.7,
                embedding_batch_size=32,
                evaluation_batch_size=5,
                enable_similarity_analysis=False,
                enable_technical_terms=True,
                minimal_mode=False,
                enable_content_evaluation=True,
                use_rule_based_scoring=False,
                processing_profile="balanced"
            )


@dataclass
class ProcessingResult:
    """Result of content processing"""
    segments: List[Segment]
    config: ProcessingConfig
    processing_time: float
    total_duration: float
    high_value_segments: List[Segment]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        def _segment_public_dict(segment: Segment) -> Dict[str, Any]:
            segment_dict = segment.to_dict()
            if not self.config.include_embeddings_in_json:
                segment_dict.pop("embedding", None)
            return segment_dict

        return {
            "segments": [_segment_public_dict(seg) for seg in self.segments],
            "config": self.config.to_dict(),
            "processing_time": self.processing_time,
            "total_duration": self.total_duration,
            "high_value_segments": [_segment_public_dict(seg) for seg in self.high_value_segments],
            "summary": {
                "total_segments": len(self.segments),
                "high_value_count": len(self.high_value_segments),
                "average_score": sum(seg.value_score or 0 for seg in self.segments) / len(self.segments) if self.segments else 0
            }
        }
    
    def to_json(self) -> str:
        """Convert result to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def save_to_file(self, filepath: str) -> None:
        """Save result to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json()) 