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
    segment_duration: int = 45
    overlap_duration: int = 10
    min_score_threshold: float = 0.7
    whisper_model: str = "base"
    embedding_model: str = "all-MiniLM-L6-v2"
    include_embeddings_in_json: bool = False
    keep_audio: bool = False
    embedding_batch_size: int = 32
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingConfig':
        """Create config from dictionary"""
        return cls(**data)


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