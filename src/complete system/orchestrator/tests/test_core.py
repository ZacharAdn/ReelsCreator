"""
Unit tests for core Content Extractor functionality
"""

import pytest
import json
from dataclasses import asdict
from unittest.mock import Mock, patch

from src.models import Segment, ProcessingConfig, ProcessingResult
from src.segmentation import SegmentProcessor
from src.embeddings import EmbeddingGenerator
from src.evaluation import ContentEvaluator

# Import the classes we'll be testing (once implemented)
# from src.content_extractor import Segment, ContentExtractor

class TestSegment:
    """Test Segment dataclass functionality"""
    
    def test_segment_creation(self):
        """Test creating a segment with basic data"""
        seg = Segment(start_time=0.0, end_time=10.0, text="hello", confidence=0.9)
        assert seg.duration() == pytest.approx(10.0)
        assert seg.text == "hello"
    
    def test_segment_duration_calculation(self):
        """Test duration calculation"""
        seg = Segment(start_time=2.5, end_time=5.0, text="t", confidence=0.5)
        assert seg.duration() == pytest.approx(2.5)
    
    def test_segment_serialization(self):
        """Test JSON serialization/deserialization"""
        seg = Segment(start_time=1.0, end_time=2.0, text="x", confidence=0.8, value_score=0.7)
        s = seg.to_json()
        seg2 = Segment.from_json(s)
        assert seg2.start_time == 1.0
        assert seg2.end_time == 2.0
        assert seg2.text == "x"
        assert seg2.confidence == pytest.approx(0.8)

class TestContentExtractor:
    """Test ContentExtractor main functionality"""
    
    def test_initialization(self, test_config):
        """Test ContentExtractor initialization with config"""
        cfg = ProcessingConfig.from_dict(test_config)
        assert cfg.segment_duration == 45
        assert cfg.overlap_duration == 10
        assert cfg.min_score_threshold == pytest.approx(0.7)
    
    def test_video_processing_workflow(self, temp_video_file, mock_whisper_model):
        """Test video processing workflow"""
        # This will be implemented once we have the actual ContentExtractor class
        pass
    
    def test_segmentation_workflow(self, sample_segments):
        """Test segmentation workflow"""
        # Convert dicts to Segment
        segments = [Segment.from_dict(s) for s in sample_segments]
        sp = SegmentProcessor(segment_duration=30, overlap_duration=10)
        windows = sp.create_overlapping_segments(segments)
        assert len(windows) >= 1
        filtered = sp.filter_segments_by_duration(windows, min_duration=10.0)
        assert all(s.duration() >= 10.0 for s in filtered)
    
    def test_embedding_generation(self, sample_segments, mock_embedding_model):
        """Test embedding generation"""
        segments = [Segment.from_dict(s) for s in sample_segments]
        eg = EmbeddingGenerator()
        out = eg.add_embeddings_to_segments(segments, batch_size=2)
        assert all(seg.embedding is not None for seg in out)
    
    def test_content_evaluation(self, sample_segments, mock_llm_model):
        """Test content evaluation with open-source LLM"""
        segs = [Segment.from_dict(sample_segments[0])]
        ev = ContentEvaluator()
        out = ev.evaluate_segments(segs)
        assert out[0].value_score is not None
        assert out[0].reasoning is not None

class TestDataValidation:
    """Test data validation and error handling"""
    
    def test_invalid_audio_file(self):
        """Test handling of invalid audio files"""
        from src.transcription import WhisperTranscriber
        tr = WhisperTranscriber()
        with pytest.raises(FileNotFoundError):
            tr.process_audio_file("/non/existent/file.wav")
    
    def test_empty_transcription(self):
        """Test handling of empty transcription results"""
        from src.transcription import WhisperTranscriber
        tr = WhisperTranscriber()
        # Simulate empty segments
        segments = tr.extract_segments({"segments": []})
        assert segments == []
    
    def test_llm_model_errors(self, mock_llm_model):
        """Test handling of LLM model errors"""
        # This will be implemented once we have the actual ContentExtractor class
        pass

class TestPerformance:
    """Test performance characteristics"""
    
    def test_processing_speed(self, temp_video_file):
        """Test processing speed for different file sizes"""
        # This will be implemented once we have the actual ContentExtractor class
        pass
    
    def test_memory_usage(self, temp_video_file):
        """Test memory usage during processing"""
        # This will be implemented once we have the actual ContentExtractor class
        pass 