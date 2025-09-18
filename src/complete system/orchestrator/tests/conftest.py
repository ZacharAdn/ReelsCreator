"""
Pytest configuration and common fixtures for Content Extractor tests
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch
import torch

@pytest.fixture
def temp_video_file():
    """Create a temporary video file for testing"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
        # Create a minimal MP4 file for testing
        f.write(b'fake_video_data')
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

@pytest.fixture
def sample_segments():
    """Sample segments for testing"""
    return [
        {
            "start_time": 0.0,
            "end_time": 25.0,
            "text": "Hello, today we'll learn about data science.",
            "confidence": 0.95
        },
        {
            "start_time": 20.0,
            "end_time": 45.0,
            "text": "Let's start with pandas and numpy basics.",
            "confidence": 0.92
        }
    ]

@pytest.fixture
def mock_whisper_model():
    """Mock Whisper model for testing"""
    with patch('whisper.load_model') as mock:
        model = Mock()
        model.transcribe.return_value = {
            "text": "Sample transcription text",
            "segments": [
                {
                    "start": 0.0,
                    "end": 25.0,
                    "text": "Sample transcription text"
                }
            ]
        }
        mock.return_value = model
        yield mock

@pytest.fixture
def mock_embedding_model():
    """Mock sentence transformer model for testing"""
    with patch('sentence_transformers.SentenceTransformer') as mock:
        model = Mock()
        model.encode.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]  # Mock embedding
        mock.return_value = model
        yield mock

@pytest.fixture
def mock_llm_model():
    """Mock open-source LLM model for testing"""
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model, \
         patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
        
        model = Mock()
        tokenizer = Mock()
        # Return a tensor to mimic return_tensors='pt'
        tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
        # Decode returns a valid JSON string
        tokenizer.decode.return_value = '{"score": 0.8, "reasoning": "Good content"}'
        tokenizer.pad_token = None
        tokenizer.eos_token_id = 1
        tokenizer.eos_token = '</s>'
        # Model.generate returns a tensor shaped like token ids
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        
        mock_model.return_value = model
        mock_tokenizer.return_value = tokenizer
        yield mock_model, mock_tokenizer

@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "segment_duration": 45,
        "overlap_duration": 10,
        "min_score_threshold": 0.7,
        "whisper_model": "base"
    } 