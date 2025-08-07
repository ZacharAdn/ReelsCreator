# Testing Strategy - Content Extractor

## Overview
This document outlines the comprehensive testing approach for the Content Extractor system, covering both system-level testing and data-driven testing methodologies.

## Testing Pyramid

### 1. Unit Tests (Base)
- **Coverage**: Individual functions and classes
- **Tools**: pytest, unittest
- **Target**: 90%+ code coverage

#### Core Components to Test:
- `Segment` dataclass operations
- Video processing and audio extraction
- Whisper transcription wrapper
- Embedding generation functions
- Open-source LLM evaluation logic
- JSON export/import functionality

### 2. Integration Tests (Middle)
- **Coverage**: Component interactions
- **Tools**: pytest with fixtures
- **Focus**: End-to-end workflows

#### Integration Test Scenarios:
- Video processing → Audio extraction → Transcription → Segmentation → Embedding pipeline
- Content evaluation with mock open-source LLM responses
- File I/O operations with various video/audio formats
- Error handling across components

### 3. System Tests (Top)
- **Coverage**: Full system behavior
- **Tools**: pytest, custom test runners
- **Focus**: Real-world usage scenarios

## Data-Driven Testing Strategy

### 1. Test Data Categories

#### A. Video/Audio Samples
- **Short clips** (30-60 seconds): Quick validation
- **Medium clips** (5-10 minutes): Standard processing
- **Long clips** (30+ minutes): Performance testing
- **Video formats**: MP4, MOV, AVI, MKV
- **Audio formats**: MP3, WAV, M4A (extracted from video)
- **Quality variations**: High/low bitrate, clear/noisy audio, different video codecs

#### B. Content Types
- **Educational content**: Data science tutorials
- **Mixed content**: Q&A sessions, discussions
- **Monologue**: Single speaker
- **Dialogue**: Multiple speakers
- **Technical jargon**: Domain-specific terminology

#### C. Edge Cases
- **Silent segments**: Audio gaps
- **Background noise**: Office, cafe environments
- **Accent variations**: Different English accents
- **Speed variations**: Fast/slow speech

### 2. Test Data Management

#### Directory Structure:
```
tests/
├── data/
│   ├── video/
│   │   ├── short_samples/
│   │   ├── medium_samples/
│   │   ├── long_samples/
│   │   └── edge_cases/
│   ├── audio/
│   │   ├── extracted_audio/
│   │   └── standalone_audio/
│   ├── expected_outputs/
│   │   ├── transcriptions/
│   │   ├── segments/
│   │   └── evaluations/
│   └── fixtures/
│       ├── mock_responses/
│       └── test_configs/
```

#### Data Versioning:
- Git LFS for large audio files
- MD5 checksums for data integrity
- Versioned test datasets

### 3. Automated Test Scenarios

#### A. Regression Testing
```python
# Example test structure
def test_video_processing_and_transcription():
    """Test video processing and transcription accuracy"""
    for video_file, expected_text in test_cases:
        result = process_video_file(video_file)
        assert similarity(result, expected_text) > 0.9

def test_segment_quality_scoring():
    """Test open-source LLM scoring consistency"""
    segments = generate_test_segments()
    scores = evaluate_segments(segments)
    assert all(0 <= score <= 1 for score in scores)
```

#### B. Performance Testing
```python
def test_processing_speed():
    """Test processing time for different video file sizes"""
    for size, expected_time in performance_baselines:
        start_time = time.time()
        process_video_file(size)
        assert time.time() - start_time < expected_time
```

#### C. Quality Assurance
```python
def test_content_evaluation_consistency():
    """Test that similar content gets similar scores"""
    similar_segments = generate_similar_content()
    scores = evaluate_segments(similar_segments)
    assert score_variance(scores) < 0.1
```

## Testing Tools and Frameworks

### 1. Core Testing Framework
- **pytest**: Main testing framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking capabilities
- **pytest-benchmark**: Performance testing

### 2. Data Testing Tools
- **pandas-testing**: DataFrame assertions
- **numpy-testing**: Array comparisons
- **audio-testing**: Audio file validation

### 3. CI/CD Integration
- **GitHub Actions**: Automated testing
- **Codecov**: Coverage tracking
- **SonarQube**: Code quality analysis

## Test Execution Strategy

### 1. Local Development
```bash
# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/integration/ -m "not slow"
```

### 2. CI/CD Pipeline
```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    pytest tests/unit/ --cov=src
    pytest tests/integration/ --cov=src
    pytest tests/system/ --cov=src
```

### 3. Performance Monitoring
- **Baseline tracking**: Performance regression detection
- **Resource monitoring**: Memory/CPU usage
- **Scalability testing**: Large file processing

## Quality Metrics

### 1. Code Quality
- **Coverage**: Minimum 90%
- **Complexity**: Cyclomatic complexity < 10
- **Duplication**: < 5% code duplication

### 2. System Quality
- **Accuracy**: Transcription accuracy > 95%
- **Performance**: Processing time < 2x real-time
- **Reliability**: 99%+ test pass rate

### 3. Data Quality
- **Consistency**: Similar inputs → similar outputs
- **Robustness**: Handles edge cases gracefully
- **Scalability**: Performance scales linearly

## Continuous Improvement

### 1. Test Data Evolution
- **Regular updates**: New audio samples
- **Feedback integration**: Real-world usage data
- **Edge case discovery**: Continuous edge case identification

### 2. Test Strategy Refinement
- **Performance optimization**: Faster test execution
- **Coverage gaps**: Identify and fill coverage holes
- **Tool evaluation**: Assess new testing tools

### 3. Documentation
- **Test documentation**: Clear test descriptions
- **Data documentation**: Test data sources and characteristics
- **Results tracking**: Historical test performance 