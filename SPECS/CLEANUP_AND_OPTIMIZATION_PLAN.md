# 🎯 Cleanup and Optimization Plan

## Overview
This document outlines three critical improvements needed for the Reels_extractor project:
1. Project cleanup and reorganization
2. Fixing segment uniformity issues
3. Resolving quality profile performance problems

## 🧹 Task 1: Project Cleanup and Reorganization

### Current Issues
- Scattered files and duplicated code
- Unnecessary cache and temporary files
- Unorganized documentation
- Inconsistent project structure

### Proposed Directory Structure
```
reels_extractor/
├── src/
│   ├── stages/          # All processing stages
│   ├── orchestrator/    # Pipeline coordination
│   └── shared/          # Common utilities
├── docs/                # All documentation
├── tests/               # Unified test suite
├── data/
│   ├── examples/        # Sample videos
│   └── models/         # Model weights
└── results/            # Processing outputs
```

### Cleanup Tasks
1. **Remove Unnecessary Files**
   - All `__pycache__` directories
   - Temporary files (`test_frame.png`, `test_video.mp4`)
   - Setup files (✅ `setup_environment.py`, `requirements_basic.txt` already removed)
   - Legacy `fixes/` directory
   - Old results in `results/`

2. **Code Reorganization**
   - Move duplicate files from `src/` to appropriate `stages/`
   - Consolidate test files into unified structure
   - Create proper documentation directory
   - Clean up duplicate implementations

3. **Update .gitignore**
   - Add missing environment paths
   - Organize by categories (Python, Media, Cache)
   - Add specific caches (Whisper, HuggingFace)

## 🔍 Task 2: Fix Segment Uniformity Issue

### Current Problems
- All segments receive identical 0.75 score
- Fixed 45-second segment lengths
- No quality variation detection
- Test suite doesn't catch uniformity issues

### Proposed Solutions

#### A. Enhanced Evaluation Algorithm
```python
class ContentEvaluator:
    def evaluate_segment(self, segment: Segment) -> float:
        # Multiple evaluation criteria
        clarity_score = self.evaluate_clarity(segment)
        interest_score = self.evaluate_interest(segment)
        info_score = self.evaluate_information_value(segment)
        audio_score = self.evaluate_audio_quality(segment)
        
        # Dynamic weights based on content type
        weights = self.get_content_type_weights(segment)
        
        # Relative scoring within video context
        return self.calculate_weighted_score(
            [clarity_score, interest_score, info_score, audio_score],
            weights,
            context=self.video_segments
        )
```

#### B. Smart Segmentation
```python
class ContentSegmentation:
    def create_segments(self, content: VideoContent) -> List[Segment]:
        # Topic-based segmentation
        topic_boundaries = self.detect_topic_changes(content)
        
        # Interest point detection
        interest_points = self.detect_high_interest_points(content)
        
        # Dynamic length based on content quality
        return self.optimize_segment_boundaries(
            topic_boundaries,
            interest_points,
            min_length=15,
            max_length=45
        )
```

#### C. Quality Tests
```python
class SegmentQualityTests:
    def test_score_variance(self, segments: List[Segment]):
        scores = [s.score for s in segments]
        variance = np.var(scores)
        assert variance > 0.1, "Scores show no variation"
        
    def test_length_variance(self, segments: List[Segment]):
        lengths = [s.duration for s in segments]
        unique_lengths = set(lengths)
        assert len(unique_lengths) > 2, "Segment lengths too uniform"
```

## ⚡ Task 3: Quality Profile Performance Issues

### Current Problems
- LLM model loading hangs
- No progress feedback
- No timeout mechanisms
- Memory management issues

### Proposed Solutions

#### A. LLM Management
```python
class LLMManager:
    def load_model(self, model_name: str, timeout: int = 300):
        try:
            with timeout_decorator.timeout(timeout):
                model = self.cached_models.get(model_name)
                if not model:
                    self.clear_gpu_memory()
                    model = self.load_with_fallback(model_name)
                return model
        except TimeoutError:
            return self.fallback_to_smaller_model(model_name)
```

#### B. Progress Monitoring
```python
class ProgressMonitor:
    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn()
        )
        
    def track_stage(self, stage_name: str, total: int):
        return self.progress.add_task(
            f"[cyan]{stage_name}",
            total=total,
            visible=True
        )
```

#### C. Performance Optimizations
```python
class QualityProfileOptimizer:
    def process_batch(self, segments: List[Segment]):
        # Parallel processing with GPU optimization
        async with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.process_segment,
                    segment,
                    device=self.get_optimal_device()
                )
                for segment in segments
            ]
            return await asyncio.gather(*futures)
```

## 📊 Implementation Priority & Roadmap

### **🔥 SPRINT 1 (Week 1) - Critical Production Blockers**

#### 1. **Quality Profile Performance Hang (URGENT)**
- **Timeline**: 3-5 days
- **Success Criteria**: 
  - Quality profile completes 5-minute video processing in <8 minutes
  - No indefinite hangs during LLM model loading
  - Graceful fallback to smaller models when loading fails
  - Progress indicators show model loading status
- **Deliverables**: 
  - Timeout decorators (300s max)
  - Model fallback hierarchy (Qwen2.5→Phi-3→DialoGPT)
  - Rich progress bars for loading stages

#### 2. **Segment Quality Uniformity (URGENT)**  
- **Timeline**: 4-6 days
- **Success Criteria**:
  - Score variance >0.1 across segments in typical video
  - Realistic score distribution (not all 0.75)
  - Top 20% segments clearly distinguishable from bottom 20%
  - Quality reasoning varies meaningfully between segments
- **Deliverables**:
  - Enhanced evaluation algorithm with relative scoring
  - Context-aware quality assessment
  - Multi-criteria evaluation (clarity, interest, information, audio)

### **📈 SPRINT 2 (Week 2) - Stabilization**

#### 3. **Project File Organization Completion**
- **Timeline**: 2-3 days  
- **Success Criteria**:
  - All documentation links functional
  - No references to deleted/moved files
  - Clear directory structure documentation
  - Developer onboarding guide updated
- **Status**: ✅ 80% complete (documentation updated)

### **🚀 SPRINT 3 (Week 3+) - Advanced Features**

#### 4. **Python 3.9+ Migration & Full Speaker Diarization**
- **Timeline**: 1-2 weeks
- **Success Criteria**:
  - pyannote.audio successfully installed and working
  - Teacher/student role detection >85% accuracy
  - Speaker confidence scores >0.85 for primary speaker
  - Advanced speaker features accessible via CLI
- **Prerequisite**: Python environment upgrade

#### 5. **Performance Benchmarking & Monitoring**
- **Timeline**: 1 week
- **Success Criteria**:
  - Automated performance regression tests
  - Memory usage tracking <8GB for quality profile
  - Processing speed benchmarks for all profiles
  - Performance dashboard for monitoring

## 🔧 Technical Requirements

### Dependencies
```requirements
psutil>=5.8.0     # System monitoring
rich>=10.0.0      # Progress display
timeout-decorator>=0.5.0  # Function timeouts
```

### Performance Targets
- Processing: <5 minutes for 3-minute video
- Memory: <8GB RAM usage
- GPU: Efficient MPS utilization on M1
- Progress: Updates every 10 seconds

### Testing Requirements
- CI/CD pipeline with performance tests
- Regression tests for segment quality
- Memory leak detection
- GPU memory monitoring

