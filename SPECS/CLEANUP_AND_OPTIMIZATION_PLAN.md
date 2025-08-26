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
   - Setup files (`setup_environment.py`, `requirements_basic.txt`)
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

## 📊 Implementation Priority

1. **Quality Profile Performance (HIGH)**
   - Immediate impact on usability
   - Blocks effective system usage
   - Critical for production deployment

2. **Segment Uniformity (MEDIUM)**
   - Affects output quality
   - Important for user trust
   - Required for accurate content selection

3. **Project Cleanup (LOW)**
   - Improves maintainability
   - Better developer experience
   - Technical debt reduction

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
