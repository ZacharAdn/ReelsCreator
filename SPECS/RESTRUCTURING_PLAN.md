# Algorithm Restructuring Plan

## 🎯 Goal
Transform the current monolithic processing pipeline into a modular, stage-based architecture with clear separation of concerns, better testability, and easier debugging.

## 📊 Current State Analysis

### Current Structure Issues
- Mixed responsibilities in single files
- Hard to test individual components
- Difficult to debug specific stages
- No clear visualization/analysis separation
- ✅ Monolithic `content_extractor.py` restructured into stage-based architecture

### Current Processing Pipeline
```
Video → Audio → Speaker Analysis → Transcription → Segmentation → Embeddings → LLM_Evaluation → Results
```

## 🏗️ Proposed New Structure

### Stage-Based Architecture
```
src/
├── stages/
│   ├── 01_audio_extraction/
│   │   ├── code/
│   │   │   ├── __init__.py
│   │   │   ├── video_processor.py
│   │   │   └── audio_extractor.py
│   │   ├── test/
│   │   │   ├── test_video_processor.py
│   │   │   └── test_audio_extractor.py
│   │   ├── visualizations/
│   │   │   └── audio_waveform_plots.py
│   │   └── README.md
│   │
│   ├── _04_speaker_segmentation/
│   │   ├── code/
│   │   │   ├── __init__.py
│   │   │   ├── hybrid_detector.py
│   │   │   ├── frequency_analyzer.py
│   │   │   ├── refined_classifier.py
│   │   │   └── temporal_smoother.py
│   │   ├── test/
│   │   │   ├── test_hybrid_detector.py
│   │   │   ├── test_frequency_analyzer.py
│   │   │   └── test_integration.py
│   │   ├── visualizations/
│   │   │   ├── frequency_plots.py
│   │   │   ├── speaker_timeline.py
│   │   │   └── *.png (generated plots)
│   │   └── README.md
│   │
│   ├── 03_transcription/
│   │   ├── code/
│   │   │   ├── __init__.py
│   │   │   ├── whisper_transcriber.py
│   │   │   └── transcript_processor.py
│   │   ├── test/
│   │   │   └── test_transcription.py
│   │   ├── visualizations/
│   │   │   └── transcription_quality.py
│   │   └── README.md
│   │
│   ├── 04_content_segmentation/
│   │   ├── code/
│   │   │   ├── __init__.py
│   │   │   ├── segment_processor.py
│   │   │   └── overlap_manager.py
│   │   ├── test/
│   │   │   └── test_segmentation.py
│   │   ├── visualizations/
│   │   │   └── segment_timeline.py
│   │   └── README.md
│   │
│   ├── 05_content_evaluation/
│   │   ├── code/
│   │   │   ├── __init__.py
│   │   │   ├── rule_based_evaluator.py
│   │   │   ├── llm_evaluator.py
│   │   │   └── scoring_engine.py
│   │   ├── test/
│   │   │   └── test_evaluation.py
│   │   ├── visualizations/
│   │   │   └── score_distribution.py
│   │   └── README.md
│   │
│   └── 06_output_generation/
│       ├── code/
│       │   ├── __init__.py
│       │   ├── csv_exporter.py
│       │   ├── json_exporter.py
│       │   └── report_generator.py
│       ├── test/
│       │   └── test_exporters.py
│       ├── visualizations/
│       │   └── results_dashboard.py
│       └── README.md
│
├── orchestrator/
│   ├── __init__.py
│   ├── pipeline_orchestrator.py
│   ├── stage_manager.py
│   ├── config_manager.py
│   └── performance_monitor.py
│
├── shared/
│   ├── __init__.py
│   ├── models.py
│   ├── utils.py
│   ├── exceptions.py
│   └── logging_config.py
│
└── main.py (new entry point)
```

## 🔄 Implementation Steps

### Phase 1: Setup Infrastructure (Week 1)
1. **Create Stage Structure**
   ```bash
   mkdir -p src/stages/{_01_audio_extraction,_02_transcription,_03_content_segmentation,_04_speaker_segmentation,_05_content_evaluation,_06_output_generation}/{code,test,visualizations}
   ```

2. **Move Existing Files**
   - ✅ Move speaker segmentation files to `_04_speaker_segmentation/code/` (COMPLETED)
   - Move visualizations to appropriate `/visualizations/` folders
   - Move tests to `/test/` folders

3. **Create Base Classes**
   ```python
   # shared/base_stage.py
   class BaseStage:
       def __init__(self, config):
           self.config = config
           self.performance_metrics = {}
       
       def execute(self, input_data):
           raise NotImplementedError
       
       def validate_input(self, input_data):
           pass
       
       def get_metrics(self):
           return self.performance_metrics
   ```

### Phase 2: Migrate Core Components (Week 2)

#### 2.1 Audio Extraction Stage
```python
# stages/01_audio_extraction/code/video_processor.py
class VideoProcessor(BaseStage):
    def execute(self, video_path):
        # Extract audio, return audio_path + metadata
        pass
```

#### 2.2 Speaker Segmentation Stage (Already Started)
```python
# stages/_04_speaker_segmentation/code/__init__.py
from .hybrid_detector import HybridSpeakerDetector
from .frequency_analyzer import FrequencyAnalyzer

class SpeakerSegmentationStage(BaseStage):
    def execute(self, audio_path):
        # Return speaker segments + analysis
        pass
```

#### 2.3 Transcription Stage
```python
# stages/03_transcription/code/whisper_transcriber.py
class TranscriptionStage(BaseStage):
    def execute(self, audio_path, speaker_segments=None):
        # Return transcribed segments
        pass
```

### Phase 3: Create Pipeline Orchestrator (Week 2)

```python
# orchestrator/pipeline_orchestrator.py
class PipelineOrchestrator:
    def __init__(self, config):
        self.config = config
        self.stages = self._initialize_stages()
        self.performance_monitor = PerformanceMonitor()
    
    def _initialize_stages(self):
        return {
            'audio_extraction': AudioExtractionStage(self.config),
            'speaker_segmentation': SpeakerSegmentationStage(self.config),
            'transcription': TranscriptionStage(self.config),
            'content_segmentation': ContentSegmentationStage(self.config),
            'content_evaluation': ContentEvaluationStage(self.config),
            'output_generation': OutputGenerationStage(self.config)
        }
    
    def process_video(self, video_path):
        results = {}
        
        # Stage 1: Audio Extraction
        audio_data = self.stages['audio_extraction'].execute(video_path)
        results['audio'] = audio_data
        
        # Stage 2: Speaker Segmentation  
        if self.config.enable_speaker_detection:
            speaker_data = self.stages['speaker_segmentation'].execute(audio_data['path'])
            results['speakers'] = speaker_data
        
        # Continue through pipeline...
        return results
```

### Phase 4: Enhanced Testing & Visualization (Week 3)

#### 4.1 Stage-Specific Tests
```python
# stages/_04_speaker_segmentation/test/test_hybrid_detector.py
class TestHybridDetector:
    def test_frequency_analysis(self):
        # Test frequency analysis accuracy
        pass
    
    def test_speaker_classification(self):
        # Test speaker detection accuracy
        pass
    
    def test_temporal_smoothing(self):
        # Test smoothing effectiveness
        pass
```

#### 4.2 Interactive Visualizations
```python
# stages/_04_speaker_segmentation/visualizations/speaker_timeline.py
def create_interactive_timeline(speaker_segments, audio_path):
    # Create plotly timeline with audio playback
    pass

def generate_accuracy_report(ground_truth, detected):
    # Generate accuracy metrics and plots
    pass
```

## 🎯 New Entry Point

```python
# main.py
from orchestrator.pipeline_orchestrator import PipelineOrchestrator
from orchestrator.config_manager import ConfigManager

def main():
    config = ConfigManager.load_from_args()
    orchestrator = PipelineOrchestrator(config)
    
    result = orchestrator.process_video(config.video_path)
    
    # Generate reports
    orchestrator.generate_performance_report()
    orchestrator.save_results(result)

if __name__ == "__main__":
    main()
```

## 📈 Benefits

### 1. **Better Debugging**
- Each stage can be tested independently
- Clear input/output contracts
- Isolated performance monitoring

### 2. **Improved Testing**
- Unit tests for each component
- Integration tests for stage combinations
- Visual regression testing

### 3. **Enhanced Visualization**
- Stage-specific visualizations
- Interactive debugging tools
- Performance dashboards

### 4. **Easier Development**
- Clear separation of concerns
- Pluggable architecture
- Easy to add new stages

### 5. **Better Error Handling**
- Stage-specific error recovery
- Clear failure points
- Detailed error reporting

## 🚀 Migration Strategy

### Immediate Actions
1. Create the folder structure
2. Move existing speaker segmentation code
3. Create basic orchestrator
4. Test with current functionality

### Gradual Migration
1. One stage at a time
2. Maintain backward compatibility
3. Add tests as you migrate
4. Create visualizations incrementally

## 🔧 Development Tools

### Testing Framework
```bash
# Run stage-specific tests
pytest src/stages/_04_speaker_segmentation/test/

# Run integration tests  
pytest tests/integration/

# Generate coverage report
pytest --cov=src/stages/
```

### Visualization Tools
```bash
# Generate stage visualizations
python -m src.stages._04_speaker_segmentation.visualizations.speaker_timeline

# Create performance dashboard
python -m orchestrator.performance_dashboard
```

This restructured approach will make the algorithm much more maintainable, testable, and easier to improve iteratively.