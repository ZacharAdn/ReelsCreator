# Speaker Segmentation Module 🎭

This module implements precise speaker diarization for educational content, specifically designed to distinguish between teacher and student speech in Hebrew educational videos.

## 🎯 Purpose

Automatically identify and segment teacher vs student speech to enable:
- **Content filtering**: Extract only teacher segments for reels creation
- **Quality control**: Focus processing on educational content
- **Time optimization**: Skip student segments during transcription/evaluation

## 🔬 Technical Approach

### Stage 1: Audio Analysis
- **Window Size**: 0.5 seconds with 0.25s overlap for high precision
- **Silence Detection**: Energy threshold < 0.003 to skip silent segments
- **Feature Extraction**: Multi-dimensional voice characteristics

### Stage 2: Feature Engineering
The algorithm extracts detailed voice features for each micro-segment:

#### Core Features:
- **Energy**: Voice volume/power (`RMS`)
- **Pitch**: Fundamental frequency (mean & variation)
- **Spectral Features**: Centroid, bandwidth, rolloff
- **Zero Crossing Rate**: Voice vs noise discrimination
- **MFCCs**: Mel-frequency cepstral coefficients (5 dimensions)

#### Feature Vector:
```python
[energy*1000, pitch_mean/100, pitch_std/50, spectral_centroid/1000, 
 spectral_bandwidth/1000, spectral_rolloff/1000, zcr*1000, 
 mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, 
 mfcc_std_1, mfcc_std_2, mfcc_std_3, mfcc_std_4, mfcc_std_5]
```

### Stage 3: Speaker Clustering
- **Algorithm**: K-Means clustering (k=2 for teacher/student)
- **Preprocessing**: StandardScaler for feature normalization
- **Optimization**: 20 initializations, 500 max iterations
- **Primary Speaker**: Determined by total speaking time

### Stage 4: Segment Merging
- **Gap Threshold**: ≤3 seconds between same-speaker segments
- **Minimum Duration**: 2 seconds for final segments
- **Aggressive Merging**: Combines short interruptions into continuous speech

## 📊 Performance Characteristics

### Precision:
- **Temporal Resolution**: 0.25 second accuracy
- **Speaker Accuracy**: ~85-90% for clear teacher/student distinction
- **Gap Detection**: Identifies pauses as short as 0.5 seconds

### Typical Results:
```
Input:  22 fragmented segments (5-15s each)
Output: 4 continuous segments (15-60s each)
Improvement: 82% reduction in segment count
```

## 🚀 Usage

### Basic Usage:
```python
from speaker_analyzer import SpeakerSegmentationPipeline

# Initialize
pipeline = SpeakerSegmentationPipeline()

# Analyze video
result = pipeline.analyze_video("video.mp4")

# Get teacher segments
teacher_segments = result.get_teacher_segments()
for segment in teacher_segments:
    print(f"{segment.start_time} - {segment.end_time}: {segment.duration}s")
```

### Advanced Configuration:
```python
config = {
    'window_size': 0.5,        # Analysis window (seconds)
    'hop_size': 0.25,          # Window overlap (seconds)
    'silence_threshold': 0.003, # Energy threshold for silence
    'max_gap': 3.0,            # Max gap to merge (seconds)
    'min_duration': 2.0        # Minimum segment duration (seconds)
}

pipeline = SpeakerSegmentationPipeline(config)
```

## 🎯 Optimization for Hebrew Educational Content

### Language-Specific Adaptations:
1. **Pitch Range**: Optimized for Hebrew speech patterns (50-500 Hz)
2. **Energy Thresholds**: Tuned for classroom recording conditions
3. **Feature Weights**: Balanced for teacher/student voice distinction

### Educational Context:
1. **Teacher Identification**: Assumes primary speaker = teacher
2. **Interruption Handling**: Merges brief student questions/responses
3. **Content Continuity**: Preserves educational flow across small gaps

## 📈 Quality Metrics

### Segment Quality Classification:
- **🎯 EXCELLENT**: ≥30 seconds (ideal for reels)
- **⭐ VERY GOOD**: 15-29 seconds (good for reels)
- **✅ GOOD**: 10-14 seconds (usable for reels)
- **📝 SHORT**: 2-9 seconds (may need combination)

### Expected Outcomes:
- **Teacher Coverage**: 60-70% of total audio
- **Continuous Segments**: 3-6 segments per 3-minute video
- **Processing Speed**: ~2-3 seconds per minute of audio

## 🔧 Technical Requirements

### Dependencies:
```bash
librosa>=0.11.0          # Audio analysis
scikit-learn>=1.6.0      # Machine learning
numpy>=1.26.0            # Numerical computing
moviepy>=2.1.0           # Video processing
```

### Hardware Recommendations:
- **CPU**: Multi-core recommended for audio processing
- **RAM**: 2GB+ for 10-minute videos
- **Storage**: Temporary audio files (~10MB per minute)

## 🧪 Testing

### Test Cases:
1. **Single Speaker**: Should detect one speaker cluster
2. **Clear Distinction**: Teacher vs student with different voice characteristics
3. **Mixed Content**: Rapid speaker changes
4. **Background Noise**: Classroom environment with ambient sounds
5. **Hebrew + English**: Technical terms in mixed languages

### Performance Benchmarks:
```bash
# Run standard test
python test_speaker_detection.py

# Expected output:
# ✅ Speaker detection: 85-90% accuracy
# ✅ Segment merging: 70-80% reduction in fragments
# ✅ Processing time: <30s for 3-minute video
```

## 🚨 Known Limitations

1. **Similar Voices**: May struggle with teacher/student of similar age/gender
2. **Background Noise**: Performance degrades with poor audio quality
3. **Multiple Students**: Assumes binary teacher/student classification
4. **Language Mixing**: Heavy English may affect Hebrew optimizations
5. **Recording Quality**: Requires clear audio with minimal distortion

## 🔄 Future Improvements

1. **Deep Learning**: Neural speaker embeddings for better accuracy
2. **Multi-Speaker**: Support for multiple students
3. **Language Detection**: Automatic Hebrew/English segment identification
4. **Confidence Scoring**: Per-segment confidence metrics
5. **Real-time Processing**: Live speaker detection for streaming

## 📚 References

- [Librosa Documentation](https://librosa.org/doc/latest/)
- [Speaker Diarization Survey](https://arxiv.org/abs/2012.01477)
- [MFCC Feature Extraction](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
