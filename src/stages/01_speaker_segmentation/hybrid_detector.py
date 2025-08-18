"""
Hybrid speaker detection combining pyannote with frequency analysis
Main integration point for enhanced speaker segmentation
"""

import logging
import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass

from .frequency_analyzer import FrequencyAnalyzer
from .refined_classifier import RefinedSpeakerClassifier, ClassificationResult
from .temporal_smoother import TemporalSmoother, SmoothedResult

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """Result from hybrid speaker detection"""
    segments: List[Dict[str, Any]]
    processing_summary: Dict[str, Any]
    accuracy_improvements: Dict[str, Any]
    fallback_used: bool


class HybridSpeakerDetector:
    """
    Hybrid speaker detection system combining multiple approaches:
    1. PyAnnote for initial speaker diarization
    2. Frequency analysis for refinement  
    3. Template-based classification for accuracy
    4. Temporal smoothing for consistency
    """
    
    def __init__(self, sample_rate: int = 16000, enable_pyannote: bool = True):
        """
        Initialize hybrid detector
        
        Args:
            sample_rate: Audio sample rate
            enable_pyannote: Whether to use pyannote (fallback to frequency-only if False)
        """
        self.sample_rate = sample_rate
        self.enable_pyannote = enable_pyannote
        
        # Initialize components
        self.frequency_analyzer = FrequencyAnalyzer(sample_rate)
        self.refined_classifier = RefinedSpeakerClassifier(sample_rate)
        self.temporal_smoother = TemporalSmoother()
        
        # PyAnnote pipeline (lazy loading)
        self.pyannote_pipeline = None
        self.pyannote_available = False
        
        logger.info(f"Initialized HybridSpeakerDetector (sample_rate={sample_rate}, pyannote={enable_pyannote})")
    
    def _load_pyannote(self):
        """Lazy load PyAnnote pipeline"""
        if self.pyannote_pipeline is None and self.enable_pyannote:
            try:
                from pyannote.audio import Pipeline
                self.pyannote_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=None
                )
                self.pyannote_available = True
                logger.info("✅ PyAnnote pipeline loaded successfully")
            except Exception as e:
                logger.warning(f"❌ Failed to load PyAnnote: {e}")
                logger.info("🔄 Falling back to frequency-only analysis")
                self.pyannote_available = False
    
    def get_pyannote_segments(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Get initial speaker segments from PyAnnote
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of speaker segments with times and labels
        """
        self._load_pyannote()
        
        if not self.pyannote_available:
            return []
        
        try:
            logger.info("🎙️ Running PyAnnote speaker diarization...")
            diarization = self.pyannote_pipeline(audio_path)
            
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start_time': turn.start,
                    'end_time': turn.end,
                    'duration': turn.duration,
                    'speaker': 'TEACHER' if speaker == diarization.labels()[0] else 'STUDENT',  # Assume first speaker is teacher
                    'confidence': 0.7,  # Default PyAnnote confidence
                    'source': 'pyannote'
                })
            
            logger.info(f"📊 PyAnnote found {len(segments)} speaker segments")
            return segments
            
        except Exception as e:
            logger.error(f"❌ PyAnnote analysis failed: {e}")
            return []
    
    def create_frequency_segments(self, audio_path: str, segment_duration: float = 3.0) -> List[Dict[str, Any]]:
        """
        Create segments for frequency analysis when PyAnnote is not available
        
        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            List of segments for analysis
        """
        # Load audio to get total duration
        audio_data, _ = librosa.load(audio_path, sr=self.sample_rate)
        total_duration = len(audio_data) / self.sample_rate
        
        segments = []
        current_time = 0.0
        
        while current_time < total_duration:
            end_time = min(current_time + segment_duration, total_duration)
            
            segments.append({
                'start_time': current_time,
                'end_time': end_time,
                'duration': end_time - current_time,
                'speaker': 'UNKNOWN',  # Will be classified
                'confidence': 0.5,
                'source': 'frequency_grid'
            })
            
            current_time += segment_duration
        
        logger.info(f"📊 Created {len(segments)} frequency analysis segments")
        return segments
    
    def refine_segments_with_frequency_analysis(self, audio_path: str, 
                                              initial_segments: List[Dict[str, Any]]) -> List[ClassificationResult]:
        """
        Refine speaker segments using frequency analysis
        
        Args:
            audio_path: Path to audio file
            initial_segments: Initial segments from PyAnnote or grid
            
        Returns:
            List of refined classification results
        """
        # Load audio data
        audio_data, _ = librosa.load(audio_path, sr=self.sample_rate)
        total_duration = len(audio_data) / self.sample_rate
        
        # Train classifier from middle portion if we have PyAnnote results
        if initial_segments and initial_segments[0].get('source') == 'pyannote':
            logger.info("🎓 Training classifier from PyAnnote middle-portion segments...")
            self.refined_classifier.train_from_middle_portion(audio_data, total_duration, initial_segments)
        
        # Classify each segment
        classification_results = []
        
        for segment in initial_segments:
            start_idx = int(segment['start_time'] * self.sample_rate)
            end_idx = int(segment['end_time'] * self.sample_rate)
            
            # Extract audio segment
            if start_idx < len(audio_data) and end_idx <= len(audio_data):
                segment_audio = audio_data[start_idx:end_idx]
                
                # Classify using refined classifier
                result = self.refined_classifier.classify_segment(segment_audio, segment['start_time'])
                classification_results.append(result)
            else:
                # Handle edge case
                logger.warning(f"⚠️ Segment {segment['start_time']:.1f}-{segment['end_time']:.1f}s out of bounds")
        
        logger.info(f"🔍 Classified {len(classification_results)} segments using frequency analysis")
        return classification_results
    
    def apply_corrections(self, classification_results: List[ClassificationResult]) -> List[ClassificationResult]:
        """
        Apply specific corrections for known problem areas
        
        Args:
            classification_results: Raw classification results
            
        Returns:
            Corrected classification results
        """
        # Apply early segment reassignment for 0-4s problem
        corrected_results = self.refined_classifier.reassign_early_segments(
            classification_results, confidence_threshold=0.6
        )
        
        # Apply additional corrections for mid-video student periods
        # (You can add more specific corrections here based on your analysis)
        
        corrections_applied = sum(1 for orig, corr in zip(classification_results, corrected_results)
                                if orig.speaker != corr.speaker)
        
        if corrections_applied > 0:
            logger.info(f"🔧 Applied {corrections_applied} specific corrections")
        
        return corrected_results
    
    def analyze_audio_file(self, audio_path: str, use_temporal_smoothing: bool = True) -> HybridResult:
        """
        Complete hybrid speaker analysis pipeline
        
        Args:
            audio_path: Path to audio file
            use_temporal_smoothing: Whether to apply temporal smoothing
            
        Returns:
            HybridResult with final speaker segments
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        logger.info(f"🎯 Starting hybrid speaker analysis: {audio_path}")
        
        # Step 1: Get initial segments (PyAnnote or frequency grid)
        initial_segments = self.get_pyannote_segments(audio_path)
        fallback_used = False
        
        if not initial_segments:
            logger.info("🔄 PyAnnote unavailable, using frequency-only analysis")
            initial_segments = self.create_frequency_segments(audio_path)
            fallback_used = True
        
        # Step 2: Refine with frequency analysis
        classification_results = self.refine_segments_with_frequency_analysis(audio_path, initial_segments)
        
        # Step 3: Apply specific corrections
        corrected_results = self.apply_corrections(classification_results)
        
        # Step 4: Apply temporal smoothing
        if use_temporal_smoothing:
            smoothed_results = self.temporal_smoother.apply_temporal_smoothing(
                corrected_results, method="combined"
            )
            final_segments = self._convert_to_segments(smoothed_results)
            smoothing_summary = self.temporal_smoother.get_smoothing_summary(corrected_results, smoothed_results)
        else:
            final_segments = self._convert_to_segments(corrected_results)
            smoothing_summary = {"smoothing": "disabled"}
        
        # Generate processing summary
        processing_summary = {
            "total_segments": len(final_segments),
            "pyannote_used": not fallback_used,
            "classifier_trained": self.refined_classifier.is_trained,
            "training_summary": self.refined_classifier.get_training_summary(),
            "smoothing_summary": smoothing_summary
        }
        
        # Calculate accuracy improvements (compared to PyAnnote baseline)
        accuracy_improvements = self._calculate_improvements(initial_segments, final_segments)
        
        logger.info(f"✅ Hybrid analysis complete: {len(final_segments)} final segments")
        
        return HybridResult(
            segments=final_segments,
            processing_summary=processing_summary,
            accuracy_improvements=accuracy_improvements,
            fallback_used=fallback_used
        )
    
    def _convert_to_segments(self, results: List) -> List[Dict[str, Any]]:
        """Convert classification/smoothed results to segment format"""
        segments = []
        
        for i, result in enumerate(results):
            # Determine end time (approximate from next segment or duration)
            if i < len(results) - 1:
                duration = results[i + 1].time_sec - result.time_sec
            else:
                duration = 3.0  # Default duration for last segment
            
            segment = {
                'start_time': result.time_sec,
                'end_time': result.time_sec + duration,
                'duration': duration,
                'speaker': result.speaker,
                'confidence': result.confidence,
                'reasoning': getattr(result, 'reasoning', ''),
                'source': 'hybrid_analysis'
            }
            
            # Add additional metadata if available
            if hasattr(result, 'smoothing_applied'):
                segment['smoothing_applied'] = result.smoothing_applied
                segment['original_speaker'] = result.original_speaker
            
            segments.append(segment)
        
        return segments
    
    def _calculate_improvements(self, initial_segments: List[Dict], final_segments: List[Dict]) -> Dict[str, Any]:
        """Calculate accuracy improvements over baseline"""
        if not initial_segments or not final_segments:
            return {"error": "insufficient_data"}
        
        # Count speaker transitions (fewer = more stable)
        def count_transitions(segments):
            transitions = 0
            for i in range(1, len(segments)):
                if segments[i]['speaker'] != segments[i-1]['speaker']:
                    transitions += 1
            return transitions
        
        initial_transitions = count_transitions(initial_segments)
        final_transitions = count_transitions(final_segments)
        
        # Count confidence improvements
        initial_avg_conf = np.mean([s.get('confidence', 0.5) for s in initial_segments])
        final_avg_conf = np.mean([s.get('confidence', 0.5) for s in final_segments])
        
        return {
            "initial_segments": len(initial_segments),
            "final_segments": len(final_segments),
            "initial_transitions": initial_transitions,
            "final_transitions": final_transitions,
            "transitions_reduced": max(0, initial_transitions - final_transitions),
            "initial_avg_confidence": initial_avg_conf,
            "final_avg_confidence": final_avg_conf,
            "confidence_improvement": final_avg_conf - initial_avg_conf
        }
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of detector capabilities and status"""
        return {
            "pyannote_available": self.pyannote_available,
            "frequency_analyzer": "ready",
            "refined_classifier": {
                "trained": self.refined_classifier.is_trained,
                "templates": self.refined_classifier.get_training_summary()
            },
            "temporal_smoother": "ready",
            "sample_rate": self.sample_rate
        }