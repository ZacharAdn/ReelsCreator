"""
Refined classification system with data-driven dynamic thresholds
Implementation of the classification improvements from TECHNICAL_ANALYSIS.md
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .frequency_analyzer import FrequencyAnalyzer, FrequencyFeatures, DynamicThresholds

logger = logging.getLogger(__name__)


@dataclass
class TemplateFeatures:
    """Template features for teacher/student baseline comparison"""
    speaker_type: str  # "TEACHER" or "STUDENT"
    features: FrequencyFeatures
    confidence: float
    time_range: Tuple[float, float]  # (start, end) in seconds


@dataclass
class ClassificationResult:
    """Result of speaker classification"""
    speaker: str
    confidence: float
    reasoning: str
    template_distance: float
    features: FrequencyFeatures
    time_sec: float


class RefinedSpeakerClassifier:
    """
    Refined speaker classifier with template-based scoring and adaptive thresholds
    """
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize refined classifier
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.frequency_analyzer = FrequencyAnalyzer(sample_rate)
        self.teacher_templates: List[TemplateFeatures] = []
        self.student_templates: List[TemplateFeatures] = []
        self.is_trained = False
        
        logger.info("Initialized RefinedSpeakerClassifier")
    
    def add_template(self, audio_segment: np.ndarray, speaker_type: str, 
                    confidence: float, time_range: Tuple[float, float]) -> None:
        """
        Add a template for baseline comparison
        
        Args:
            audio_segment: Audio data
            speaker_type: "TEACHER" or "STUDENT"
            confidence: Confidence in the template labeling
            time_range: Time range of the template
        """
        features = self.frequency_analyzer.extract_frequency_features(audio_segment)
        template = TemplateFeatures(
            speaker_type=speaker_type,
            features=features,
            confidence=confidence,
            time_range=time_range
        )
        
        if speaker_type == "TEACHER":
            self.teacher_templates.append(template)
        elif speaker_type == "STUDENT":
            self.student_templates.append(template)
        
        logger.debug(f"Added {speaker_type} template from {time_range[0]:.1f}-{time_range[1]:.1f}s")
    
    def train_from_middle_portion(self, audio_data: np.ndarray, total_duration: float,
                                 initial_labels: List[Dict]) -> None:
        """
        Train templates from the middle portion of the recording where classification is more reliable
        
        Args:
            audio_data: Complete audio data
            total_duration: Total duration in seconds
            initial_labels: Initial speaker labels from pyannote or other source
        """
        # Focus on middle 50% of recording for training templates
        start_time = total_duration * 0.25
        end_time = total_duration * 0.75
        
        # Extract high-confidence segments from middle portion
        middle_labels = [
            label for label in initial_labels
            if (label.get('start_time', 0) >= start_time and 
                label.get('end_time', 0) <= end_time and
                label.get('confidence', 0) > 0.8)
        ]
        
        for label in middle_labels:
            start_idx = int(label['start_time'] * self.sample_rate)
            end_idx = int(label['end_time'] * self.sample_rate)
            
            if start_idx < len(audio_data) and end_idx <= len(audio_data):
                segment_audio = audio_data[start_idx:end_idx]
                self.add_template(
                    segment_audio,
                    label['speaker'],
                    label['confidence'],
                    (label['start_time'], label['end_time'])
                )
        
        # Learn dynamic thresholds from teacher templates
        if self.teacher_templates:
            teacher_segments = [
                # Create mock audio segments for the analyzer
                np.random.normal(0, t.features.energy, int(self.sample_rate * 2))
                for t in self.teacher_templates
            ]
            self.frequency_analyzer.learn_teacher_baseline(teacher_segments)
        
        self.is_trained = True
        logger.info(f"🎓 Trained classifier with {len(self.teacher_templates)} teacher and {len(self.student_templates)} student templates")
    
    def calculate_template_distance(self, features: FrequencyFeatures, 
                                  templates: List[TemplateFeatures]) -> float:
        """
        Calculate distance to closest template
        
        Args:
            features: Features to compare
            templates: List of template features
            
        Returns:
            Minimum distance to templates (lower = more similar)
        """
        if not templates:
            return 1.0  # Maximum distance if no templates
        
        distances = []
        for template in templates:
            # Weighted Euclidean distance
            distance = np.sqrt(
                0.3 * (features.mid_ratio - template.features.mid_ratio) ** 2 +
                0.3 * (features.high_ratio - template.features.high_ratio) ** 2 +
                0.2 * (features.energy - template.features.energy) ** 2 +
                0.1 * (features.stability - template.features.stability) ** 2 +
                0.1 * (features.pitch_variation - template.features.pitch_variation) ** 2
            )
            # Weight by template confidence
            weighted_distance = distance / template.confidence
            distances.append(weighted_distance)
        
        return min(distances)
    
    def classify_with_templates(self, features: FrequencyFeatures, time_sec: float) -> ClassificationResult:
        """
        Classify speaker using template-based approach with adaptive confidence
        
        Args:
            features: Extracted frequency features
            time_sec: Time position in recording
            
        Returns:
            ClassificationResult with speaker and confidence
        """
        # Calculate distances to templates
        teacher_distance = self.calculate_template_distance(features, self.teacher_templates)
        student_distance = self.calculate_template_distance(features, self.student_templates)
        
        # Base classification using frequency analyzer
        base_speaker, base_confidence = self.frequency_analyzer.classify_speaker_data_driven(features, time_sec)
        
        # Template-based refinement
        template_confidence = 0.5  # Default
        template_speaker = base_speaker
        
        if self.is_trained and (teacher_distance < 0.5 or student_distance < 0.5):
            # Strong template match
            if teacher_distance < student_distance:
                template_speaker = "TEACHER"
                template_confidence = 1.0 - teacher_distance
            else:
                template_speaker = "STUDENT"
                template_confidence = 1.0 - student_distance
            
            # Combine base and template predictions
            if base_speaker == template_speaker:
                # Agreement - boost confidence
                final_confidence = min(1.0, (base_confidence + template_confidence) / 2 * 1.2)
                final_speaker = base_speaker
            else:
                # Disagreement - use template if high confidence, otherwise reduce confidence
                if template_confidence > 0.7:
                    final_speaker = template_speaker
                    final_confidence = template_confidence * 0.8  # Slightly reduce for disagreement
                else:
                    final_speaker = base_speaker
                    final_confidence = base_confidence * 0.6  # Reduce for uncertainty
        else:
            # No strong template match - use base classification
            final_speaker = base_speaker
            final_confidence = base_confidence * 0.9  # Slight reduction without template support
        
        # Generate reasoning
        reasoning = self._generate_template_reasoning(
            features, final_speaker, final_confidence, time_sec,
            teacher_distance, student_distance, base_speaker, template_speaker
        )
        
        return ClassificationResult(
            speaker=final_speaker,
            confidence=final_confidence,
            reasoning=reasoning,
            template_distance=min(teacher_distance, student_distance),
            features=features,
            time_sec=time_sec
        )
    
    def _generate_template_reasoning(self, features: FrequencyFeatures, speaker: str, 
                                   confidence: float, time_sec: float,
                                   teacher_dist: float, student_dist: float,
                                   base_speaker: str, template_speaker: str) -> str:
        """Generate detailed reasoning for template-based classification"""
        reasons = []
        
        # Template distance information
        if teacher_dist < 0.5:
            reasons.append(f"close to teacher template ({teacher_dist:.2f})")
        if student_dist < 0.5:
            reasons.append(f"close to student template ({student_dist:.2f})")
        
        # Feature-based reasons
        if features.mid_ratio < 0.35:
            reasons.append("low mid-freq")
        if features.high_ratio > 0.55:
            reasons.append("high high-freq")
        if features.energy < 0.01:
            reasons.append("low energy")
        
        # Agreement/disagreement
        if base_speaker != template_speaker and self.is_trained:
            reasons.append(f"base={base_speaker}, template={template_speaker}")
        
        # Time-based factors
        if time_sec < 5:
            reasons.append("early period")
        
        reasoning = f"{speaker} ({confidence:.2f}): " + ", ".join(reasons) if reasons else f"{speaker} ({confidence:.2f}): standard features"
        
        return reasoning
    
    def classify_segment(self, audio_segment: np.ndarray, time_sec: float) -> ClassificationResult:
        """
        Main classification method for audio segment
        
        Args:
            audio_segment: Audio data to classify
            time_sec: Time position in recording
            
        Returns:
            ClassificationResult
        """
        features = self.frequency_analyzer.extract_frequency_features(audio_segment)
        return self.classify_with_templates(features, time_sec)
    
    def get_training_summary(self) -> Dict:
        """Get summary of training templates and thresholds"""
        return {
            "is_trained": self.is_trained,
            "teacher_templates": len(self.teacher_templates),
            "student_templates": len(self.student_templates),
            "dynamic_thresholds": {
                "mid_low": self.frequency_analyzer.dynamic_thresholds.mid_low,
                "mid_high": self.frequency_analyzer.dynamic_thresholds.mid_high,
                "high_high": self.frequency_analyzer.dynamic_thresholds.high_high,
                "energy_baseline": self.frequency_analyzer.dynamic_thresholds.energy_teacher_median
            }
        }
    
    def reassign_early_segments(self, segments_results: List[ClassificationResult], 
                              confidence_threshold: float = 0.6) -> List[ClassificationResult]:
        """
        Reassign early segments with low confidence using template distances
        
        Args:
            segments_results: List of classification results
            confidence_threshold: Minimum confidence for keeping original classification
            
        Returns:
            List with reassigned segments
        """
        if not self.is_trained:
            return segments_results
        
        reassigned_results = []
        
        for result in segments_results:
            # Focus on early segments (0-10 seconds) with low confidence
            if result.time_sec < 10.0 and result.confidence < confidence_threshold:
                # Reassign based purely on template distance
                teacher_dist = self.calculate_template_distance(result.features, self.teacher_templates)
                student_dist = self.calculate_template_distance(result.features, self.student_templates)
                
                if abs(teacher_dist - student_dist) > 0.2:  # Significant difference
                    if teacher_dist < student_dist:
                        new_speaker = "TEACHER"
                        new_confidence = min(0.8, 1.0 - teacher_dist)
                    else:
                        new_speaker = "STUDENT"
                        new_confidence = min(0.8, 1.0 - student_dist)
                    
                    # Create new result
                    new_result = ClassificationResult(
                        speaker=new_speaker,
                        confidence=new_confidence,
                        reasoning=f"Reassigned early segment: {result.reasoning}",
                        template_distance=min(teacher_dist, student_dist),
                        features=result.features,
                        time_sec=result.time_sec
                    )
                    reassigned_results.append(new_result)
                    continue
            
            # Keep original result
            reassigned_results.append(result)
        
        # Count reassignments
        reassigned_count = sum(1 for orig, new in zip(segments_results, reassigned_results) 
                             if orig.speaker != new.speaker)
        
        if reassigned_count > 0:
            logger.info(f"🔄 Reassigned {reassigned_count} early segments using template distances")
        
        return reassigned_results