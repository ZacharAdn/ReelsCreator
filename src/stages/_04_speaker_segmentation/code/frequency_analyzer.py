"""
Enhanced frequency analysis for speaker segmentation
Based on TECHNICAL_ANALYSIS.md findings and proposed solutions
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.fft import fft, fftfreq
import librosa

logger = logging.getLogger(__name__)


@dataclass
class FrequencyFeatures:
    """Container for frequency-based features"""
    low_ratio: float      # 50-200 Hz energy ratio
    mid_ratio: float      # 200-1000 Hz energy ratio  
    high_ratio: float     # 1000-4000 Hz energy ratio
    energy: float         # Overall energy level
    stability: float      # Frequency stability measure
    pitch_variation: float # Pitch variation measure


@dataclass
class DynamicThresholds:
    """Dynamic thresholds learned from teacher baseline"""
    mid_low: float = 0.35      # Mid-frequency low threshold for student detection
    mid_high: float = 0.50     # Mid-frequency high threshold for teacher detection
    high_low: float = 0.20     # High-frequency low threshold
    high_mid: float = 0.50     # High-frequency mid threshold  
    high_high: float = 0.55    # High-frequency high threshold for student detection
    energy_teacher_median: float = 0.02  # Teacher baseline energy
    confidence_threshold: float = 0.7     # Minimum confidence for classification


class FrequencyAnalyzer:
    """Enhanced frequency analyzer for speaker classification"""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize frequency analyzer
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
        self.dynamic_thresholds = DynamicThresholds()
        logger.info(f"Initialized FrequencyAnalyzer with sample_rate={sample_rate}")
    
    def extract_frequency_features(self, audio_segment: np.ndarray) -> FrequencyFeatures:
        """
        Extract enhanced frequency features from audio segment
        
        Args:
            audio_segment: Audio data as numpy array
            
        Returns:
            FrequencyFeatures object with extracted features
        """
        # Ensure audio_segment is not empty
        if len(audio_segment) == 0:
            return FrequencyFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        # Apply window to reduce spectral leakage
        windowed_segment = audio_segment * np.hanning(len(audio_segment))
        
        # FFT analysis
        freqs = fftfreq(len(windowed_segment), 1/self.sample_rate)
        fft_vals = np.abs(fft(windowed_segment))
        
        # Only use positive frequencies
        positive_freq_mask = freqs >= 0
        freqs = freqs[positive_freq_mask]
        fft_vals = fft_vals[positive_freq_mask]
        
        # Define frequency bands (as specified in TECHNICAL_ANALYSIS.md)
        low_band_mask = (freqs >= 50) & (freqs <= 200)     # Low: 50-200 Hz
        mid_band_mask = (freqs >= 200) & (freqs <= 1000)   # Mid: 200-1000 Hz  
        high_band_mask = (freqs >= 1000) & (freqs <= 4000) # High: 1000-4000 Hz
        
        # Calculate energy in each band
        low_energy = np.sum(fft_vals[low_band_mask])
        mid_energy = np.sum(fft_vals[mid_band_mask])
        high_energy = np.sum(fft_vals[high_band_mask])
        
        # Total energy in relevant frequency range
        total_energy = low_energy + mid_energy + high_energy
        
        # Calculate ratios (avoid division by zero)
        if total_energy > 0:
            low_ratio = low_energy / total_energy
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy
        else:
            low_ratio = mid_ratio = high_ratio = 0.0
        
        # Overall energy level (RMS)
        energy = np.sqrt(np.mean(audio_segment**2))
        
        # Frequency stability (inverse of spectral centroid variation)
        stability = self._calculate_frequency_stability(fft_vals, freqs)
        
        # Pitch variation using zero-crossing rate
        pitch_variation = self._calculate_pitch_variation(audio_segment)
        
        return FrequencyFeatures(
            low_ratio=low_ratio,
            mid_ratio=mid_ratio,
            high_ratio=high_ratio,
            energy=energy,
            stability=stability,
            pitch_variation=pitch_variation
        )
    
    def _calculate_frequency_stability(self, fft_vals: np.ndarray, freqs: np.ndarray) -> float:
        """
        Calculate frequency stability measure
        
        Args:
            fft_vals: FFT magnitude values
            freqs: Frequency values
            
        Returns:
            Stability measure (higher = more stable)
        """
        if len(fft_vals) == 0 or np.sum(fft_vals) == 0:
            return 0.0
        
        # Calculate spectral centroid
        centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)
        
        # Calculate spectral spread (standard deviation around centroid)
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * fft_vals) / np.sum(fft_vals))
        
        # Convert spread to stability (inverse relationship, normalized)
        stability = 1.0 / (1.0 + spread / 1000.0)  # Normalize by 1kHz
        
        return stability
    
    def _calculate_pitch_variation(self, audio_segment: np.ndarray) -> float:
        """
        Calculate pitch variation using zero-crossing rate
        
        Args:
            audio_segment: Audio data
            
        Returns:
            Pitch variation measure
        """
        if len(audio_segment) <= 1:
            return 0.0
        
        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.signbit(audio_segment)))[0]
        zcr = len(zero_crossings) / len(audio_segment)
        
        # Normalize to 0-1 range
        pitch_variation = min(1.0, zcr * self.sample_rate / 1000.0)
        
        return pitch_variation
    
    def learn_teacher_baseline(self, teacher_segments: list) -> None:
        """
        Learn dynamic thresholds from teacher speech segments
        
        Args:
            teacher_segments: List of audio segments from teacher
        """
        if not teacher_segments:
            logger.warning("No teacher segments provided for baseline learning")
            return
        
        features_list = []
        for segment_audio in teacher_segments:
            if isinstance(segment_audio, str):
                # If it's a file path, load the audio
                segment_audio, _ = librosa.load(segment_audio, sr=self.sample_rate)
            
            features = self.extract_frequency_features(segment_audio)
            features_list.append(features)
        
        # Calculate statistics for dynamic thresholds
        mid_ratios = [f.mid_ratio for f in features_list]
        high_ratios = [f.high_ratio for f in features_list]
        energies = [f.energy for f in features_list]
        
        # Update dynamic thresholds based on teacher baseline
        self.dynamic_thresholds.mid_high = np.percentile(mid_ratios, 75)  # 75th percentile
        self.dynamic_thresholds.mid_low = np.percentile(mid_ratios, 25)   # 25th percentile
        self.dynamic_thresholds.high_mid = np.percentile(high_ratios, 50) # Median
        self.dynamic_thresholds.high_high = np.percentile(high_ratios, 75)
        self.dynamic_thresholds.energy_teacher_median = np.median(energies)
        
        logger.info(f"📊 Learned teacher baseline from {len(teacher_segments)} segments:")
        logger.info(f"   Mid-frequency: {self.dynamic_thresholds.mid_low:.3f} - {self.dynamic_thresholds.mid_high:.3f}")
        logger.info(f"   High-frequency median: {self.dynamic_thresholds.high_mid:.3f}")
        logger.info(f"   Energy baseline: {self.dynamic_thresholds.energy_teacher_median:.4f}")
    
    def classify_speaker_data_driven(self, features: FrequencyFeatures, time_sec: float) -> Tuple[str, float]:
        """
        Classify speaker using data-driven approach without hard time rules
        
        Args:
            features: Extracted frequency features
            time_sec: Time position in recording (for adaptive confidence)
            
        Returns:
            Tuple of (speaker_label, confidence)
        """
        student_score = 0.0
        teacher_score = 0.0
        
        # Frequency-based evidence (as specified in TECHNICAL_ANALYSIS.md)
        if (features.mid_ratio < self.dynamic_thresholds.mid_low and 
            features.high_ratio > self.dynamic_thresholds.high_high):
            student_score += 2.0  # Strong student indicator
        
        if (features.mid_ratio > self.dynamic_thresholds.mid_high and 
            self.dynamic_thresholds.high_low < features.high_ratio < self.dynamic_thresholds.high_mid):
            teacher_score += 1.0  # Teacher indicator
        
        # Energy-based evidence relative to teacher baseline
        if features.energy < self.dynamic_thresholds.energy_teacher_median * 0.8:
            student_score += 1.0  # Lower energy suggests student
        
        if features.energy > self.dynamic_thresholds.energy_teacher_median * 1.1:
            teacher_score += 1.0  # Higher energy suggests teacher
        
        # Stability-based evidence (teachers typically have more stable speech)
        if features.stability > 0.7:
            teacher_score += 0.5
        elif features.stability < 0.4:
            student_score += 0.5
        
        # Adaptive confidence that increases with time (not hardcoded rules)
        early_factor = max(0.0, 1.0 - min(time_sec, 10.0) / 10.0)  # 1→0 by 10 seconds
        teacher_score *= (1.0 - 0.3 * early_factor)  # Reduce teacher confidence early on
        
        # Determine classification and confidence
        if student_score > teacher_score:
            speaker = "STUDENT"
            confidence = min(1.0, student_score / (student_score + teacher_score + 0.1))
        else:
            speaker = "TEACHER"  
            confidence = min(1.0, teacher_score / (student_score + teacher_score + 0.1))
        
        # Apply minimum confidence threshold
        if confidence < self.dynamic_thresholds.confidence_threshold:
            confidence *= 0.7  # Reduce confidence for uncertain classifications
        
        return speaker, confidence
    
    def analyze_segment_with_context(self, audio_segment: np.ndarray, time_sec: float, 
                                   context_segments: Optional[list] = None) -> Dict:
        """
        Analyze audio segment with contextual information
        
        Args:
            audio_segment: Audio data to analyze
            time_sec: Time position in recording
            context_segments: Previous segments for context (optional)
            
        Returns:
            Dictionary with analysis results
        """
        features = self.extract_frequency_features(audio_segment)
        speaker, confidence = self.classify_speaker_data_driven(features, time_sec)
        
        # Contextual adjustment (simple version)
        if context_segments and len(context_segments) > 0:
            recent_speakers = [seg.get('speaker', 'UNKNOWN') for seg in context_segments[-3:]]
            if len(set(recent_speakers)) == 1 and recent_speakers[0] != speaker:
                # If context strongly suggests different speaker, reduce confidence
                confidence *= 0.8
        
        return {
            'speaker': speaker,
            'confidence': confidence,
            'features': features,
            'time_sec': time_sec,
            'reasoning': self._generate_reasoning(features, speaker, confidence, time_sec)
        }
    
    def _generate_reasoning(self, features: FrequencyFeatures, speaker: str, 
                          confidence: float, time_sec: float) -> str:
        """Generate human-readable reasoning for classification"""
        reasons = []
        
        if features.mid_ratio < self.dynamic_thresholds.mid_low:
            reasons.append(f"low mid-freq ({features.mid_ratio:.2f})")
        if features.high_ratio > self.dynamic_thresholds.high_high:
            reasons.append(f"high high-freq ({features.high_ratio:.2f})")
        if features.energy < self.dynamic_thresholds.energy_teacher_median * 0.8:
            reasons.append("low energy")
        if features.stability < 0.4:
            reasons.append("unstable speech")
        
        early_factor = max(0.0, 1.0 - min(time_sec, 10.0) / 10.0)
        if early_factor > 0.5:
            reasons.append("early period")
        
        reasoning = f"{speaker} ({confidence:.2f}): " + ", ".join(reasons) if reasons else f"{speaker} ({confidence:.2f}): balanced features"
        
        return reasoning