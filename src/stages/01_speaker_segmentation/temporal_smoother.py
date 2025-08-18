"""
Temporal context integration with HMM/Viterbi smoothing
Implementation of temporal smoothing to reduce classification flicker
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.signal import medfilt
from .refined_classifier import ClassificationResult

logger = logging.getLogger(__name__)


@dataclass
class HMMState:
    """HMM state representation"""
    name: str  # "TEACHER" or "STUDENT"
    emission_prob: float  # Probability of observing current features
    transition_prob: float  # Probability of transitioning to this state


@dataclass
class SmoothedResult:
    """Result after temporal smoothing"""
    speaker: str
    confidence: float
    original_speaker: str
    original_confidence: float
    smoothing_applied: bool
    reasoning: str
    time_sec: float


class TemporalSmoother:
    """
    Temporal context integration using HMM-like smoothing and post-processing
    """
    
    def __init__(self, stay_probability: float = 0.95, min_segment_duration: float = 2.0):
        """
        Initialize temporal smoother
        
        Args:
            stay_probability: Probability of staying in the same state (high = less flickering)
            min_segment_duration: Minimum duration for speaker segments (seconds)
        """
        self.stay_probability = stay_probability
        self.transition_probability = 1.0 - stay_probability
        self.min_segment_duration = min_segment_duration
        
        logger.info(f"Initialized TemporalSmoother (stay_prob={stay_probability}, min_duration={min_segment_duration}s)")
    
    def median_filter_smooth(self, classifications: List[ClassificationResult], 
                           window_size: int = 5) -> List[SmoothedResult]:
        """
        Apply median filtering to reduce classification flicker
        
        Args:
            classifications: List of classification results
            window_size: Window size for median filter (odd number)
            
        Returns:
            List of smoothed results
        """
        if len(classifications) < window_size:
            # Too few segments for effective smoothing
            return [self._create_smoothed_result(c, c.speaker, c.confidence, False, "insufficient_data") 
                   for c in classifications]
        
        # Convert speaker labels to numeric values for median filtering
        speaker_numeric = []
        for c in classifications:
            speaker_numeric.append(1 if c.speaker == "TEACHER" else 0)
        
        # Apply median filter
        smoothed_numeric = medfilt(speaker_numeric, kernel_size=window_size)
        
        # Convert back to speaker labels and create results
        smoothed_results = []
        for i, (original, smoothed_val) in enumerate(zip(classifications, smoothed_numeric)):
            smoothed_speaker = "TEACHER" if smoothed_val >= 0.5 else "STUDENT"
            smoothing_applied = smoothed_speaker != original.speaker
            
            # Adjust confidence based on smoothing
            if smoothing_applied:
                # Reduce confidence when changing the classification
                new_confidence = original.confidence * 0.8
                reasoning = f"median_filter_changed_{original.speaker}_to_{smoothed_speaker}"
            else:
                # Keep original confidence
                new_confidence = original.confidence
                reasoning = "median_filter_unchanged"
            
            smoothed_results.append(self._create_smoothed_result(
                original, smoothed_speaker, new_confidence, smoothing_applied, reasoning
            ))
        
        changed_count = sum(1 for r in smoothed_results if r.smoothing_applied)
        logger.info(f"📊 Median filter smoothing: {changed_count}/{len(classifications)} segments changed")
        
        return smoothed_results
    
    def hmm_viterbi_smooth(self, classifications: List[ClassificationResult]) -> List[SmoothedResult]:
        """
        Apply HMM-based Viterbi smoothing for temporal consistency
        
        Args:
            classifications: List of classification results
            
        Returns:
            List of smoothed results
        """
        if len(classifications) < 2:
            return [self._create_smoothed_result(c, c.speaker, c.confidence, False, "single_segment") 
                   for c in classifications]
        
        # Simplified 2-state HMM implementation
        states = ["TEACHER", "STUDENT"]
        n_states = len(states)
        n_obs = len(classifications)
        
        # Initialize emission probabilities based on confidence
        emission_probs = np.zeros((n_obs, n_states))
        for i, c in enumerate(classifications):
            if c.speaker == "TEACHER":
                emission_probs[i, 0] = c.confidence  # TEACHER state
                emission_probs[i, 1] = 1.0 - c.confidence  # STUDENT state
            else:
                emission_probs[i, 0] = 1.0 - c.confidence  # TEACHER state
                emission_probs[i, 1] = c.confidence  # STUDENT state
        
        # Transition matrix (high stay probability)
        transition_matrix = np.array([
            [self.stay_probability, self.transition_probability],  # TEACHER to [TEACHER, STUDENT]
            [self.transition_probability, self.stay_probability]   # STUDENT to [TEACHER, STUDENT]
        ])
        
        # Initial state probabilities (uniform)
        initial_probs = np.array([0.5, 0.5])
        
        # Viterbi algorithm
        path_probs = np.zeros((n_obs, n_states))
        path_indices = np.zeros((n_obs, n_states), dtype=int)
        
        # Initialize
        path_probs[0] = initial_probs * emission_probs[0]
        
        # Forward pass
        for t in range(1, n_obs):
            for s in range(n_states):
                # Calculate probabilities for all previous states
                transition_scores = path_probs[t-1] * transition_matrix[:, s]
                # Find best previous state
                best_prev_state = np.argmax(transition_scores)
                path_indices[t, s] = best_prev_state
                path_probs[t, s] = transition_scores[best_prev_state] * emission_probs[t, s]
        
        # Backward pass (find best path)
        best_path = np.zeros(n_obs, dtype=int)
        best_path[-1] = np.argmax(path_probs[-1])
        
        for t in range(n_obs - 2, -1, -1):
            best_path[t] = path_indices[t + 1, best_path[t + 1]]
        
        # Create smoothed results
        smoothed_results = []
        for i, (original, best_state_idx) in enumerate(zip(classifications, best_path)):
            smoothed_speaker = states[best_state_idx]
            smoothing_applied = smoothed_speaker != original.speaker
            
            # Calculate new confidence based on emission probability and path confidence
            emission_conf = emission_probs[i, best_state_idx]
            path_conf = path_probs[i, best_state_idx] / np.sum(path_probs[i])
            new_confidence = min(1.0, (emission_conf + path_conf) / 2)
            
            reasoning = f"viterbi_{'changed' if smoothing_applied else 'kept'}_{original.speaker}_{'to_' + smoothed_speaker if smoothing_applied else ''}"
            
            smoothed_results.append(self._create_smoothed_result(
                original, smoothed_speaker, new_confidence, smoothing_applied, reasoning
            ))
        
        changed_count = sum(1 for r in smoothed_results if r.smoothing_applied)
        logger.info(f"🔄 Viterbi smoothing: {changed_count}/{len(classifications)} segments changed")
        
        return smoothed_results
    
    def merge_short_segments(self, smoothed_results: List[SmoothedResult]) -> List[SmoothedResult]:
        """
        Merge segments that are shorter than minimum duration
        
        Args:
            smoothed_results: List of smoothed results with time information
            
        Returns:
            List with short segments merged
        """
        if len(smoothed_results) < 2:
            return smoothed_results
        
        merged_results = []
        current_segment = smoothed_results[0]
        
        for i in range(1, len(smoothed_results)):
            next_segment = smoothed_results[i]
            
            # Calculate current segment duration (approximate)
            if i < len(smoothed_results) - 1:
                current_duration = smoothed_results[i].time_sec - current_segment.time_sec
            else:
                current_duration = self.min_segment_duration + 1  # Assume last segment is long enough
            
            # Check if current segment is too short
            if (current_duration < self.min_segment_duration and 
                current_segment.speaker != next_segment.speaker):
                
                # Merge with the segment that has higher confidence
                if current_segment.confidence > next_segment.confidence:
                    # Change next segment to match current
                    merged_segment = SmoothedResult(
                        speaker=current_segment.speaker,
                        confidence=(current_segment.confidence + next_segment.confidence) / 2,
                        original_speaker=next_segment.original_speaker,
                        original_confidence=next_segment.original_confidence,
                        smoothing_applied=True,
                        reasoning=f"merged_short_segment_{next_segment.speaker}_to_{current_segment.speaker}",
                        time_sec=next_segment.time_sec
                    )
                    merged_results.append(current_segment)
                    current_segment = merged_segment
                else:
                    # Change current segment to match next
                    merged_segment = SmoothedResult(
                        speaker=next_segment.speaker,
                        confidence=(current_segment.confidence + next_segment.confidence) / 2,
                        original_speaker=current_segment.original_speaker,
                        original_confidence=current_segment.original_confidence,
                        smoothing_applied=True,
                        reasoning=f"merged_short_segment_{current_segment.speaker}_to_{next_segment.speaker}",
                        time_sec=current_segment.time_sec
                    )
                    merged_results.append(merged_segment)
                    current_segment = next_segment
            else:
                # Keep current segment as is
                merged_results.append(current_segment)
                current_segment = next_segment
        
        # Add last segment
        merged_results.append(current_segment)
        
        merged_count = len(smoothed_results) - len(merged_results)
        if merged_count > 0:
            logger.info(f"🔗 Merged {merged_count} short segments")
        
        return merged_results
    
    def apply_temporal_smoothing(self, classifications: List[ClassificationResult], 
                                method: str = "viterbi") -> List[SmoothedResult]:
        """
        Apply temporal smoothing using specified method
        
        Args:
            classifications: List of classification results
            method: Smoothing method ("viterbi", "median", or "combined")
            
        Returns:
            List of smoothed results
        """
        if not classifications:
            return []
        
        logger.info(f"🎯 Applying temporal smoothing method: {method}")
        
        if method == "median":
            smoothed = self.median_filter_smooth(classifications)
        elif method == "viterbi":
            smoothed = self.hmm_viterbi_smooth(classifications)
        elif method == "combined":
            # Apply median filtering first, then Viterbi
            median_smoothed = self.median_filter_smooth(classifications)
            # Convert back to ClassificationResult for Viterbi
            temp_classifications = [
                ClassificationResult(
                    speaker=r.speaker,
                    confidence=r.confidence,
                    reasoning=r.reasoning,
                    template_distance=0.0,  # Not used in smoothing
                    features=classifications[i].features,  # Keep original features
                    time_sec=r.time_sec
                )
                for i, r in enumerate(median_smoothed)
            ]
            smoothed = self.hmm_viterbi_smooth(temp_classifications)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        # Apply segment merging as final step
        final_smoothed = self.merge_short_segments(smoothed)
        
        return final_smoothed
    
    def _create_smoothed_result(self, original: ClassificationResult, new_speaker: str, 
                               new_confidence: float, smoothing_applied: bool, 
                               reasoning: str) -> SmoothedResult:
        """Create a SmoothedResult from original classification"""
        return SmoothedResult(
            speaker=new_speaker,
            confidence=new_confidence,
            original_speaker=original.speaker,
            original_confidence=original.confidence,
            smoothing_applied=smoothing_applied,
            reasoning=reasoning,
            time_sec=original.time_sec
        )
    
    def get_smoothing_summary(self, original: List[ClassificationResult], 
                            smoothed: List[SmoothedResult]) -> Dict:
        """Get summary of smoothing effects"""
        if not original or not smoothed:
            return {"error": "Empty input"}
        
        changes = sum(1 for s in smoothed if s.smoothing_applied)
        
        # Count speaker transitions before and after
        def count_transitions(results):
            transitions = 0
            for i in range(1, len(results)):
                if hasattr(results[i], 'speaker'):
                    curr_speaker = results[i].speaker
                    prev_speaker = results[i-1].speaker
                else:
                    curr_speaker = results[i]
                    prev_speaker = results[i-1]
                if curr_speaker != prev_speaker:
                    transitions += 1
            return transitions
        
        original_transitions = count_transitions([r.speaker for r in original])
        smoothed_transitions = count_transitions([r.speaker for r in smoothed])
        
        return {
            "total_segments": len(original),
            "segments_changed": changes,
            "change_percentage": (changes / len(original)) * 100,
            "transitions_before": original_transitions,
            "transitions_after": smoothed_transitions,
            "transitions_reduced": original_transitions - smoothed_transitions
        }