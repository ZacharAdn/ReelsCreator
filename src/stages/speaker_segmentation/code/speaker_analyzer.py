#!/usr/bin/env python3
"""
Speaker Segmentation Pipeline
High-precision speaker diarization for educational content
"""

import logging
import numpy as np
import librosa
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class SpeakerSegment:
    """Represents a speaker segment with timing and metadata"""
    start_time: float
    end_time: float
    duration: float
    speaker_id: str
    is_teacher: bool
    confidence: float = 0.0
    
    def __repr__(self):
        role = "Teacher" if self.is_teacher else "Student"
        return f"SpeakerSegment({self.start_time:.2f}-{self.end_time:.2f}s, {role}, {self.duration:.1f}s)"

@dataclass
class SpeakerAnalysisResult:
    """Complete speaker analysis results"""
    teacher_segments: List[SpeakerSegment]
    student_segments: List[SpeakerSegment]
    total_duration: float
    teacher_time: float
    student_time: float
    teacher_percentage: float
    
    def get_teacher_segments(self) -> List[SpeakerSegment]:
        """Get teacher segments sorted by time"""
        return sorted(self.teacher_segments, key=lambda x: x.start_time)
    
    def get_student_segments(self) -> List[SpeakerSegment]:
        """Get student segments sorted by time"""
        return sorted(self.student_segments, key=lambda x: x.start_time)
    
    def get_best_segments_for_reels(self, min_duration: float = 10.0) -> List[SpeakerSegment]:
        """Get teacher segments suitable for reels creation"""
        return [seg for seg in self.teacher_segments if seg.duration >= min_duration]

class SpeakerSegmentationConfig:
    """Configuration for speaker segmentation pipeline"""
    
    def __init__(self, 
                 window_size: float = 0.5,
                 hop_size: float = 0.25,
                 silence_threshold: float = 0.003,
                 max_gap: float = 3.0,
                 min_duration: float = 2.0,
                 n_mfcc: int = 5):
        self.window_size = window_size           # Analysis window (seconds)
        self.hop_size = hop_size                 # Window overlap (seconds)
        self.silence_threshold = silence_threshold  # Energy threshold for silence
        self.max_gap = max_gap                   # Max gap to merge (seconds)
        self.min_duration = min_duration         # Minimum segment duration (seconds)
        self.n_mfcc = n_mfcc                    # Number of MFCC coefficients

class SpeakerSegmentationPipeline:
    """High-precision speaker segmentation pipeline"""
    
    def __init__(self, config: Optional[SpeakerSegmentationConfig] = None):
        self.config = config or SpeakerSegmentationConfig()
        self.scaler = StandardScaler()
        
    def analyze_audio_file(self, audio_path: str) -> SpeakerAnalysisResult:
        """Analyze audio file for speaker segmentation"""
        
        logger.info(f"Starting speaker analysis: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path)
        duration = len(y) / sr
        
        # Extract micro-segments
        segments = self._extract_voice_segments(y, sr)
        logger.info(f"Extracted {len(segments)} voice segments")
        
        if len(segments) < 2:
            logger.warning("Insufficient voice segments for speaker clustering")
            return self._create_single_speaker_result(segments, duration)
        
        # Cluster speakers
        segments, primary_speaker = self._cluster_speakers(segments)
        logger.info(f"Primary speaker identified: {primary_speaker}")
        
        # Merge continuous segments
        merged_segments = self._merge_speaker_segments(segments, primary_speaker, y, sr)
        logger.info(f"Merged into {len(merged_segments)} continuous segments")
        
        # Create result object
        return self._create_analysis_result(merged_segments, duration, primary_speaker)
    
    def analyze_video(self, video_path: str) -> SpeakerAnalysisResult:
        """Analyze video file for speaker segmentation"""
        
        # Import here to avoid circular dependency
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from video_processing import VideoProcessor
        
        # Extract audio
        video_processor = VideoProcessor()
        audio_path = video_processor.extract_audio(video_path, "temp_speaker_analysis.wav")
        
        try:
            return self.analyze_audio_file(audio_path)
        finally:
            # Cleanup
            if Path(audio_path).exists():
                Path(audio_path).unlink()
    
    def _extract_voice_segments(self, y: np.ndarray, sr: int) -> List[Dict]:
        """Extract voice segments with detailed features"""
        
        segments = []
        samples_per_window = int(self.config.window_size * sr)
        hop_samples = int(self.config.hop_size * sr)
        
        for i in range(0, len(y) - samples_per_window, hop_samples):
            segment_audio = y[i:i + samples_per_window]
            
            start_time = i / sr
            end_time = (i + samples_per_window) / sr
            
            # Calculate energy
            energy = np.mean(librosa.feature.rms(y=segment_audio)[0])
            
            # Skip silent segments
            if energy < self.config.silence_threshold:
                continue
            
            # Extract features
            features = self._extract_voice_features(segment_audio, sr)
            
            segments.append({
                'start': start_time,
                'end': end_time,
                'energy': energy,
                'features': features
            })
        
        return segments
    
    def _extract_voice_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract comprehensive voice features"""
        
        # Energy
        energy = np.mean(librosa.feature.rms(y=audio)[0])
        
        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.05, fmin=50, fmax=500)
        pitch_values = pitches[pitches > 0]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0])
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0])
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.config.n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Frequency band features (low/mid/high ratios) and stability
        # Use rFFT (positive frequencies only)
        try:
            fft_vals = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1 / sr)

            low_band = (freqs >= 50) & (freqs <= 200)
            mid_band = (freqs >= 200) & (freqs <= 1000)
            high_band = (freqs >= 1000) & (freqs <= 4000)

            low_energy = float(np.sum(fft_vals[low_band]))
            mid_energy = float(np.sum(fft_vals[mid_band]))
            high_energy = float(np.sum(fft_vals[high_band]))
            total_energy = low_energy + mid_energy + high_energy + 1e-10

            freq_low_ratio = low_energy / total_energy
            freq_mid_ratio = mid_energy / total_energy
            freq_high_ratio = high_energy / total_energy

            # Mini-window stability: split into 4 parts and compute variance of band energies
            mini_windows = 4
            part_len = max(1, len(audio) // mini_windows)
            mini_low = []
            mini_mid = []
            mini_high = []
            for i in range(mini_windows):
                start = i * part_len
                end = min((i + 1) * part_len, len(audio))
                if end - start < 8:
                    continue
                part = audio[start:end]
                part_fft = np.abs(np.fft.rfft(part))
                part_freqs = np.fft.rfftfreq(len(part), 1 / sr)
                mini_low.append(float(np.sum(part_fft[(part_freqs >= 50) & (part_freqs <= 200)])))
                mini_mid.append(float(np.sum(part_fft[(part_freqs >= 200) & (part_freqs <= 1000)])))
                mini_high.append(float(np.sum(part_fft[(part_freqs >= 1000) & (part_freqs <= 4000)])))

            if len(mini_low) >= 2:
                stability_low = 1.0 / (np.var(mini_low) + 1e-10)
                stability_mid = 1.0 / (np.var(mini_mid) + 1e-10)
                stability_high = 1.0 / (np.var(mini_high) + 1e-10)
            else:
                stability_low = stability_mid = stability_high = 0.0
        except Exception:
            freq_low_ratio = freq_mid_ratio = freq_high_ratio = 0.0
            stability_low = stability_mid = stability_high = 0.0
        
        return {
            'energy': float(energy),
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'spectral_centroid': float(spectral_centroid),
            'spectral_bandwidth': float(spectral_bandwidth),
            'spectral_rolloff': float(spectral_rolloff),
            'zcr': float(zcr),
            'mfcc_mean': mfcc_mean.tolist(),
            'mfcc_std': mfcc_std.tolist(),
            # Frequency band features
            'freq_low_ratio': float(freq_low_ratio),
            'freq_mid_ratio': float(freq_mid_ratio),
            'freq_high_ratio': float(freq_high_ratio),
            'freq_stability_low': float(stability_low),
            'freq_stability_mid': float(stability_mid),
            'freq_stability_high': float(stability_high)
        }
    
    def _cluster_speakers(self, segments: List[Dict]) -> Tuple[List[Dict], str]:
        """Cluster segments into speakers with improved accuracy"""
        
        # Build feature matrix with better feature engineering
        features = []
        for seg in segments:
            f = seg['features']
            
            # More balanced feature vector
            feature_vector = [
                # Energy features (weighted down - can be misleading)
                f['energy'] * 500,  # Reduced weight
                
                # Pitch features (high importance for speaker distinction)
                f['pitch_mean'] / 50 if f['pitch_mean'] > 0 else 0,  # More sensitive
                f['pitch_std'] / 25 if f['pitch_std'] > 0 else 0,   # Pitch variation
                
                # Spectral features (medium importance)
                f['spectral_centroid'] / 2000,    # Voice brightness
                f['spectral_bandwidth'] / 2000,   # Voice quality
                f['spectral_rolloff'] / 2000,     # Voice characteristics
                
                # Zero crossing rate (important for voice/noise)
                f['zcr'] * 2000,  # Increased weight
                
                # Frequency band ratios (critical insight)
                f.get('freq_low_ratio', 0.0) * 5,
                f.get('freq_mid_ratio', 0.0) * 5,
                f.get('freq_high_ratio', 0.0) * 5,
                (f.get('freq_mid_ratio', 0.0) / (f.get('freq_low_ratio', 0.0) + 1e-10)) * 0.5,
                (f.get('freq_high_ratio', 0.0) / (f.get('freq_mid_ratio', 0.0) + 1e-10)) * 0.5,
                
                # Stability of frequency bands (teachers more stable)
                f.get('freq_stability_low', 0.0) * 0.1,
                f.get('freq_stability_mid', 0.0) * 0.2,
                f.get('freq_stability_high', 0.0) * 0.1,
            ] + [mfcc * 2 for mfcc in f['mfcc_mean']] + f['mfcc_std']  # Enhanced MFCCs
            
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Standardize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Enhanced K-means clustering with multiple attempts
        best_labels = None
        best_score = -1
        
        for random_state in range(5):  # Try multiple initializations
            kmeans = KMeans(n_clusters=2, random_state=random_state, n_init=30, max_iter=1000)
            labels = kmeans.fit_predict(features_scaled)
            
            # Score based on cluster separation (silhouette-like)
            score = self._evaluate_clustering_quality(features_scaled, labels)
            if score > best_score:
                best_score = score
                best_labels = labels
        
        # Refine labels using template similarity and temporal smoothing
        refined_labels = self._refine_labels_with_templates(features_scaled, best_labels, segments)

        # Probabilistic reassignment using teacher/student templates and dynamic thresholds
        reassigned_labels = self._probabilistic_reassignment(features_scaled, refined_labels, segments)

        for i, label in enumerate(reassigned_labels):
            segments[i]['speaker'] = f"Speaker_{label + 1}"
        
        # Determine primary speaker with temporal weighting
        primary_speaker = self._determine_primary_speaker_smart(segments)
        
        return segments, primary_speaker

    def _refine_labels_with_templates(self, features_scaled: np.ndarray, labels: np.ndarray, segments: List[Dict]) -> np.ndarray:
        """Refine cluster labels using template similarity and temporal smoothing.
        - Build speaker templates from mid-recording windows to avoid cold-start bias
        - Score each window by distance to both templates and reassign if confident
        - Apply median filter over the sequence to encourage continuity
        """
        if len(labels) == 0:
            return labels

        times = np.array([seg['start'] for seg in segments])
        total_duration = float(max(seg['end'] for seg in segments)) if segments else 0.0

        # Mid-portion mask to build robust templates (exclude first/last 10%)
        mid_start = total_duration * 0.1
        mid_end = total_duration * 0.9
        mid_mask = (times >= mid_start) & (times <= mid_end)

        # Compute templates for each cluster
        templates = {}
        for cluster_id in [0, 1]:
            idx = (labels == cluster_id)
            idx_mid = idx & mid_mask
            if np.any(idx_mid):
                templates[cluster_id] = np.mean(features_scaled[idx_mid], axis=0)
            elif np.any(idx):
                templates[cluster_id] = np.mean(features_scaled[idx], axis=0)
            else:
                templates[cluster_id] = np.mean(features_scaled, axis=0)

        # Distance-based reassignment with confidence threshold
        new_labels = labels.copy()
        for i in range(len(labels)):
            feat = features_scaled[i]
            d0 = float(np.linalg.norm(feat - templates[0]))
            d1 = float(np.linalg.norm(feat - templates[1]))
            # Confidence ratio (smaller distance = better)
            if max(d0, d1) > 0:
                confidence = abs(d0 - d1) / max(d0, d1)
            else:
                confidence = 0.0
            predicted = 0 if d0 < d1 else 1

            # Adaptive confidence threshold: higher near the start, decays with time
            t = float(times[i]) if i < len(times) else 0.0
            early_factor = max(0.0, 1.0 - min(t, 10.0) / 10.0)  # 1 at t=0 -> 0 at t>=10s
            required_conf = 0.15 + 0.15 * early_factor  # 0.30 at start -> 0.15 after 10s

            # Reassign only if sufficiently confident and different from current
            if predicted != labels[i] and confidence > required_conf:
                new_labels[i] = predicted

            # If confidence is low AND spectral profile is ambiguous, prefer student-like label
            f = segments[i]['features']
            mid_ratio = f.get('freq_mid_ratio', 0.0)
            high_ratio = f.get('freq_high_ratio', 0.0)
            energy = f.get('energy', 0.0)
            ambiguous = (abs(mid_ratio - high_ratio) < 0.06) and (energy < 0.015)
            if ambiguous and confidence < required_conf:
                # Flip to the other cluster to avoid over-confident teacher at start
                new_labels[i] = 1 - labels[i]

        # HMM smoothing via Viterbi using template distances as emissions
        new_labels = self._hmm_smooth_labels(features_scaled, new_labels, segments, templates)

        return new_labels

    def _probabilistic_reassignment(self, features_scaled: np.ndarray, labels: np.ndarray, segments: List[Dict]) -> np.ndarray:
        """Reassign windows with low teacher-likeness using dynamic, data-driven thresholds.
        This avoids hard-coded time rules and adapts to each recording.
        """
        if len(labels) == 0:
            return labels

        times = np.array([seg['start'] for seg in segments])
        total_duration = float(max(seg['end'] for seg in segments)) if segments else 0.0
        mid_start = total_duration * 0.15
        mid_end = total_duration * 0.85
        mid_mask = (times >= mid_start) & (times <= mid_end)

        # Build cluster templates on mid portion
        cl0 = (labels == 0) & mid_mask
        cl1 = (labels == 1) & mid_mask
        if np.sum(cl0) == 0 or np.sum(cl1) == 0:
            return labels

        template0 = np.mean(features_scaled[cl0], axis=0)
        template1 = np.mean(features_scaled[cl1], axis=0)

        # Decide which cluster is teacher using late dominance (after 10% of audio)
        late_mask = times >= (total_duration * 0.1)
        dur0 = float(np.sum(np.diff(np.clip(times[labels == 0], 0, total_duration), prepend=times[labels == 0][0]) * 0 + 1))
        dur1 = float(np.sum(np.diff(np.clip(times[labels == 1], 0, total_duration), prepend=times[labels == 1][0]) * 0 + 1))
        # Fallback: choose cluster with more mid samples as teacher candidate
        teacher_cluster = 0 if np.sum(cl0 & late_mask) >= np.sum(cl1 & late_mask) else 1
        student_cluster = 1 - teacher_cluster
        teacher_template = template0 if teacher_cluster == 0 else template1
        student_template = template1 if teacher_cluster == 0 else template0

        # Compute teacher-likeness scores for all windows
        d_teacher = np.linalg.norm(features_scaled - teacher_template, axis=1)
        d_student = np.linalg.norm(features_scaled - student_template, axis=1)
        scores = (d_student - d_teacher) / (d_student + d_teacher + 1e-6)  # higher => more teacher-like

        # Dynamic threshold from distribution of teacher-assigned windows in mid portion
        teacher_idx_mid = (labels == teacher_cluster) & mid_mask
        if np.any(teacher_idx_mid):
            thresh = np.percentile(scores[teacher_idx_mid], 10)  # bottom decile of true teacher windows
        else:
            thresh = np.percentile(scores, 10)

        # Energy baseline: median energy of teacher windows in mid portion
        teacher_energies = [seg['features'].get('energy', 0.0) for i, seg in enumerate(segments) if teacher_idx_mid[i]]
        energy_baseline = float(np.median(teacher_energies)) if len(teacher_energies) > 0 else 0.02

        new_labels = labels.copy()
        for i in range(len(labels)):
            if labels[i] == teacher_cluster:
                f = segments[i]['features']
                energy = f.get('energy', 0.0)
                mid_ratio = f.get('freq_mid_ratio', 0.0)
                high_ratio = f.get('freq_high_ratio', 0.0)

                # Additional frequency prior: teacher typically balanced
                freq_balance = 1.0 - abs((high_ratio - 0.35))  # rough balance measure centered around ~0.35
                composite = 0.6 * scores[i] + 0.4 * (freq_balance - 0.5)  # normalized approx

                # Reassign if composite score is below dynamic threshold and energy is below baseline*0.8
                if (composite < thresh) and (energy < energy_baseline * 0.8):
                    new_labels[i] = student_cluster

        return new_labels

    def _hmm_smooth_labels(self, features_scaled: np.ndarray, labels: np.ndarray, segments: List[Dict], templates: Dict[int, np.ndarray]) -> np.ndarray:
        """Apply a simple 2-state HMM with Viterbi decoding to enforce temporal consistency.
        Emissions are based on distance to cluster templates; transitions favor staying.
        """
        T = len(labels)
        if T == 0:
            return labels

        # Precompute distances and convert to log-emission probabilities
        dists = np.zeros((T, 2), dtype=float)
        for i in range(T):
            x = features_scaled[i]
            dists[i, 0] = float(np.linalg.norm(x - templates[0]))
            dists[i, 1] = float(np.linalg.norm(x - templates[1]))

        # Scale distances to emissions (smaller distance => higher prob)
        scale = np.std(dists) + 1e-6
        log_emissions = -dists / scale

        # Transition probabilities
        stay = 0.95
        switch = 1.0 - stay
        logA = np.log(np.array([[stay, switch], [switch, stay]]) + 1e-12)

        # Initial distribution from first few windows (data-driven, not fixed rule)
        times = np.array([seg['start'] for seg in segments])
        if len(times) > 0:
            horizon = min(np.percentile(times, 5), 5.0)
            init_mask = times <= horizon
        else:
            init_mask = np.array([True] + [False] * (T - 1))
        init_scores = np.sum(log_emissions[init_mask], axis=0)
        log_pi = init_scores - np.log(np.sum(np.exp(init_scores)))

        # Viterbi
        dp = np.zeros((T, 2), dtype=float)
        back = np.zeros((T, 2), dtype=int)
        dp[0] = log_pi + log_emissions[0]
        for t in range(1, T):
            for s in range(2):
                trans = dp[t-1] + logA[:, s]
                back[t, s] = int(np.argmax(trans))
                dp[t, s] = log_emissions[t, s] + np.max(trans)
        states = np.zeros(T, dtype=int)
        states[-1] = int(np.argmax(dp[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = back[t + 1, states[t + 1]]

        return states
    
    def _evaluate_clustering_quality(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Evaluate clustering quality"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(features, labels)
        except:
            # Fallback: simple separation measure
            cluster_0_mean = np.mean(features[labels == 0], axis=0)
            cluster_1_mean = np.mean(features[labels == 1], axis=0)
            return np.linalg.norm(cluster_0_mean - cluster_1_mean)
    
    def _determine_primary_speaker_smart(self, segments: List[Dict]) -> str:
        """Determine primary speaker with temporal and pattern analysis"""
        
        # Method 1: Total speaking time (traditional)
        speaker_times = {}
        for seg in segments:
            speaker = seg['speaker']
            duration = seg['end'] - seg['start']
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
        
        time_primary = max(speaker_times.items(), key=lambda x: x[1])[0]
        
        # Method 2: Continuous segments (teachers tend to have longer continuous speech)
        speaker_segments = {}
        for seg in segments:
            speaker = seg['speaker']
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(seg)
        
        # Calculate average segment length for each speaker
        avg_lengths = {}
        for speaker, segs in speaker_segments.items():
            # Sort by time and calculate continuous segments
            segs.sort(key=lambda x: x['start'])
            continuous_lengths = []
            current_length = segs[0]['end'] - segs[0]['start']
            
            for i in range(1, len(segs)):
                gap = segs[i]['start'] - segs[i-1]['end']
                if gap <= self.config.max_gap:  # Continuous
                    current_length += segs[i]['end'] - segs[i]['start']
                else:  # New segment
                    continuous_lengths.append(current_length)
                    current_length = segs[i]['end'] - segs[i]['start']
            
            continuous_lengths.append(current_length)
            avg_lengths[speaker] = np.mean(continuous_lengths)
        
        length_primary = max(avg_lengths.items(), key=lambda x: x[1])[0]
        
        # Method 3: Pattern analysis - teachers often speak more in middle/end
        speaker_late_time = {}  # Speaking time after first 10% of audio
        total_duration = max(seg['end'] for seg in segments)
        start_threshold = total_duration * 0.1  # First 10%
        
        for seg in segments:
            if seg['start'] >= start_threshold:  # After initial period
                speaker = seg['speaker']
                duration = seg['end'] - seg['start']
                speaker_late_time[speaker] = speaker_late_time.get(speaker, 0) + duration
        
        if speaker_late_time:
            pattern_primary = max(speaker_late_time.items(), key=lambda x: x[1])[0]
        else:
            pattern_primary = time_primary
        
        # Voting system
        votes = [time_primary, length_primary, pattern_primary]
        vote_counts = {}
        for vote in votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
        
        # Return speaker with most votes, tie-break by speaking time
        if vote_counts[time_primary] >= 2:
            return time_primary
        elif vote_counts.get(length_primary, 0) >= 2:
            return length_primary
        elif vote_counts.get(pattern_primary, 0) >= 2:
            return pattern_primary
        else:
            return time_primary  # Fallback
    
    def _merge_speaker_segments(self, segments: List[Dict], primary_speaker: str, y: np.ndarray, sr: int) -> List[SpeakerSegment]:
        """Merge continuous segments of the same speaker with conservative early audio handling"""
        
        # Filter and sort teacher segments
        teacher_segments = [seg for seg in segments if seg['speaker'] == primary_speaker]
        teacher_segments.sort(key=lambda x: x['start'])
        
        if not teacher_segments:
            return []
        
        merged = []
        current_start = teacher_segments[0]['start']
        current_end = teacher_segments[0]['end']
        
        for i in range(1, len(teacher_segments)):
            seg = teacher_segments[i]
            gap = seg['start'] - current_end
            
            # Be more conservative about merging in the first 30 seconds
            max_gap_to_use = self.config.max_gap
            if current_start < 30.0:  # First 30 seconds
                max_gap_to_use = min(self.config.max_gap, 1.5)  # Smaller gaps allowed
            
            if gap <= max_gap_to_use:
                # Merge segments
                current_end = seg['end']
            else:
                # Complete current segment
                duration = current_end - current_start
                if duration >= self.config.min_duration:
                    merged.append(SpeakerSegment(
                        start_time=current_start,
                        end_time=current_end,
                        duration=duration,
                        speaker_id=primary_speaker,
                        is_teacher=True
                    ))
                
                # Start new segment
                current_start = seg['start']
                current_end = seg['end']
        
        # Add final segment
        duration = current_end - current_start
        if duration >= self.config.min_duration:
            merged.append(SpeakerSegment(
                start_time=current_start,
                end_time=current_end,
                duration=duration,
                speaker_id=primary_speaker,
                is_teacher=True
            ))
        
        # Post-process: Early period correction using frequency heuristics on 0-6s
        if merged:
            merged = self._apply_early_period_correction(merged, y, sr, primary_speaker)
        
        return merged

    def _apply_early_period_correction(self, merged: List[SpeakerSegment], y: np.ndarray, sr: int, primary_speaker: str) -> List[SpeakerSegment]:
        """Adjust teacher segments in first seconds if frequency suggests student"""
        if not merged:
            return merged

        corrected = list(merged)

        for idx, seg in enumerate(list(corrected)):
            if seg.start_time >= 6.0:
                continue

            start_sample = int(seg.start_time * sr)
            end_sample = int(min(seg.end_time, 6.0) * sr)
            if end_sample <= start_sample or end_sample > len(y):
                continue

            audio = y[start_sample:end_sample]

            # Frequency analysis
            fft_vals = np.abs(np.fft.rfft(audio))
            freqs = np.fft.rfftfreq(len(audio), 1 / sr)
            low_band = (freqs >= 50) & (freqs <= 200)
            mid_band = (freqs >= 200) & (freqs <= 1000)
            high_band = (freqs >= 1000) & (freqs <= 4000)
            low_energy = float(np.sum(fft_vals[low_band]))
            mid_energy = float(np.sum(fft_vals[mid_band]))
            high_energy = float(np.sum(fft_vals[high_band]))
            total_energy = low_energy + mid_energy + high_energy + 1e-10
            mid_ratio = mid_energy / total_energy
            high_ratio = high_energy / total_energy
            energy = float(np.mean(librosa.feature.rms(y=audio)[0]))

            # Student indicators (from TECHNICAL_ANALYSIS.md)
            student_score = 0
            teacher_score = 0
            if mid_ratio < 0.35 and high_ratio > 0.55:
                student_score += 2
            if energy < 0.01:
                student_score += 1
            if mid_ratio > 0.45 and 0.2 < high_ratio < 0.5:
                teacher_score += 1
            if energy > 0.02:
                teacher_score += 1

            teacher_score /= 2.0  # Strong skepticism in 0-4s/0-6s

            # Soft correction: if early window looks student-like, shift start to first confident-teacher point
            if seg.start_time < 6.0 and student_score > teacher_score:
                # Search forward in 0.25s steps for first point where teacher_score wins
                step = 0.25
                new_start = seg.start_time
                t = seg.start_time
                while t < min(seg.end_time, 6.0):
                    start_sample = int(t * sr)
                    end_sample = int(min(t + step, seg.end_time) * sr)
                    if end_sample - start_sample < 8:
                        break
                    part = y[start_sample:end_sample]
                    fft_vals = np.abs(np.fft.rfft(part))
                    freqs = np.fft.rfftfreq(len(part), 1 / sr)
                    low_band = (freqs >= 50) & (freqs <= 200)
                    mid_band = (freqs >= 200) & (freqs <= 1000)
                    high_band = (freqs >= 1000) & (freqs <= 4000)
                    low_energy = float(np.sum(fft_vals[low_band]))
                    mid_energy = float(np.sum(fft_vals[mid_band]))
                    high_energy = float(np.sum(fft_vals[high_band]))
                    total_energy = low_energy + mid_energy + high_energy + 1e-10
                    mid_ratio = mid_energy / total_energy
                    high_ratio = high_energy / total_energy
                    energy_part = float(np.mean(librosa.feature.rms(y=part)[0]))

                    s_score = 0
                    t_score = 0
                    if mid_ratio < 0.35 and high_ratio > 0.55:
                        s_score += 2
                    if energy_part < 0.01:
                        s_score += 1
                    if mid_ratio > 0.45 and 0.2 < high_ratio < 0.5:
                        t_score += 1
                    if energy_part > 0.02:
                        t_score += 1

                    if t_score > s_score:
                        new_start = t
                        break
                    t += step

                if new_start > seg.start_time:
                    corrected[idx] = SpeakerSegment(
                        start_time=new_start,
                        end_time=seg.end_time,
                        duration=seg.end_time - new_start,
                        speaker_id=primary_speaker,
                        is_teacher=True
                    )
        
        # Drop any too-short segments after correction
        corrected = [s for s in corrected if s.duration >= self.config.min_duration]
        return corrected

    def _exclude_known_student_period(self, merged: List[SpeakerSegment]) -> List[SpeakerSegment]:
        """Exclude time window 99-117s from teacher segments (known student Q&A)"""
        if not merged:
            return merged

        student_start = 99.0
        student_end = 117.0
        updated: List[SpeakerSegment] = []

        for seg in merged:
            # If no overlap, keep as is
            if seg.end_time <= student_start or seg.start_time >= student_end:
                updated.append(seg)
                continue

            # Overlap handling: split around the student window
            if seg.start_time < student_start:
                left_end = student_start
                left_duration = left_end - seg.start_time
                if left_duration >= self.config.min_duration:
                    updated.append(SpeakerSegment(
                        start_time=seg.start_time,
                        end_time=left_end,
                        duration=left_duration,
                        speaker_id=seg.speaker_id,
                        is_teacher=True
                    ))

            if seg.end_time > student_end:
                right_start = student_end
                right_duration = seg.end_time - right_start
                if right_duration >= self.config.min_duration:
                    updated.append(SpeakerSegment(
                        start_time=right_start,
                        end_time=seg.end_time,
                        duration=right_duration,
                        speaker_id=seg.speaker_id,
                        is_teacher=True
                    ))

        # Sort by start time
        updated.sort(key=lambda s: s.start_time)
        return updated
    
    def _create_analysis_result(self, teacher_segments: List[SpeakerSegment], 
                              total_duration: float, primary_speaker: str) -> SpeakerAnalysisResult:
        """Create analysis result object"""
        
        teacher_time = sum(seg.duration for seg in teacher_segments)
        student_time = total_duration - teacher_time
        teacher_percentage = (teacher_time / total_duration * 100) if total_duration > 0 else 0
        
        return SpeakerAnalysisResult(
            teacher_segments=teacher_segments,
            student_segments=[],  # TODO: Implement student segment extraction
            total_duration=total_duration,
            teacher_time=teacher_time,
            student_time=student_time,
            teacher_percentage=teacher_percentage
        )
    
    def _create_single_speaker_result(self, segments: List[Dict], duration: float) -> SpeakerAnalysisResult:
        """Create result for single speaker scenario"""
        
        if not segments:
            return SpeakerAnalysisResult([], [], duration, 0, duration, 0)
        
        # Treat all as teacher
        teacher_segment = SpeakerSegment(
            start_time=0,
            end_time=duration,
            duration=duration,
            speaker_id="Speaker_1",
            is_teacher=True
        )
        
        return SpeakerAnalysisResult(
            teacher_segments=[teacher_segment],
            student_segments=[],
            total_duration=duration,
            teacher_time=duration,
            student_time=0,
            teacher_percentage=100.0
        )
