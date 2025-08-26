#!/usr/bin/env python3
"""
Voice Pattern Analysis Tool
Visualizes differences between teacher and student speech patterns
"""

import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def analyze_voice_patterns(video_path: str):
    """Comprehensive voice pattern analysis with visualizations"""
    
    print(f"🔍 Analyzing voice patterns in: {Path(video_path).name}")
    print("=" * 60)
    
    # Extract audio
    from src.video_processing import VideoProcessor
    video_processor = VideoProcessor()
    audio_path = video_processor.extract_audio(video_path, "temp_analysis.wav")
    
    try:
        # Load and analyze audio
        y, sr = librosa.load(audio_path)
        duration = len(y) / sr
        
        print(f"📊 Audio info: {duration:.1f} seconds, {sr} Hz sample rate")
        
        # Extract detailed segments for analysis
        segments_data = extract_detailed_segments(y, sr)
        
        if len(segments_data) < 10:
            print("❌ Not enough voice segments for analysis")
            return
        
        # Cluster into speakers
        clustered_data = cluster_and_analyze(segments_data)
        
        # Create comprehensive visualizations
        create_voice_analysis_plots(clustered_data, duration)
        
        # Analyze specific time periods
        analyze_time_periods(clustered_data, duration)
        
        # Focus on the problematic first 10 seconds
        analyze_first_seconds(clustered_data)
        
        # Analyze both problematic periods with frequency insights
        analyze_problematic_periods(clustered_data, y, sr)
        
    finally:
        if Path(audio_path).exists():
            Path(audio_path).unlink()

def extract_detailed_segments(y, sr):
    """Extract detailed voice segments with comprehensive features"""
    
    window_size = 0.5  # 0.5 second windows
    hop_size = 0.25    # 0.25 second overlap
    
    samples_per_window = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    
    segments = []
    
    for i in range(0, len(y) - samples_per_window, hop_samples):
        segment_audio = y[i:i + samples_per_window]
        
        start_time = i / sr
        end_time = (i + samples_per_window) / sr
        
        # Calculate energy
        energy = np.mean(librosa.feature.rms(y=segment_audio)[0])
        
        # Skip very quiet segments
        if energy < 0.003:
            continue
        
        # Extract comprehensive features
        features = extract_comprehensive_features(segment_audio, sr)
        features['start_time'] = start_time
        features['end_time'] = end_time
        features['energy'] = energy
        
        segments.append(features)
    
    print(f"📈 Extracted {len(segments)} voice segments")
    return segments

def extract_comprehensive_features(audio, sr):
    """Extract comprehensive voice features for analysis"""
    
    # Energy
    energy = np.mean(librosa.feature.rms(y=audio)[0])
    energy_max = np.max(librosa.feature.rms(y=audio)[0])
    energy_std = np.std(librosa.feature.rms(y=audio)[0])
    
    # Pitch analysis
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, threshold=0.05, fmin=50, fmax=500)
    pitch_values = pitches[pitches > 0]
    
    pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
    pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
    pitch_max = np.max(pitch_values) if len(pitch_values) > 0 else 0
    pitch_min = np.min(pitch_values) if len(pitch_values) > 0 else 0
    
    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)[0])
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0])
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)[0])
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio)[0])
    
    # Zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio)[0])
    zcr_std = np.std(librosa.feature.zero_crossing_rate(audio)[0])
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # Chroma features (harmonic content)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Tempo and rhythm
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
    
    return {
        'energy_mean': float(energy),
        'energy_max': float(energy_max),
        'energy_std': float(energy_std),
        'pitch_mean': float(pitch_mean),
        'pitch_std': float(pitch_std),
        'pitch_max': float(pitch_max),
        'pitch_min': float(pitch_min),
        'pitch_range': float(pitch_max - pitch_min) if pitch_max > 0 else 0,
        'spectral_centroid': float(spectral_centroid),
        'spectral_bandwidth': float(spectral_bandwidth),
        'spectral_rolloff': float(spectral_rolloff),
        'spectral_flatness': float(spectral_flatness),
        'zcr_mean': float(zcr),
        'zcr_std': float(zcr_std),
        'mfcc_mean': mfcc_mean.tolist(),
        'mfcc_std': mfcc_std.tolist(),
        'chroma_mean': chroma_mean.tolist(),
        'tempo': float(tempo)
    }

def cluster_and_analyze(segments_data):
    """Cluster segments and add speaker labels"""
    
    # Build feature matrix
    features = []
    for seg in segments_data:
        feature_vector = [
            seg['energy_mean'] * 1000,
            seg['pitch_mean'] / 100 if seg['pitch_mean'] > 0 else 0,
            seg['pitch_std'] / 50 if seg['pitch_std'] > 0 else 0,
            seg['pitch_range'] / 100 if seg['pitch_range'] > 0 else 0,
            seg['spectral_centroid'] / 1000,
            seg['spectral_bandwidth'] / 1000,
            seg['spectral_rolloff'] / 1000,
            seg['spectral_flatness'] * 100,
            seg['zcr_mean'] * 1000,
            seg['zcr_std'] * 1000,
            seg['tempo'] / 100
        ] + seg['mfcc_mean'][:5] + seg['chroma_mean'][:3]
        
        features.append(feature_vector)
    
    features = np.array(features)
    
    # Standardize and cluster
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features_scaled)
    
    # Add labels to segments
    for i, label in enumerate(labels):
        segments_data[i]['speaker_cluster'] = f"Speaker_{label + 1}"
    
    # Determine which speaker is primary (more speaking time)
    speaker_times = {}
    for seg in segments_data:
        speaker = seg['speaker_cluster']
        duration = seg['end_time'] - seg['start_time']
        speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
    
    primary_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
    
    # Add teacher/student labels
    for seg in segments_data:
        seg['is_teacher'] = seg['speaker_cluster'] == primary_speaker
        seg['role'] = 'Teacher' if seg['is_teacher'] else 'Student'
    
    print(f"🎭 Clustering results:")
    print(f"   Primary speaker (Teacher): {primary_speaker}")
    print(f"   Teacher time: {speaker_times[primary_speaker]:.1f}s")
    other_speaker = [s for s in speaker_times.keys() if s != primary_speaker][0]
    print(f"   Student time: {speaker_times[other_speaker]:.1f}s")
    
    return segments_data

def create_voice_analysis_plots(segments_data, duration):
    """Create comprehensive voice analysis plots"""
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(segments_data)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Timeline plot
    plt.subplot(4, 3, 1)
    teacher_segments = df[df['is_teacher']]
    student_segments = df[~df['is_teacher']]
    
    plt.scatter(teacher_segments['start_time'], [1]*len(teacher_segments), 
               c='blue', alpha=0.7, s=50, label='Teacher')
    plt.scatter(student_segments['start_time'], [0]*len(student_segments), 
               c='red', alpha=0.7, s=50, label='Student')
    
    plt.axvline(x=4, color='green', linestyle='--', alpha=0.7, label='Problem area 1 (0-4s)')
    if duration > 99:
        plt.axvspan(99, 117, color='orange', alpha=0.2, label='Problem area 2 (1:39-1:57)')
    plt.xlim(0, min(duration, 130))  # Show up to 2+ minutes to include second problem area
    plt.ylim(-0.5, 1.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speaker')
    plt.title('🕐 Speaking Timeline (First 60s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Energy comparison
    plt.subplot(4, 3, 2)
    plt.boxplot([teacher_segments['energy_mean'], student_segments['energy_mean']], 
                labels=['Teacher', 'Student'])
    plt.ylabel('Energy Level')
    plt.title('🔊 Voice Energy Comparison')
    plt.grid(True, alpha=0.3)
    
    # 3. Pitch comparison
    plt.subplot(4, 3, 3)
    teacher_pitch = teacher_segments['pitch_mean'][teacher_segments['pitch_mean'] > 0]
    student_pitch = student_segments['pitch_mean'][student_segments['pitch_mean'] > 0]
    
    plt.boxplot([teacher_pitch, student_pitch], labels=['Teacher', 'Student'])
    plt.ylabel('Pitch (Hz)')
    plt.title('🎵 Pitch Comparison')
    plt.grid(True, alpha=0.3)
    
    # 4. Spectral centroid comparison
    plt.subplot(4, 3, 4)
    plt.boxplot([teacher_segments['spectral_centroid'], student_segments['spectral_centroid']], 
                labels=['Teacher', 'Student'])
    plt.ylabel('Spectral Centroid (Hz)')
    plt.title('✨ Voice Brightness')
    plt.grid(True, alpha=0.3)
    
    # 5. Energy over time
    plt.subplot(4, 3, 5)
    plt.scatter(teacher_segments['start_time'], teacher_segments['energy_mean'], 
               c='blue', alpha=0.6, label='Teacher')
    plt.scatter(student_segments['start_time'], student_segments['energy_mean'], 
               c='red', alpha=0.6, label='Student')
    plt.axvline(x=4, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Energy')
    plt.title('🔊 Energy Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Pitch over time
    plt.subplot(4, 3, 6)
    teacher_with_pitch = teacher_segments[teacher_segments['pitch_mean'] > 0]
    student_with_pitch = student_segments[student_segments['pitch_mean'] > 0]
    
    plt.scatter(teacher_with_pitch['start_time'], teacher_with_pitch['pitch_mean'], 
               c='blue', alpha=0.6, label='Teacher')
    plt.scatter(student_with_pitch['start_time'], student_with_pitch['pitch_mean'], 
               c='red', alpha=0.6, label='Student')
    plt.axvline(x=4, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Pitch (Hz)')
    plt.title('🎵 Pitch Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Zero crossing rate
    plt.subplot(4, 3, 7)
    plt.boxplot([teacher_segments['zcr_mean'], student_segments['zcr_mean']], 
                labels=['Teacher', 'Student'])
    plt.ylabel('Zero Crossing Rate')
    plt.title('🌊 Voice Texture (ZCR)')
    plt.grid(True, alpha=0.3)
    
    # 8. Spectral bandwidth
    plt.subplot(4, 3, 8)
    plt.boxplot([teacher_segments['spectral_bandwidth'], student_segments['spectral_bandwidth']], 
                labels=['Teacher', 'Student'])
    plt.ylabel('Spectral Bandwidth (Hz)')
    plt.title('📊 Voice Bandwidth')
    plt.grid(True, alpha=0.3)
    
    # 9. Feature correlation heatmap
    plt.subplot(4, 3, 9)
    feature_cols = ['energy_mean', 'pitch_mean', 'spectral_centroid', 'spectral_bandwidth', 'zcr_mean']
    correlation_matrix = df[feature_cols].corr()
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('🔗 Feature Correlations')
    
    # 10. First 10 seconds detailed view
    plt.subplot(4, 3, 10)
    first_10s = df[df['start_time'] < 10]
    teacher_first_10 = first_10s[first_10s['is_teacher']]
    student_first_10 = first_10s[~first_10s['is_teacher']]
    
    plt.scatter(teacher_first_10['start_time'], teacher_first_10['energy_mean'], 
               c='blue', alpha=0.8, s=80, label='Teacher', marker='o')
    plt.scatter(student_first_10['start_time'], student_first_10['energy_mean'], 
               c='red', alpha=0.8, s=80, label='Student', marker='s')
    
    plt.axvline(x=4, color='green', linestyle='--', linewidth=2, label='Problem area')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Energy')
    plt.title('🔍 First 10 Seconds - Energy')
    plt.xlim(0, 10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 11. Pitch range comparison
    plt.subplot(4, 3, 11)
    plt.boxplot([teacher_segments['pitch_range'], student_segments['pitch_range']], 
                labels=['Teacher', 'Student'])
    plt.ylabel('Pitch Range (Hz)')
    plt.title('🎼 Pitch Variation')
    plt.grid(True, alpha=0.3)
    
    # 12. Speaking pattern (segments per minute)
    plt.subplot(4, 3, 12)
    time_bins = np.arange(0, duration + 10, 10)  # 10-second bins
    teacher_counts = []
    student_counts = []
    
    for i in range(len(time_bins) - 1):
        start_bin = time_bins[i]
        end_bin = time_bins[i + 1]
        
        teacher_in_bin = len(teacher_segments[
            (teacher_segments['start_time'] >= start_bin) & 
            (teacher_segments['start_time'] < end_bin)
        ])
        student_in_bin = len(student_segments[
            (student_segments['start_time'] >= start_bin) & 
            (student_segments['start_time'] < end_bin)
        ])
        
        teacher_counts.append(teacher_in_bin)
        student_counts.append(student_in_bin)
    
    x_pos = time_bins[:-1]
    width = 4
    
    plt.bar(x_pos, teacher_counts, width, alpha=0.7, label='Teacher', color='blue')
    plt.bar(x_pos + width, student_counts, width, alpha=0.7, label='Student', color='red')
    
    plt.axvline(x=4, color='green', linestyle='--', alpha=0.7)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Segments per 10s')
    plt.title('📈 Speaking Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, min(duration, 60))
    
    plt.tight_layout()
    plt.savefig('voice_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Comprehensive voice analysis plot saved as 'voice_analysis_comprehensive.png'")

def analyze_time_periods(segments_data, duration):
    """Analyze different time periods"""
    
    df = pd.DataFrame(segments_data)
    
    periods = [
        (0, 5, "First 5 seconds"),
        (0, 10, "First 10 seconds"), 
        (10, 30, "Seconds 10-30"),
        (30, 60, "Seconds 30-60"),
        (60, 99, "Seconds 60-99"),
        (99, 117, "1:39-1:57 (Student Period)"),  # New problematic period
        (117, duration, "After 1:57")
    ]
    
    print(f"\n⏰ TIME PERIOD ANALYSIS:")
    print("-" * 50)
    
    for start, end, label in periods:
        period_data = df[(df['start_time'] >= start) & (df['start_time'] < end)]
        
        if len(period_data) == 0:
            continue
            
        teacher_count = len(period_data[period_data['is_teacher']])
        student_count = len(period_data[~period_data['is_teacher']])
        total = teacher_count + student_count
        
        teacher_pct = (teacher_count / total * 100) if total > 0 else 0
        
        print(f"{label}:")
        print(f"  Teacher: {teacher_count}/{total} segments ({teacher_pct:.1f}%)")
        print(f"  Student: {student_count}/{total} segments ({100-teacher_pct:.1f}%)")
        
        if teacher_pct < 50 and start < 10:
            print(f"  ⚠️  WARNING: Student dominated in early period!")
        elif start == 99 and teacher_pct > 50:  # Should be student period
            print(f"  ⚠️  WARNING: Teacher detected in known student period (1:39-1:57)!")
        
        print()

def analyze_first_seconds(segments_data):
    """Detailed analysis of the problematic first seconds"""
    
    df = pd.DataFrame(segments_data)
    first_10s = df[df['start_time'] < 10].sort_values('start_time')
    
    print(f"\n🔍 DETAILED FIRST 10 SECONDS ANALYSIS:")
    print("-" * 50)
    
    print("Segment breakdown:")
    for _, seg in first_10s.iterrows():
        start = seg['start_time']
        end = seg['end_time']
        role = seg['role']
        energy = seg['energy_mean']
        pitch = seg['pitch_mean']
        
        marker = "❌" if role == "Teacher" and start < 4 else "✅"
        
        print(f"  {marker} {start:5.2f}s - {end:5.2f}s: {role:7s} "
              f"(Energy: {energy:.4f}, Pitch: {pitch:5.1f} Hz)")
    
    # Check feature differences in first 4 seconds
    first_4s = first_10s[first_10s['start_time'] < 4]
    
    if len(first_4s) > 0:
        teacher_first_4 = first_4s[first_4s['is_teacher']]
        student_first_4 = first_4s[~first_4s['is_teacher']]
        
        print(f"\nFirst 4 seconds breakdown:")
        print(f"  Teacher segments: {len(teacher_first_4)}")
        print(f"  Student segments: {len(student_first_4)}")
        
        if len(teacher_first_4) > 0:
            print(f"\n❌ PROBLEM: Teacher detected in first 4s!")
            print(f"Teacher features in first 4s:")
            print(f"  Avg Energy: {teacher_first_4['energy_mean'].mean():.4f}")
            print(f"  Avg Pitch: {teacher_first_4['pitch_mean'].mean():.1f} Hz")
            print(f"  Avg Spectral Centroid: {teacher_first_4['spectral_centroid'].mean():.1f} Hz")
        
        if len(student_first_4) > 0:
            print(f"\nStudent features in first 4s:")
            print(f"  Avg Energy: {student_first_4['energy_mean'].mean():.4f}")
            print(f"  Avg Pitch: {student_first_4['pitch_mean'].mean():.1f} Hz")
            print(f"  Avg Spectral Centroid: {student_first_4['spectral_centroid'].mean():.1f} Hz")

def analyze_problematic_periods(segments_data, y, sr):
    """Analyze both problematic periods (0-4s and 99-117s) with frequency insights"""
    
    from scipy.fft import fft, fftfreq
    
    print(f"\n🔬 FREQUENCY ANALYSIS OF PROBLEMATIC PERIODS:")
    print("=" * 60)
    
    # Define the two problematic periods
    periods = [
        (0, 4, "First 4 seconds (should be student)"),
        (99, 117, "1:39-1:57 (should be student)")
    ]
    
    df = pd.DataFrame(segments_data)
    
    for start_time, end_time, description in periods:
        print(f"\n🎯 {description}")
        print("-" * 40)
        
        # Get segments in this period
        period_segments = df[(df['start_time'] >= start_time) & (df['start_time'] < end_time)]
        
        if len(period_segments) == 0:
            print("  No segments found in this period")
            continue
        
        # Current algorithm classification
        teacher_count = len(period_segments[period_segments['is_teacher']])
        student_count = len(period_segments[~period_segments['is_teacher']])
        
        print(f"Current algorithm says:")
        print(f"  Teacher: {teacher_count} segments")
        print(f"  Student: {student_count} segments")
        
        if teacher_count > student_count:
            print(f"  ❌ PROBLEM: Algorithm thinks this is TEACHER period!")
        else:
            print(f"  ✅ Correct: Algorithm identifies this as STUDENT period")
        
        # Extract audio for this time period for frequency analysis
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        if end_sample <= len(y):
            period_audio = y[start_sample:end_sample]
            
            if len(period_audio) > 0:
                # Perform frequency band analysis
                freqs = fftfreq(len(period_audio), 1/sr)
                fft_vals = np.abs(fft(period_audio))
                
                # Define frequency bands
                low_band = (freqs >= 50) & (freqs <= 200)
                mid_band = (freqs >= 200) & (freqs <= 1000)
                high_band = (freqs >= 1000) & (freqs <= 4000)
                
                low_energy = np.sum(fft_vals[low_band])
                mid_energy = np.sum(fft_vals[mid_band])
                high_energy = np.sum(fft_vals[high_band])
                total_energy = low_energy + mid_energy + high_energy
                
                if total_energy > 0:
                    low_ratio = low_energy / total_energy
                    mid_ratio = mid_energy / total_energy
                    high_ratio = high_energy / total_energy
                    
                    print(f"\nFrequency band analysis:")
                    print(f"  Low (50-200Hz):    {low_ratio:.3f} ({low_ratio*100:.1f}%)")
                    print(f"  Mid (200-1000Hz):  {mid_ratio:.3f} ({mid_ratio*100:.1f}%)")
                    print(f"  High (1000-4000Hz): {high_ratio:.3f} ({high_ratio*100:.1f}%)")
                    
                    # Apply our frequency insights
                    print(f"\nFrequency-based prediction:")
                    
                    # Teacher characteristics: Higher mid-frequency content, balanced high frequencies
                    # Student characteristics: Lower mid-frequency content or very high-frequency content
                    
                    teacher_indicators = 0
                    student_indicators = 0
                    
                    # Mid-frequency dominance (teacher indicator)
                    if mid_ratio > 0.4:
                        teacher_indicators += 1
                        print(f"  ✓ High mid-frequency content ({mid_ratio:.3f}) suggests TEACHER")
                    else:
                        student_indicators += 1
                        print(f"  ✓ Low mid-frequency content ({mid_ratio:.3f}) suggests STUDENT")
                    
                    # High-frequency content
                    if high_ratio > 0.6:
                        student_indicators += 1
                        print(f"  ✓ Very high-frequency content ({high_ratio:.3f}) suggests STUDENT")
                    elif 0.2 <= high_ratio <= 0.5:
                        teacher_indicators += 1
                        print(f"  ✓ Balanced high-frequency content ({high_ratio:.3f}) suggests TEACHER")
                    else:
                        print(f"  ? Unclear high-frequency pattern ({high_ratio:.3f})")
                    
                    # Overall energy level
                    energy_level = np.mean(librosa.feature.rms(y=period_audio)[0])
                    print(f"  Energy level: {energy_level:.4f}")
                    
                    if energy_level < 0.01:
                        print(f"  ✓ Very low energy suggests SILENCE or quiet STUDENT")
                        student_indicators += 1
                    elif energy_level > 0.03:
                        print(f"  ✓ Higher energy suggests active TEACHER")
                        teacher_indicators += 1
                    
                    # Final frequency-based prediction
                    if teacher_indicators > student_indicators:
                        freq_prediction = "TEACHER"
                        confidence = teacher_indicators / (teacher_indicators + student_indicators)
                    elif student_indicators > teacher_indicators:
                        freq_prediction = "STUDENT"
                        confidence = student_indicators / (teacher_indicators + student_indicators)
                    else:
                        freq_prediction = "UNCLEAR"
                        confidence = 0.5
                    
                    print(f"\n🎯 Frequency-based prediction: {freq_prediction} (confidence: {confidence:.1%})")
                    
                    # Compare with expected (should be student)
                    if freq_prediction == "STUDENT":
                        print(f"  ✅ Frequency analysis CORRECTLY identifies this as STUDENT period")
                    elif freq_prediction == "TEACHER":
                        print(f"  ❌ Frequency analysis INCORRECTLY identifies this as TEACHER period")
                    else:
                        print(f"  ? Frequency analysis is UNCLEAR about this period")
                    
                    # Segment-by-segment breakdown
                    print(f"\nSegment breakdown:")
                    for _, seg in period_segments.iterrows():
                        role = "Teacher" if seg['is_teacher'] else "Student"
                        seg_start = seg['start_time']
                        seg_end = seg['end_time']
                        
                        # Extract segment audio for detailed analysis
                        seg_start_sample = int(seg_start * sr)
                        seg_end_sample = int(seg_end * sr)
                        
                        if seg_end_sample <= len(y):
                            seg_audio = y[seg_start_sample:seg_end_sample]
                            
                            if len(seg_audio) > 0:
                                # Quick frequency analysis for this segment
                                seg_freqs = fftfreq(len(seg_audio), 1/sr)
                                seg_fft = np.abs(fft(seg_audio))
                                
                                seg_low = np.sum(seg_fft[(seg_freqs >= 50) & (seg_freqs <= 200)])
                                seg_mid = np.sum(seg_fft[(seg_freqs >= 200) & (seg_freqs <= 1000)])
                                seg_high = np.sum(seg_fft[(seg_freqs >= 1000) & (seg_freqs <= 4000)])
                                seg_total = seg_low + seg_mid + seg_high
                                
                                if seg_total > 0:
                                    seg_mid_ratio = seg_mid / seg_total
                                    seg_high_ratio = seg_high / seg_total
                                    
                                    marker = "❌" if role == "Teacher" else "✅"
                                    print(f"    {marker} {seg_start:5.1f}s-{seg_end:5.1f}s: {role} "
                                          f"(Mid: {seg_mid_ratio:.2f}, High: {seg_high_ratio:.2f})")
        
        print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_voice_patterns.py <video_file>")
        print("Example: python analyze_voice_patterns.py ../../../data/IMG_4225.MP4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    if not Path(video_file).exists():
        print(f"❌ Video file not found: {video_file}")
        sys.exit(1)
    
    analyze_voice_patterns(video_file)
