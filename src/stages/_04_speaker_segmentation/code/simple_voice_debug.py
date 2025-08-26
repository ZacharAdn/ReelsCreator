#!/usr/bin/env python3
"""
Simple Voice Debug Tool - Clear visualization of teacher vs student classification
"""

import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def debug_voice_classification(video_path: str):
    """Simple debug of voice classification with clear plots"""
    
    print(f"🔍 Simple Voice Debug: {Path(video_path).name}")
    print("=" * 50)
    
    # Extract audio
    from src.video_processing import VideoProcessor
    video_processor = VideoProcessor()
    audio_path = video_processor.extract_audio(video_path, "temp_debug.wav")
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path)
        
        # Extract segments with simple features
        segments = extract_simple_segments(y, sr)
        
        if len(segments) < 10:
            print("❌ Not enough segments")
            return
        
        # Cluster into teacher/student
        segments_with_labels = classify_speakers(segments)
        
        # Create simple, clear plots
        create_debug_plots(segments_with_labels)
        
        # Print the problem clearly
        analyze_problem(segments_with_labels)
        
    finally:
        if Path(audio_path).exists():
            Path(audio_path).unlink()

def extract_simple_segments(y, sr):
    """Extract segments with just the essential features"""
    
    window_size = 0.5  # 0.5 second windows
    hop_size = 0.25    # 0.25 second overlap
    
    samples_per_window = int(window_size * sr)
    hop_samples = int(hop_size * sr)
    
    segments = []
    
    for i in range(0, len(y) - samples_per_window, hop_samples):
        segment_audio = y[i:i + samples_per_window]
        
        start_time = i / sr
        end_time = (i + samples_per_window) / sr
        
        # Calculate basic features
        energy = np.mean(librosa.feature.rms(y=segment_audio)[0])
        
        # Skip very quiet segments
        if energy < 0.003:
            continue
        
        # Pitch
        pitches, _ = librosa.piptrack(y=segment_audio, sr=sr, threshold=0.05)
        pitch_values = pitches[pitches > 0]
        pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0
        
        # Spectral centroid (brightness)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment_audio, sr=sr)[0])
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(segment_audio)[0])
        
        segments.append({
            'start_time': start_time,
            'end_time': end_time,
            'energy': energy,
            'pitch': pitch_mean,
            'spectral_centroid': spectral_centroid,
            'zcr': zcr
        })
    
    return segments

def classify_speakers(segments):
    """Classify segments into teacher/student"""
    
    # Build feature matrix
    features = []
    for seg in segments:
        feature_vector = [
            seg['energy'] * 1000,
            seg['pitch'] / 100 if seg['pitch'] > 0 else 0,
            seg['spectral_centroid'] / 1000,
            seg['zcr'] * 1000
        ]
        features.append(feature_vector)
    
    features = np.array(features)
    
    # Standardize and cluster
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features_scaled)
    
    # Add labels to segments
    for i, label in enumerate(labels):
        segments[i]['cluster'] = label
        segments[i]['speaker'] = f"Speaker_{label + 1}"
    
    # Determine which cluster is teacher (more speaking time)
    cluster_times = {0: 0, 1: 0}
    for seg in segments:
        duration = seg['end_time'] - seg['start_time']
        cluster_times[seg['cluster']] += duration
    
    teacher_cluster = 0 if cluster_times[0] > cluster_times[1] else 1
    
    # Add final labels
    for seg in segments:
        seg['is_teacher'] = seg['cluster'] == teacher_cluster
        seg['role'] = 'Teacher' if seg['is_teacher'] else 'Student'
    
    print(f"📊 Classification results:")
    print(f"   Teacher cluster: {teacher_cluster}")
    print(f"   Teacher time: {cluster_times[teacher_cluster]:.1f}s")
    print(f"   Student time: {cluster_times[1-teacher_cluster]:.1f}s")
    
    return segments

def create_debug_plots(segments):
    """Create simple, clear debug plots"""
    
    df = pd.DataFrame(segments)
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Voice Classification Debug - Teacher vs Student', fontsize=16)
    
    # Plot 1: Timeline (focus on first 30 seconds)
    ax = axes[0, 0]
    first_30 = df[df['start_time'] < 30]
    
    teacher_data = first_30[first_30['is_teacher']]
    student_data = first_30[~first_30['is_teacher']]
    
    ax.scatter(teacher_data['start_time'], [1]*len(teacher_data), 
              c='blue', s=80, alpha=0.7, label='Teacher')
    ax.scatter(student_data['start_time'], [0]*len(student_data), 
              c='red', s=80, alpha=0.7, label='Student')
    
    ax.axvline(x=4, color='green', linestyle='--', linewidth=2, label='4s Problem Line')
    ax.set_xlim(0, 30)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Speaker Classification')
    ax.set_title('Timeline - First 30 seconds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Energy comparison
    ax = axes[0, 1]
    teacher_energy = df[df['is_teacher']]['energy']
    student_energy = df[~df['is_teacher']]['energy']
    
    ax.boxplot([teacher_energy, student_energy], tick_labels=['Teacher', 'Student'])
    ax.set_ylabel('Energy Level')
    ax.set_title('Energy Comparison')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Pitch comparison
    ax = axes[0, 2]
    teacher_pitch = df[df['is_teacher'] & (df['pitch'] > 0)]['pitch']
    student_pitch = df[~df['is_teacher'] & (df['pitch'] > 0)]['pitch']
    
    ax.boxplot([teacher_pitch, student_pitch], tick_labels=['Teacher', 'Student'])
    ax.set_ylabel('Pitch (Hz)')
    ax.set_title('Pitch Comparison')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Energy over time (first 15 seconds - the problem area)
    ax = axes[1, 0]
    first_15 = df[df['start_time'] < 15]
    
    teacher_15 = first_15[first_15['is_teacher']]
    student_15 = first_15[~first_15['is_teacher']]
    
    ax.scatter(teacher_15['start_time'], teacher_15['energy'], 
              c='blue', s=100, alpha=0.8, label='Teacher')
    ax.scatter(student_15['start_time'], student_15['energy'], 
              c='red', s=100, alpha=0.8, label='Student')
    
    ax.axvline(x=4, color='green', linestyle='--', linewidth=3, alpha=0.8, label='Problem: Teacher before 4s')
    ax.axvspan(0, 4, alpha=0.2, color='red', label='Problem Zone')
    
    ax.set_xlim(0, 15)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Energy')
    ax.set_title('PROBLEM AREA: Energy in First 15 Seconds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Feature scatter (Energy vs Pitch)
    ax = axes[1, 1]
    
    # Filter out zero pitches for clearer plot
    df_with_pitch = df[df['pitch'] > 0]
    teacher_scatter = df_with_pitch[df_with_pitch['is_teacher']]
    student_scatter = df_with_pitch[~df_with_pitch['is_teacher']]
    
    ax.scatter(teacher_scatter['energy'], teacher_scatter['pitch'], 
              c='blue', s=60, alpha=0.6, label='Teacher')
    ax.scatter(student_scatter['energy'], student_scatter['pitch'], 
              c='red', s=60, alpha=0.6, label='Student')
    
    ax.set_xlabel('Energy')
    ax.set_ylabel('Pitch (Hz)')
    ax.set_title('Feature Space: Energy vs Pitch')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: First 10 seconds detailed
    ax = axes[1, 2]
    first_10 = df[df['start_time'] < 10].sort_values('start_time')
    
    colors = ['blue' if teacher else 'red' for teacher in first_10['is_teacher']]
    markers = ['o' if teacher else 's' for teacher in first_10['is_teacher']]
    
    for i, (_, row) in enumerate(first_10.iterrows()):
        ax.scatter(row['start_time'], i, 
                  c='blue' if row['is_teacher'] else 'red',
                  s=150, alpha=0.8,
                  marker='o' if row['is_teacher'] else 's')
    
    ax.axvline(x=4, color='green', linestyle='--', linewidth=3, alpha=0.8)
    ax.axvspan(0, 4, alpha=0.3, color='red')
    
    ax.set_xlim(0, 10)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Segment Index')
    ax.set_title('First 10 Seconds: Each Segment')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.8, label='Teacher'),
        Patch(facecolor='red', alpha=0.8, label='Student'),
        Patch(facecolor='red', alpha=0.3, label='Problem Zone (0-4s)')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('voice_debug_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Debug plots saved as 'voice_debug_simple.png'")

def analyze_problem(segments):
    """Analyze and explain the problem clearly"""
    
    print(f"\n🔍 PROBLEM ANALYSIS:")
    print("=" * 50)
    
    # Focus on first 10 seconds
    first_10 = [seg for seg in segments if seg['start_time'] < 10]
    first_4 = [seg for seg in segments if seg['start_time'] < 4]
    
    print(f"📊 First 10 seconds:")
    teacher_count_10 = len([s for s in first_10 if s['is_teacher']])
    student_count_10 = len([s for s in first_10 if not s['is_teacher']])
    print(f"   Teacher segments: {teacher_count_10}")
    print(f"   Student segments: {student_count_10}")
    
    print(f"\n❌ PROBLEM - First 4 seconds:")
    teacher_count_4 = len([s for s in first_4 if s['is_teacher']])
    student_count_4 = len([s for s in first_4 if not s['is_teacher']])
    print(f"   Teacher segments: {teacher_count_4} (SHOULD BE 0!)")
    print(f"   Student segments: {student_count_4}")
    
    if teacher_count_4 > 0:
        print(f"\n🔍 WHY IS THIS HAPPENING?")
        
        # Compare features between early teacher and later student
        early_teacher = [s for s in first_4 if s['is_teacher']]
        later_student = [s for s in segments if not s['is_teacher'] and s['start_time'] > 10]
        
        if early_teacher and later_student:
            print(f"\n📈 Feature comparison:")
            
            early_teacher_avg_energy = np.mean([s['energy'] for s in early_teacher])
            later_student_avg_energy = np.mean([s['energy'] for s in later_student])
            
            early_teacher_avg_pitch = np.mean([s['pitch'] for s in early_teacher if s['pitch'] > 0])
            later_student_avg_pitch = np.mean([s['pitch'] for s in later_student if s['pitch'] > 0])
            
            print(f"   Early 'Teacher' (0-4s):")
            print(f"     Energy: {early_teacher_avg_energy:.4f}")
            print(f"     Pitch: {early_teacher_avg_pitch:.1f} Hz")
            
            print(f"   Later Student (10s+):")
            print(f"     Energy: {later_student_avg_energy:.4f}")
            print(f"     Pitch: {later_student_avg_pitch:.1f} Hz")
            
            print(f"\n💡 ANALYSIS:")
            if early_teacher_avg_energy < later_student_avg_energy:
                print("   🔍 Early segments have LOWER energy than student segments!")
                print("   🤔 This suggests they might actually be STUDENT, not teacher")
            
            if abs(early_teacher_avg_pitch - later_student_avg_pitch) < 20:
                print("   🔍 Early segments have SIMILAR pitch to student segments!")
                print("   🤔 The algorithm might be confusing them")
    
    # Show detailed timeline
    print(f"\n⏰ DETAILED TIMELINE (first 8 seconds):")
    first_8 = sorted([s for s in segments if s['start_time'] < 8], key=lambda x: x['start_time'])
    
    for seg in first_8:
        marker = "❌" if seg['is_teacher'] and seg['start_time'] < 4 else "✅"
        role = seg['role']
        energy = seg['energy']
        pitch = seg['pitch']
        
        print(f"  {marker} {seg['start_time']:5.2f}s: {role:7s} "
              f"(Energy: {energy:.4f}, Pitch: {pitch:6.1f} Hz)")
    
    print(f"\n🎯 CONCLUSION:")
    if teacher_count_4 > 0:
        print("   The algorithm is incorrectly classifying early segments as 'Teacher'")
        print("   These early segments likely belong to a STUDENT speaking first")
        print("   Need to improve the classification algorithm!")
    else:
        print("   ✅ Classification looks correct!")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python simple_voice_debug.py <video_file>")
        print("Example: python simple_voice_debug.py ../../../data/IMG_4225.MP4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    if not Path(video_file).exists():
        print(f"❌ Video file not found: {video_file}")
        sys.exit(1)
    
    debug_voice_classification(video_file)
