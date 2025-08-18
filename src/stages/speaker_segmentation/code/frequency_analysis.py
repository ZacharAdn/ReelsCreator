#!/usr/bin/env python3
"""
Frequency Analysis Tool - Deep dive into frequency patterns
"""

import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.fft import fft, fftfreq

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

def frequency_deep_analysis(video_path: str):
    """Deep frequency analysis to understand classification issues"""
    
    print(f"🔊 Frequency Deep Analysis: {Path(video_path).name}")
    print("=" * 60)
    
    # Extract audio
    from src.video_processing import VideoProcessor
    video_processor = VideoProcessor()
    audio_path = video_processor.extract_audio(video_path, "temp_freq_analysis.wav")
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path)
        
        # 1. Create spectrogram for visual analysis
        create_spectrogram_analysis(y, sr)
        
        # 2. Analyze specific time segments
        analyze_time_segments(y, sr)
        
        # 3. Compare frequency profiles
        compare_frequency_profiles(y, sr)
        
        # 4. Formant analysis
        analyze_formants(y, sr)
        
    finally:
        if Path(audio_path).exists():
            Path(audio_path).unlink()

def create_spectrogram_analysis(y, sr):
    """Create detailed spectrogram analysis"""
    
    print("📊 Creating spectrogram analysis...")
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 16))
    fig.suptitle('Frequency Analysis - Teacher vs Student Classification', fontsize=16)
    
    # 1. Full spectrogram (first 60 seconds)
    ax = axes[0, 0]
    duration_to_show = min(60, len(y) / sr)
    y_segment = y[:int(duration_to_show * sr)]
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_segment)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax)
    ax.set_title('Full Spectrogram (0-60s)')
    ax.axvline(x=4, color='red', linestyle='--', linewidth=3, alpha=0.8, label='Problem boundary')
    ax.set_ylim(0, 3000)  # Focus on speech frequencies
    ax.legend()
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # 2. Detailed view: First 15 seconds
    ax = axes[0, 1]
    y_first_15 = y[:int(15 * sr)]
    
    D_15 = librosa.amplitude_to_db(np.abs(librosa.stft(y_first_15)), ref=np.max)
    img = librosa.display.specshow(D_15, y_axis='hz', x_axis='time', sr=sr, ax=ax)
    ax.set_title('First 15 Seconds - Problem Area')
    ax.axvline(x=4, color='red', linestyle='--', linewidth=3, alpha=0.8)
    ax.axvspan(0, 4, alpha=0.3, color='red', label='Problem zone')
    ax.set_ylim(0, 2000)
    ax.legend()
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # 3. Pitch tracking over time
    ax = axes[1, 0]
    pitches, magnitudes = librosa.piptrack(y=y[:int(30*sr)], sr=sr, threshold=0.1)
    
    # Extract pitch over time
    times = librosa.times_like(pitches, sr=sr)
    pitch_track = []
    
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_track.append(pitch if pitch > 0 else np.nan)
    
    ax.plot(times, pitch_track, 'b-', alpha=0.7, linewidth=2)
    ax.axvline(x=4, color='red', linestyle='--', linewidth=3, alpha=0.8)
    ax.axvspan(0, 4, alpha=0.3, color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Pitch (Hz)')
    ax.set_title('Pitch Tracking (First 30s)')
    ax.set_ylim(50, 500)
    ax.grid(True, alpha=0.3)
    
    # 4. Energy over time with frequency bands
    ax = axes[1, 1]
    
    # Calculate energy in different frequency bands
    times = np.arange(0, min(30, len(y)/sr), 0.5)  # Every 0.5 seconds
    low_freq_energy = []
    mid_freq_energy = []
    high_freq_energy = []
    
    for t in times:
        start_sample = int(t * sr)
        end_sample = int((t + 0.5) * sr)
        
        if end_sample < len(y):
            segment = y[start_sample:end_sample]
            
            # FFT analysis
            freqs = fftfreq(len(segment), 1/sr)
            fft_vals = np.abs(fft(segment))
            
            # Energy in different bands
            low_band = (freqs >= 50) & (freqs <= 200)    # Low frequencies
            mid_band = (freqs >= 200) & (freqs <= 1000)  # Mid frequencies  
            high_band = (freqs >= 1000) & (freqs <= 4000) # High frequencies
            
            low_freq_energy.append(np.sum(fft_vals[low_band]))
            mid_freq_energy.append(np.sum(fft_vals[mid_band]))
            high_freq_energy.append(np.sum(fft_vals[high_band]))
    
    ax.plot(times, np.array(low_freq_energy), 'r-', alpha=0.7, label='Low (50-200Hz)')
    ax.plot(times, np.array(mid_freq_energy), 'g-', alpha=0.7, label='Mid (200-1000Hz)')
    ax.plot(times, np.array(high_freq_energy), 'b-', alpha=0.7, label='High (1000-4000Hz)')
    
    ax.axvline(x=4, color='red', linestyle='--', linewidth=3, alpha=0.8)
    ax.axvspan(0, 4, alpha=0.3, color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Energy')
    ax.set_title('Frequency Band Energy Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Average frequency spectrum comparison
    ax = axes[2, 0]
    
    # Compare 0-4s vs 5-10s
    early_segment = y[0:int(4*sr)]
    later_segment = y[int(5*sr):int(10*sr)]
    
    # Calculate average spectra
    early_freqs = fftfreq(len(early_segment), 1/sr)
    early_fft = np.abs(fft(early_segment))
    
    later_freqs = fftfreq(len(later_segment), 1/sr)
    later_fft = np.abs(fft(later_segment))
    
    # Plot positive frequencies only
    early_pos = early_freqs > 0
    later_pos = later_freqs > 0
    
    ax.semilogy(early_freqs[early_pos], early_fft[early_pos], 'r-', alpha=0.7, 
                label='0-4s (Problem area)', linewidth=2)
    ax.semilogy(later_freqs[later_pos], later_fft[later_pos], 'b-', alpha=0.7, 
                label='5-10s (Later speech)', linewidth=2)
    
    ax.set_xlim(0, 3000)  # Focus on speech frequencies
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('Frequency Spectrum Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Formant analysis
    ax = axes[2, 1]
    
    # Analyze formants for different time periods
    time_periods = [
        (0, 2, "0-2s", 'red'),
        (2, 4, "2-4s", 'orange'), 
        (5, 7, "5-7s", 'blue'),
        (8, 10, "8-10s", 'green')
    ]
    
    for start_t, end_t, label, color in time_periods:
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        
        if end_sample < len(y):
            segment = y[start_sample:end_sample]
            
            # Calculate LPC (Linear Predictive Coding) for formant estimation
            # This is a simplified formant analysis
            try:
                # Pre-emphasis filter
                pre_emphasized = np.append(segment[0], segment[1:] - 0.97 * segment[:-1])
                
                # LPC analysis
                order = 12  # LPC order
                a = librosa.lpc(pre_emphasized, order=order)
                
                # Calculate frequency response
                w, h = signal.freqz([1], a, worN=512, fs=sr)
                
                # Plot
                ax.plot(w, 20 * np.log10(np.abs(h)), color=color, alpha=0.8, 
                       linewidth=2, label=label)
                
            except:
                # Fallback: simple FFT
                freqs = fftfreq(len(segment), 1/sr)
                fft_vals = np.abs(fft(segment))
                pos_freqs = freqs > 0
                
                # Smooth the spectrum
                from scipy.ndimage import gaussian_filter1d
                smoothed_fft = gaussian_filter1d(fft_vals[pos_freqs], sigma=2)
                
                ax.plot(freqs[pos_freqs][:len(smoothed_fft)], 
                       20 * np.log10(smoothed_fft + 1e-10), 
                       color=color, alpha=0.8, linewidth=2, label=label)
    
    ax.set_xlim(0, 3000)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title('Formant Analysis - Different Time Periods')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('frequency_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Detailed frequency analysis saved as 'frequency_analysis_detailed.png'")

def analyze_time_segments(y, sr):
    """Analyze specific time segments in detail"""
    
    print(f"\n🔍 TIME SEGMENT ANALYSIS:")
    print("-" * 50)
    
    segments = [
        (0, 2, "First 2 seconds (Should be student)"),
        (2, 4, "Seconds 2-4 (Should be student)"), 
        (5, 7, "Seconds 5-7 (Mixed period)"),
        (8, 10, "Seconds 8-10 (Should be teacher)")
    ]
    
    for start_t, end_t, description in segments:
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        
        if end_sample < len(y):
            segment = y[start_sample:end_sample]
            
            # Calculate features
            energy = np.mean(librosa.feature.rms(y=segment)[0])
            
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=segment, sr=sr, threshold=0.1)
            pitch_values = pitches[pitches > 0]
            avg_pitch = np.mean(pitch_values) if len(pitch_values) > 0 else 0
            pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)[0])
            spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr)[0])
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(segment)[0])
            
            # Dominant frequency analysis
            freqs = fftfreq(len(segment), 1/sr)
            fft_vals = np.abs(fft(segment))
            
            # Find peaks in spectrum
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(fft_vals, height=np.max(fft_vals) * 0.1)
            dominant_freqs = freqs[peaks]
            dominant_freqs = dominant_freqs[(dominant_freqs > 50) & (dominant_freqs < 2000)]
            
            print(f"\n{description}:")
            print(f"  Energy: {energy:.4f}")
            print(f"  Avg Pitch: {avg_pitch:.1f} Hz (±{pitch_std:.1f})")
            print(f"  Spectral Centroid: {spectral_centroid:.1f} Hz")
            print(f"  Spectral Bandwidth: {spectral_bandwidth:.1f} Hz")
            print(f"  Zero Crossing Rate: {zcr:.4f}")
            print(f"  Dominant Frequencies: {dominant_freqs[:5]}")

def compare_frequency_profiles(y, sr):
    """Compare frequency profiles to understand classification issues"""
    
    print(f"\n📈 FREQUENCY PROFILE COMPARISON:")
    print("-" * 50)
    
    # Define segments for comparison
    problem_segment = y[0:int(4*sr)]  # 0-4s (labeled as teacher, should be student)
    clear_student = y[int(5*sr):int(9*sr)]  # 5-9s (clear student periods)
    clear_teacher = y[int(10*sr):int(14*sr)]  # 10-14s (clear teacher periods)
    
    segments = [
        (problem_segment, "Problem area (0-4s)", "Labeled as TEACHER"),
        (clear_student, "Clear student (5-9s)", "Should be STUDENT"),
        (clear_teacher, "Clear teacher (10-14s)", "Should be TEACHER")
    ]
    
    print("Comparing frequency characteristics:")
    
    for segment, name, label in segments:
        if len(segment) == 0:
            continue
            
        # Fundamental frequency analysis
        pitches, magnitudes = librosa.piptrack(y=segment, sr=sr, threshold=0.1)
        pitch_values = pitches[pitches > 0]
        
        # Frequency distribution
        freqs = fftfreq(len(segment), 1/sr)
        fft_vals = np.abs(fft(segment))
        
        # Calculate energy in key frequency bands
        low_energy = np.sum(fft_vals[(freqs >= 50) & (freqs <= 200)])
        mid_energy = np.sum(fft_vals[(freqs >= 200) & (freqs <= 1000)])
        high_energy = np.sum(fft_vals[(freqs >= 1000) & (freqs <= 4000)])
        
        total_energy = low_energy + mid_energy + high_energy
        
        print(f"\n{name} ({label}):")
        print(f"  Avg Pitch: {np.mean(pitch_values):.1f} Hz" if len(pitch_values) > 0 else "  No clear pitch")
        print(f"  Pitch Range: {np.max(pitch_values) - np.min(pitch_values):.1f} Hz" if len(pitch_values) > 0 else "")
        print(f"  Low Freq Energy (50-200Hz): {low_energy/total_energy*100:.1f}%")
        print(f"  Mid Freq Energy (200-1000Hz): {mid_energy/total_energy*100:.1f}%")
        print(f"  High Freq Energy (1000-4000Hz): {high_energy/total_energy*100:.1f}%")
        
        # Find spectral peaks
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(fft_vals, height=np.max(fft_vals) * 0.2)
        peak_freqs = freqs[peaks]
        peak_freqs = peak_freqs[(peak_freqs > 50) & (peak_freqs < 2000)]
        
        print(f"  Main Spectral Peaks: {peak_freqs[:3]}")

def analyze_formants(y, sr):
    """Analyze formants (vocal tract resonances)"""
    
    print(f"\n🎵 FORMANT ANALYSIS:")
    print("-" * 50)
    
    # Analyze formants for key time periods
    periods = [
        (0, 4, "Problem period (0-4s)"),
        (5, 9, "Student period (5-9s)"),
        (10, 14, "Teacher period (10-14s)")
    ]
    
    for start_t, end_t, description in periods:
        start_sample = int(start_t * sr)
        end_sample = int(end_t * sr)
        
        if end_sample < len(y):
            segment = y[start_sample:end_sample]
            
            print(f"\n{description}:")
            
            try:
                # Estimate formants using LPC
                # Pre-emphasis
                pre_emphasized = np.append(segment[0], segment[1:] - 0.97 * segment[:-1])
                
                # LPC analysis
                order = 12
                a = librosa.lpc(pre_emphasized, order=order)
                
                # Find formants (peaks in LPC spectrum)
                w, h = signal.freqz([1], a, worN=1024, fs=sr)
                
                from scipy.signal import find_peaks
                spectrum_db = 20 * np.log10(np.abs(h) + 1e-10)
                peaks, _ = find_peaks(spectrum_db, height=np.max(spectrum_db) - 20)
                
                formant_freqs = w[peaks] / (2 * np.pi) * sr
                formant_freqs = formant_freqs[formant_freqs < 3000]  # Only speech formants
                formant_freqs = sorted(formant_freqs)
                
                print(f"  Estimated formants: {formant_freqs[:4]} Hz")
                
                if len(formant_freqs) >= 2:
                    f1, f2 = formant_freqs[0], formant_freqs[1]
                    print(f"  F1 (jaw height): {f1:.0f} Hz")
                    print(f"  F2 (tongue position): {f2:.0f} Hz")
                    print(f"  F2/F1 ratio: {f2/f1:.2f}")
                
            except Exception as e:
                print(f"  Formant analysis failed: {e}")
                # Fallback: simple spectral analysis
                freqs = fftfreq(len(segment), 1/sr)
                fft_vals = np.abs(fft(segment))
                
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(fft_vals, height=np.max(fft_vals) * 0.3)
                peak_freqs = freqs[peaks]
                peak_freqs = peak_freqs[(peak_freqs > 100) & (peak_freqs < 3000)]
                peak_freqs = sorted(peak_freqs)
                
                print(f"  Dominant frequencies: {peak_freqs[:4]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python frequency_analysis.py <video_file>")
        print("Example: python frequency_analysis.py ../../../data/IMG_4225.MP4")
        sys.exit(1)
    
    video_file = sys.argv[1]
    if not Path(video_file).exists():
        print(f"❌ Video file not found: {video_file}")
        sys.exit(1)
    
    frequency_deep_analysis(video_file)
