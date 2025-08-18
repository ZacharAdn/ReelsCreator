"""
Audio waveform visualization for Stage 1: Audio Extraction
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available - audio visualizations will be limited")

logger = logging.getLogger(__name__)


def create_waveform_plot(audio_path: str, output_path: Optional[str] = None, 
                        duration_limit: float = 60.0) -> str:
    """
    Create waveform visualization for audio file
    
    Args:
        audio_path: Path to audio file
        output_path: Where to save the plot (optional)
        duration_limit: Maximum duration to visualize (seconds)
        
    Returns:
        Path to saved plot
    """
    if not LIBROSA_AVAILABLE:
        logger.error("Cannot create waveform plot - librosa not installed")
        return ""
    
    try:
        # Load audio
        y, sr = librosa.load(audio_path, duration=duration_limit)
        
        # Create plot
        plt.figure(figsize=(15, 8))
        
        # Main waveform
        plt.subplot(2, 1, 1)
        time_axis = np.linspace(0, len(y)/sr, len(y))
        plt.plot(time_axis, y)
        plt.title(f'Audio Waveform - {Path(audio_path).name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
        
        # Spectrogram
        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Spectrogram')
        
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            audio_name = Path(audio_path).stem
            output_path = f"audio_waveform_{audio_name}.png"
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Waveform plot saved: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to create waveform plot: {e}")
        return ""


def create_audio_analysis_plot(stage_result: Dict[str, Any], output_dir: str = "visualizations") -> str:
    """
    Create comprehensive audio analysis visualization
    
    Args:
        stage_result: Result from AudioExtractionStage
        output_dir: Directory to save visualizations
        
    Returns:
        Path to saved plot
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        audio_path = stage_result.get('audio_path', '')
        video_info = stage_result.get('video_info', {})
        
        if not audio_path or not Path(audio_path).exists():
            logger.warning("No valid audio path found for visualization")
            return ""
        
        # Create waveform plot
        waveform_path = Path(output_dir) / "stage1_audio_waveform.png"
        result_path = create_waveform_plot(audio_path, str(waveform_path))
        
        # Create summary info plot
        if video_info:
            info_plot_path = Path(output_dir) / "stage1_audio_info.png"
            _create_info_plot(stage_result, str(info_plot_path))
        
        return result_path
        
    except Exception as e:
        logger.error(f"Failed to create audio analysis plot: {e}")
        return ""


def _create_info_plot(stage_result: Dict[str, Any], output_path: str) -> None:
    """Create audio information summary plot"""
    video_info = stage_result.get('video_info', {})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Create text summary
    info_text = [
        "🎵 AUDIO EXTRACTION SUMMARY",
        "=" * 40,
        f"Duration: {stage_result.get('duration', 0):.1f} seconds",
        f"Sample Rate: {stage_result.get('sample_rate', 0)} Hz",
        f"Video Size: {video_info.get('size', 'Unknown')}",
        f"Video FPS: {video_info.get('fps', 'Unknown')}",
        f"Has Audio: {'Yes' if video_info.get('has_audio') else 'No'}",
        f"Audio Path: {Path(stage_result.get('audio_path', '')).name}"
    ]
    
    ax.text(0.1, 0.9, '\n'.join(info_text), transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Audio info plot saved: {output_path}")