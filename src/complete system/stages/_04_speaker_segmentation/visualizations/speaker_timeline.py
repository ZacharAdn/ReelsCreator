"""
Speaker segmentation timeline visualization for Stage 2
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def create_speaker_timeline(stage_result: Dict[str, Any], output_dir: str = "visualizations") -> str:
    """
    Create speaker timeline visualization
    
    Args:
        stage_result: Result from SpeakerSegmentationStage
        output_dir: Directory to save visualizations
        
    Returns:
        Path to saved plot
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        speaker_segments = stage_result.get('speaker_segments', [])
        duration = stage_result.get('duration', 0)
        
        if not speaker_segments:
            logger.warning("No speaker segments found for visualization")
            return ""
        
        # Create timeline plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Speaker timeline
        _plot_speaker_timeline(ax1, speaker_segments, duration)
        
        # Speaker statistics
        _plot_speaker_stats(ax2, speaker_segments)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / "stage2_speaker_timeline.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Speaker timeline saved: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to create speaker timeline: {e}")
        return ""


def _plot_speaker_timeline(ax, speaker_segments: List[Dict], total_duration: float):
    """Plot speaker timeline"""
    # Colors for different speakers
    speaker_colors = {
        'TEACHER': '#2E8B57',    # Sea green
        'STUDENT': '#4169E1',    # Royal blue
        'UNKNOWN': '#808080'     # Gray
    }
    
    # Plot segments
    y_pos = 0.5
    for segment in speaker_segments:
        start_time = segment.get('start_time', 0)
        end_time = segment.get('end_time', start_time + segment.get('duration', 1))
        speaker = segment.get('speaker', 'UNKNOWN')
        confidence = segment.get('confidence', 0.5)
        
        # Adjust alpha based on confidence
        alpha = max(0.3, confidence)
        color = speaker_colors.get(speaker, '#808080')
        
        # Draw rectangle for segment
        ax.barh(y_pos, end_time - start_time, left=start_time, height=0.3, 
               color=color, alpha=alpha, edgecolor='black', linewidth=0.5)
    
    ax.set_xlim(0, max(total_duration, 1))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Speaker')
    ax.set_title('Speaker Segmentation Timeline')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, color=color, alpha=0.7, label=speaker) 
                      for speaker, color in speaker_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')


def _plot_speaker_stats(ax, speaker_segments: List[Dict]):
    """Plot speaker statistics"""
    # Calculate statistics
    speaker_stats = {}
    total_duration = 0
    
    for segment in speaker_segments:
        speaker = segment.get('speaker', 'UNKNOWN')
        duration = segment.get('duration', segment.get('end_time', 0) - segment.get('start_time', 0))
        
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {'count': 0, 'total_time': 0, 'confidences': []}
        
        speaker_stats[speaker]['count'] += 1
        speaker_stats[speaker]['total_time'] += duration
        speaker_stats[speaker]['confidences'].append(segment.get('confidence', 0.5))
        total_duration += duration
    
    # Create bar chart
    speakers = list(speaker_stats.keys())
    percentages = [speaker_stats[speaker]['total_time'] / total_duration * 100 for speaker in speakers]
    avg_confidences = [np.mean(speaker_stats[speaker]['confidences']) for speaker in speakers]
    
    bars = ax.bar(speakers, percentages, alpha=0.7)
    
    # Color bars based on average confidence
    for bar, confidence in zip(bars, avg_confidences):
        # Color intensity based on confidence
        intensity = confidence
        if speakers[bars.index(bar)] == 'TEACHER':
            bar.set_color(plt.cm.Greens(intensity))
        else:
            bar.set_color(plt.cm.Blues(intensity))
    
    ax.set_ylabel('Speaking Time (%)')
    ax.set_title('Speaker Time Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{percentage:.1f}%', ha='center', va='bottom')


def create_speaker_accuracy_plot(stage_result: Dict[str, Any], output_dir: str = "visualizations") -> str:
    """
    Create speaker accuracy analysis plot
    
    Args:
        stage_result: Result from SpeakerSegmentationStage
        output_dir: Directory to save visualizations
        
    Returns:
        Path to saved plot
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        accuracy_improvements = stage_result.get('accuracy_improvements', {})
        processing_summary = stage_result.get('analysis_summary', {})
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Confidence distribution
        speaker_segments = stage_result.get('speaker_segments', [])
        confidences = [s.get('confidence', 0) for s in speaker_segments]
        
        ax1.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Number of Segments')
        ax1.set_title('Confidence Score Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy improvements
        if accuracy_improvements:
            improvements = [
                ('Initial Segments', accuracy_improvements.get('initial_segments', 0)),
                ('Final Segments', accuracy_improvements.get('final_segments', 0)),
                ('Transitions Reduced', accuracy_improvements.get('transitions_reduced', 0))
            ]
            
            categories, values = zip(*improvements)
            ax2.bar(categories, values, alpha=0.7)
            ax2.set_title('Processing Improvements')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Processing summary
        if processing_summary:
            summary_text = [
                f"Total Segments: {processing_summary.get('total_segments', 0)}",
                f"PyAnnote Used: {'Yes' if processing_summary.get('pyannote_used', False) else 'No'}",
                f"Classifier Trained: {'Yes' if processing_summary.get('classifier_trained', False) else 'No'}"
            ]
            
            ax3.text(0.1, 0.5, '\n'.join(summary_text), transform=ax3.transAxes,
                    fontsize=12, verticalalignment='center', fontfamily='monospace')
            ax3.set_title('Processing Summary')
            ax3.axis('off')
        
        # Fallback information
        fallback_used = stage_result.get('fallback_used', False)
        detector_info = stage_result.get('detector_info', {})
        
        fallback_text = [
            f"Fallback Used: {'Yes' if fallback_used else 'No'}",
            f"PyAnnote Available: {'Yes' if detector_info.get('pyannote_available', False) else 'No'}",
            f"Frequency Analyzer: {detector_info.get('frequency_analyzer', 'Unknown')}",
            f"Sample Rate: {detector_info.get('sample_rate', 'Unknown')} Hz"
        ]
        
        ax4.text(0.1, 0.5, '\n'.join(fallback_text), transform=ax4.transAxes,
                fontsize=12, verticalalignment='center', fontfamily='monospace')
        ax4.set_title('System Information')
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / "stage2_speaker_accuracy.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Speaker accuracy plot saved: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to create speaker accuracy plot: {e}")
        return ""