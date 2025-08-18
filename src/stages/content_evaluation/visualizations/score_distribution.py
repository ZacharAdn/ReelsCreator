"""
Content evaluation score distribution visualization for Stage 5
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


def create_score_distribution_plot(stage_result: Dict[str, Any], output_dir: str = "visualizations") -> str:
    """
    Create score distribution visualization
    
    Args:
        stage_result: Result from ContentEvaluationStage
        output_dir: Directory to save visualizations
        
    Returns:
        Path to saved plot
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        evaluated_segments = stage_result.get('evaluated_segments', [])
        high_value_segments = stage_result.get('high_value_segments', [])
        evaluation_summary = stage_result.get('evaluation_summary', {})
        
        if not evaluated_segments:
            logger.warning("No evaluated segments found for visualization")
            return ""
        
        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Score distribution histogram
        _plot_score_histogram(ax1, evaluated_segments)
        
        # Score vs Duration scatter plot
        _plot_score_vs_duration(ax2, evaluated_segments)
        
        # High-value segments analysis
        _plot_high_value_analysis(ax3, evaluated_segments, high_value_segments, evaluation_summary)
        
        # Quality metrics summary
        _plot_quality_summary(ax4, evaluation_summary)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / "stage5_score_distribution.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Score distribution plot saved: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to create score distribution plot: {e}")
        return ""


def _plot_score_histogram(ax, evaluated_segments: List):
    """Plot score distribution histogram"""
    scores = [getattr(s, 'value_score', 0) or 0 for s in evaluated_segments]
    
    ax.hist(scores, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.2f}')
    ax.axvline(np.median(scores), color='green', linestyle='--', label=f'Median: {np.median(scores):.2f}')
    
    ax.set_xlabel('Value Score')
    ax.set_ylabel('Number of Segments')
    ax.set_title('Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_score_vs_duration(ax, evaluated_segments: List):
    """Plot score vs duration scatter plot"""
    scores = [getattr(s, 'value_score', 0) or 0 for s in evaluated_segments]
    durations = [s.duration() if hasattr(s, 'duration') else 
                (getattr(s, 'end_time', 0) - getattr(s, 'start_time', 0)) for s in evaluated_segments]
    
    # Color by confidence if available
    confidences = [getattr(s, 'confidence', 0.5) for s in evaluated_segments]
    
    scatter = ax.scatter(durations, scores, c=confidences, cmap='viridis', alpha=0.6, s=50)
    ax.set_xlabel('Duration (seconds)')
    ax.set_ylabel('Value Score')
    ax.set_title('Score vs Duration')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for confidence
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Confidence')


def _plot_high_value_analysis(ax, all_segments: List, high_value_segments: List, evaluation_summary: Dict):
    """Plot high-value segments analysis"""
    categories = ['All Segments', 'High-Value', 'Filtered Out']
    counts = [
        len(all_segments),
        len(high_value_segments),
        len(all_segments) - len(high_value_segments)
    ]
    
    colors = ['lightblue', 'gold', 'lightcoral']
    wedges, texts, autotexts = ax.pie(counts, labels=categories, colors=colors, 
                                     autopct='%1.1f%%', startangle=90)
    
    ax.set_title('Segment Quality Distribution')
    
    # Add success rate in the center
    success_rate = evaluation_summary.get('high_value_percentage', 0)
    ax.text(0, 0, f'Success Rate\n{success_rate:.1f}%', 
           horizontalalignment='center', verticalalignment='center',
           fontsize=12, fontweight='bold')


def _plot_quality_summary(ax, evaluation_summary: Dict):
    """Plot quality metrics summary"""
    ax.axis('off')
    
    # Create summary text
    summary_text = [
        "🏆 CONTENT EVALUATION SUMMARY",
        "=" * 35,
        f"Total Segments: {evaluation_summary.get('total_segments', 0)}",
        f"High-Value: {evaluation_summary.get('high_value_segments', 0)}",
        f"Success Rate: {evaluation_summary.get('high_value_percentage', 0):.1f}%",
        "",
        f"Average Score: {evaluation_summary.get('average_score', 0):.3f}",
        f"Min Score: {evaluation_summary.get('min_score', 0):.3f}",
        f"Max Score: {evaluation_summary.get('max_score', 0):.3f}",
        "",
        f"Threshold: {evaluation_summary.get('threshold_used', 0.7)}",
        f"Method: {evaluation_summary.get('evaluation_method', 'unknown')}",
        f"Embeddings: {'Yes' if evaluation_summary.get('embeddings_generated', False) else 'No'}"
    ]
    
    ax.text(0.05, 0.95, '\n'.join(summary_text), transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))


def create_reasoning_analysis_plot(stage_result: Dict[str, Any], output_dir: str = "visualizations") -> str:
    """
    Create reasoning analysis visualization
    
    Args:
        stage_result: Result from ContentEvaluationStage
        output_dir: Directory to save visualizations
        
    Returns:
        Path to saved plot
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        evaluated_segments = stage_result.get('evaluated_segments', [])
        
        if not evaluated_segments:
            logger.warning("No evaluated segments found for reasoning analysis")
            return ""
        
        # Extract reasoning keywords
        reasoning_keywords = {}
        for segment in evaluated_segments:
            reasoning = getattr(segment, 'reasoning', '') or ''
            if reasoning:
                # Extract key phrases
                for phrase in ['educational content', 'good length', 'engaging', 'practical', 'low confidence', 'technical']:
                    if phrase in reasoning.lower():
                        reasoning_keywords[phrase] = reasoning_keywords.get(phrase, 0) + 1
        
        if not reasoning_keywords:
            logger.warning("No reasoning keywords found")
            return ""
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Reasoning frequency
        phrases = list(reasoning_keywords.keys())
        counts = list(reasoning_keywords.values())
        
        ax1.barh(phrases, counts, color='steelblue', alpha=0.7)
        ax1.set_xlabel('Frequency')
        ax1.set_title('Reasoning Phrase Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Score distribution by reasoning category
        categories = ['High Educational', 'Good Length', 'Low Confidence', 'Other']
        category_scores = {cat: [] for cat in categories}
        
        for segment in evaluated_segments:
            reasoning = getattr(segment, 'reasoning', '') or ''
            score = getattr(segment, 'value_score', 0) or 0
            
            if 'educational content' in reasoning.lower():
                category_scores['High Educational'].append(score)
            elif 'good length' in reasoning.lower():
                category_scores['Good Length'].append(score)
            elif 'low confidence' in reasoning.lower():
                category_scores['Low Confidence'].append(score)
            else:
                category_scores['Other'].append(score)
        
        # Box plot
        box_data = [scores for scores in category_scores.values() if scores]
        box_labels = [cat for cat, scores in category_scores.items() if scores]
        
        if box_data:
            ax2.boxplot(box_data, labels=box_labels)
            ax2.set_ylabel('Value Score')
            ax2.set_title('Score Distribution by Reasoning Category')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / "stage5_reasoning_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Reasoning analysis plot saved: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to create reasoning analysis plot: {e}")
        return ""