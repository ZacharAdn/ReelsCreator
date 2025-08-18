"""
Results dashboard visualization for Stage 6: Output Generation
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.patches as patches

logger = logging.getLogger(__name__)


def create_results_dashboard(stage_result: Dict[str, Any], pipeline_metrics: Dict[str, Any],
                           output_dir: str = "visualizations") -> str:
    """
    Create comprehensive results dashboard
    
    Args:
        stage_result: Result from OutputGenerationStage
        pipeline_metrics: Pipeline performance metrics
        output_dir: Directory to save visualizations
        
    Returns:
        Path to saved dashboard
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Performance overview (top row)
        ax1 = fig.add_subplot(gs[0, :2])
        _plot_performance_overview(ax1, pipeline_metrics)
        
        ax2 = fig.add_subplot(gs[0, 2:])
        _plot_stage_performance(ax2, pipeline_metrics)
        
        # Processing results (middle row)
        ax3 = fig.add_subplot(gs[1, 0])
        _plot_processing_funnel(ax3, stage_result)
        
        ax4 = fig.add_subplot(gs[1, 1])
        _plot_output_files(ax4, stage_result)
        
        ax5 = fig.add_subplot(gs[1, 2:])
        _plot_quality_metrics(ax5, stage_result)
        
        # Summary and recommendations (bottom row)
        ax6 = fig.add_subplot(gs[2, :2])
        _plot_summary_info(ax6, stage_result, pipeline_metrics)
        
        ax7 = fig.add_subplot(gs[2, 2:])
        _plot_recommendations(ax7, pipeline_metrics)
        
        plt.suptitle('🎬 Reels Content Extraction - Results Dashboard', fontsize=16, fontweight='bold')
        
        # Save dashboard
        output_path = Path(output_dir) / "stage6_results_dashboard.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Results dashboard saved: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to create results dashboard: {e}")
        return ""


def _plot_performance_overview(ax, pipeline_metrics: Dict[str, Any]):
    """Plot performance overview"""
    ax.axis('off')
    
    # Performance metrics
    total_time = pipeline_metrics.get('total_pipeline_time', 0)
    processing_speed = pipeline_metrics.get('processing_speed', 0)
    efficiency = pipeline_metrics.get('efficiency', 0)
    
    # Create visual metrics display
    metrics = [
        ('Total Time', f'{total_time:.1f}s', 'steelblue'),
        ('Processing Speed', f'{processing_speed:.1f}x realtime', 'green' if processing_speed > 1 else 'orange'),
        ('Efficiency', f'{efficiency*100:.1f}%', 'green' if efficiency > 0.8 else 'orange')
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        # Create metric box
        rect = patches.Rectangle((i*0.3, 0.3), 0.25, 0.4, 
                               linewidth=2, edgecolor=color, facecolor=color, alpha=0.2)
        ax.add_patch(rect)
        
        ax.text(i*0.3 + 0.125, 0.6, value, ha='center', va='center', 
               fontsize=14, fontweight='bold', color=color)
        ax.text(i*0.3 + 0.125, 0.2, label, ha='center', va='center', 
               fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('⚡ Performance Overview', fontsize=12, fontweight='bold')


def _plot_stage_performance(ax, pipeline_metrics: Dict[str, Any]):
    """Plot stage performance breakdown"""
    stage_times = pipeline_metrics.get('stage_times', {})
    
    if not stage_times:
        ax.text(0.5, 0.5, 'No stage timing data', ha='center', va='center')
        ax.set_title('📊 Stage Performance')
        return
    
    stages = list(stage_times.keys())
    times = list(stage_times.values())
    
    # Create horizontal bar chart
    bars = ax.barh(stages, times, color=plt.cm.Set3(np.linspace(0, 1, len(stages))))
    
    # Add time labels on bars
    for bar, time in zip(bars, times):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
               f'{time:.1f}s', ha='left', va='center')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_title('📊 Stage Performance Breakdown')
    ax.grid(True, alpha=0.3, axis='x')


def _plot_processing_funnel(ax, stage_result: Dict[str, Any]):
    """Plot processing funnel"""
    export_summary = stage_result.get('export_summary', {})
    
    # Extract segment counts from export summary or estimate
    total_segments = export_summary.get('total_segments_exported', 0)
    high_value_segments = export_summary.get('high_value_segments_exported', 0)
    
    if total_segments == 0:
        ax.text(0.5, 0.5, 'No segment data', ha='center', va='center')
        ax.set_title('🔍 Processing Funnel')
        return
    
    # Create funnel visualization
    stages = ['Input\nSegments', 'Evaluated\nSegments', 'High-Value\nSegments']
    counts = [total_segments + 50, total_segments, high_value_segments]  # Add buffer for visual effect
    
    # Create trapezoids for funnel
    colors = ['lightblue', 'orange', 'gold']
    
    for i, (stage, count, color) in enumerate(zip(stages, counts, colors)):
        width = count / max(counts) * 0.8
        y_pos = 1 - (i * 0.3)
        
        # Draw trapezoid
        rect = patches.Rectangle(((1-width)/2, y_pos-0.1), width, 0.2, 
                               facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        
        # Add labels
        ax.text(0.5, y_pos, f'{stage}\n{count}', ha='center', va='center', 
               fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('🔍 Processing Funnel')
    ax.axis('off')


def _plot_output_files(ax, stage_result: Dict[str, Any]):
    """Plot output files information"""
    export_summary = stage_result.get('export_summary', {})
    output_files = stage_result.get('output_files', [])
    
    # File type counts
    file_types = {}
    for file_path in output_files:
        ext = Path(file_path).suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    
    if file_types:
        # Pie chart of file types
        labels = [f'{ext} ({count})' for ext, count in file_types.items()]
        sizes = list(file_types.values())
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(file_types)))
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        ax.set_title('📁 Output Files')
    else:
        ax.text(0.5, 0.5, 'No output files', ha='center', va='center')
        ax.set_title('📁 Output Files')


def _plot_quality_metrics(ax, stage_result: Dict[str, Any]):
    """Plot quality metrics"""
    ax.axis('off')
    
    export_summary = stage_result.get('export_summary', {})
    
    # Create quality metrics display
    metrics_text = [
        "🎯 QUALITY METRICS",
        "=" * 20,
        f"Total Segments: {export_summary.get('total_segments_exported', 0)}",
        f"High-Value: {export_summary.get('high_value_segments_exported', 0)}",
        f"Files Created: {export_summary.get('files_created', 0)}",
        f"Output Directory: {Path(export_summary.get('output_directory', '')).name}",
        f"Timestamp: {export_summary.get('timestamp', 'N/A')}"
    ]
    
    ax.text(0.05, 0.95, '\n'.join(metrics_text), transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))


def _plot_summary_info(ax, stage_result: Dict[str, Any], pipeline_metrics: Dict[str, Any]):
    """Plot summary information"""
    ax.axis('off')
    
    # Processing summary
    export_summary = stage_result.get('export_summary', {})
    
    success_rate = 0
    total_segments = export_summary.get('total_segments_exported', 0)
    high_value = export_summary.get('high_value_segments_exported', 0)
    if total_segments > 0:
        success_rate = (high_value / total_segments) * 100
    
    summary_text = [
        "📋 PROCESSING SUMMARY",
        "=" * 25,
        f"Success Rate: {success_rate:.1f}%",
        f"Processing Speed: {pipeline_metrics.get('processing_speed', 0):.1f}x realtime",
        f"Total Pipeline Time: {pipeline_metrics.get('total_pipeline_time', 0):.1f}s",
        f"Successful Stages: {pipeline_metrics.get('successful_stages', 0)}/{pipeline_metrics.get('total_stages', 0)}",
        "",
        "🎬 Ready for Reels creation!",
        f"High-quality segments exported to:",
        f"{Path(export_summary.get('output_directory', '')).name}/"
    ]
    
    ax.text(0.05, 0.95, '\n'.join(summary_text), transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))


def _plot_recommendations(ax, pipeline_metrics: Dict[str, Any]):
    """Plot performance recommendations"""
    ax.axis('off')
    
    # Generate recommendations based on performance
    recommendations = []
    
    processing_speed = pipeline_metrics.get('processing_speed', 0)
    bottleneck = pipeline_metrics.get('bottleneck_stage', '')
    efficiency = pipeline_metrics.get('efficiency', 0)
    
    recommendations.append("💡 RECOMMENDATIONS")
    recommendations.append("=" * 20)
    
    if processing_speed < 1:
        recommendations.append("⚠️  Processing slower than realtime")
        recommendations.append("   Consider using 'draft' profile")
    elif processing_speed > 5:
        recommendations.append("🚀 Excellent processing speed!")
        recommendations.append("   Consider using 'quality' profile")
    
    if bottleneck:
        recommendations.append(f"🔍 Main bottleneck: {bottleneck}")
        if bottleneck == 'transcription':
            recommendations.append("   • Use smaller Whisper model")
            recommendations.append("   • Enable GPU acceleration")
        elif bottleneck == 'evaluation':
            recommendations.append("   • Use rule-based evaluation")
            recommendations.append("   • Increase batch size")
    
    if efficiency < 0.8:
        recommendations.append("⚠️  Some stages failed")
        recommendations.append("   Check logs for errors")
    else:
        recommendations.append("✅ All stages completed successfully!")
    
    recommendations.append("")
    recommendations.append("🎯 Next steps:")
    recommendations.append("   • Review high-value segments")
    recommendations.append("   • Create Reels from best clips")
    recommendations.append("   • Iterate with different settings")
    
    ax.text(0.05, 0.95, '\n'.join(recommendations), transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))