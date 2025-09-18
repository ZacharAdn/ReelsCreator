"""
Performance monitoring for the pipeline
"""

import time
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single stage"""
    stage_name: str
    execution_time: float
    success: bool
    timestamp: float
    error: str = None
    input_size: int = 0
    output_size: int = 0
    

class PerformanceMonitor:
    """
    Monitor performance across the entire pipeline
    """
    
    def __init__(self):
        self.pipeline_start_time = None
        self.stage_metrics: List[StageMetrics] = []
        self.pipeline_metrics = {}
    
    def start_pipeline(self) -> None:
        """Start monitoring the pipeline"""
        self.pipeline_start_time = time.time()
        self.stage_metrics = []
        logger.info("Started pipeline performance monitoring")
    
    def add_stage_metrics(self, stage_metrics: Dict[str, Any]) -> None:
        """
        Add metrics from a stage
        
        Args:
            stage_metrics: Metrics dictionary from stage
        """
        metrics = StageMetrics(
            stage_name=stage_metrics.get('stage_name', 'unknown'),
            execution_time=stage_metrics.get('execution_time', 0),
            success=stage_metrics.get('success', False),
            timestamp=stage_metrics.get('timestamp', time.time()),
            error=stage_metrics.get('error'),
            input_size=stage_metrics.get('input_size', 0),
            output_size=stage_metrics.get('output_size', 0)
        )
        
        self.stage_metrics.append(metrics)
        logger.debug(f"Added metrics for stage: {metrics.stage_name}")
    
    def finish_pipeline(self, total_duration: float = 0) -> Dict[str, Any]:
        """
        Finish monitoring and calculate final metrics
        
        Args:
            total_duration: Total content duration in seconds
            
        Returns:
            Complete pipeline metrics
        """
        if not self.pipeline_start_time:
            logger.warning("Pipeline monitoring was not started")
            return {}
        
        total_pipeline_time = time.time() - self.pipeline_start_time
        
        # Calculate stage-wise metrics
        stage_times = {m.stage_name: m.execution_time for m in self.stage_metrics}
        successful_stages = [m for m in self.stage_metrics if m.success]
        failed_stages = [m for m in self.stage_metrics if not m.success]
        
        # Find bottleneck (excluding 0-time stages)
        meaningful_stages = [(m.stage_name, m.execution_time) for m in self.stage_metrics if m.execution_time > 0.1]
        bottleneck_stage = max(meaningful_stages, key=lambda x: x[1])[0] if meaningful_stages else None
        
        # Calculate processing speed
        processing_speed = 0
        if total_duration > 0 and total_pipeline_time > 0:
            processing_speed = total_duration / total_pipeline_time
        
        self.pipeline_metrics = {
            'total_pipeline_time': total_pipeline_time,
            'total_content_duration': total_duration,
            'processing_speed': processing_speed,
            'processing_speed_text': f"{processing_speed:.1f}x realtime" if processing_speed > 0 else "N/A",
            'stage_times': stage_times,
            'bottleneck_stage': bottleneck_stage,
            'successful_stages': len(successful_stages),
            'failed_stages': len(failed_stages),
            'total_stages': len(self.stage_metrics),
            'efficiency': self.calculate_efficiency(),
            'stage_breakdown': [asdict(m) for m in self.stage_metrics]
        }
        
        logger.info(f"Pipeline completed in {total_pipeline_time:.2f}s")
        if processing_speed > 0:
            logger.info(f"Processing speed: {processing_speed:.1f}x realtime")
        if bottleneck_stage:
            logger.info(f"Main bottleneck: {bottleneck_stage}")
        
        return self.pipeline_metrics
    
    def calculate_efficiency(self) -> float:
        """
        Calculate overall pipeline efficiency
        
        Returns:
            Efficiency score (0-1)
        """
        if not self.stage_metrics:
            return 0.0
        
        successful_stages = len([m for m in self.stage_metrics if m.success])
        total_stages = len(self.stage_metrics)
        
        return successful_stages / total_stages if total_stages > 0 else 0.0
    
    def get_stage_performance(self, stage_name: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific stage
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Stage performance metrics
        """
        stage_metrics = [m for m in self.stage_metrics if m.stage_name == stage_name]
        
        if not stage_metrics:
            return {'error': f'No metrics found for stage: {stage_name}'}
        
        latest_metrics = stage_metrics[-1]  # Get most recent
        return asdict(latest_metrics)
    
    def generate_performance_report(self) -> str:
        """
        Generate a human-readable performance report
        
        Returns:
            Performance report string
        """
        if not self.pipeline_metrics:
            return "No performance data available"
        
        report = []
        report.append("=== PIPELINE PERFORMANCE REPORT ===")
        report.append(f"Total Pipeline Time: {self.pipeline_metrics['total_pipeline_time']:.2f}s")
        report.append(f"Content Duration: {self.pipeline_metrics['total_content_duration']:.1f}s")
        report.append(f"Processing Speed: {self.pipeline_metrics['processing_speed_text']}")
        report.append(f"Pipeline Efficiency: {self.pipeline_metrics['efficiency']*100:.1f}%")
        
        if self.pipeline_metrics['bottleneck_stage']:
            report.append(f"Main Bottleneck: {self.pipeline_metrics['bottleneck_stage']}")
        
        report.append("\n=== STAGE BREAKDOWN ===")
        for stage_name, stage_time in self.pipeline_metrics['stage_times'].items():
            percentage = (stage_time / self.pipeline_metrics['total_pipeline_time']) * 100
            report.append(f"{stage_name}: {stage_time:.2f}s ({percentage:.1f}%)")
        
        if self.pipeline_metrics['failed_stages'] > 0:
            report.append(f"\n⚠️  {self.pipeline_metrics['failed_stages']} stages failed")
        
        return "\n".join(report)