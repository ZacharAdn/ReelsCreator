"""
Output Generation Stage - Export results in various formats
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.base_stage import BaseStage
from shared.exceptions import StageException
from shared.models import Segment, ProcessingResult
from shared.utils import format_time, ensure_directory_exists

logger = logging.getLogger(__name__)


class OutputGenerationStage(BaseStage):
    """
    Stage 6: Generate outputs in multiple formats (CSV, JSON, reports)
    
    Input: {
        'evaluated_segments': List[Segment],
        'high_value_segments': List[Segment],
        'evaluation_summary': Dict,
        ...
    }
    Output: {
        'output_files': List[str],
        'export_summary': Dict
    }
    """
    
    def __init__(self, config):
        super().__init__(config, "OutputGeneration")
        
        # Create date-based output directory structure like: results/2025-09-10/
        base_dir = Path(getattr(config, 'output_dir', 'results'))
        date_folder = datetime.now().strftime("%Y-%m-%d")
        self.output_dir = base_dir / date_folder
        self.export_formats = getattr(config, 'export_formats', ['csv', 'json', 'report'])
        
        # Ensure output directory exists
        ensure_directory_exists(self.output_dir)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input contains evaluated segments"""
        super().validate_input(input_data)
        
        if not isinstance(input_data, dict):
            raise StageException(self.stage_name, "Input must be a dictionary")
        
        if 'evaluated_segments' not in input_data:
            raise StageException(self.stage_name, "Input must contain 'evaluated_segments'")
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate output files in requested formats
        
        Args:
            input_data: Dictionary containing all processing results
            
        Returns:
            Dictionary with output file information
        """
        try:
            evaluated_segments = input_data['evaluated_segments']
            high_value_segments = input_data.get('high_value_segments', [])
            
            output_files = []
            
            # Create base filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"reels_extraction_{timestamp}"
            
            # Export CSV
            if 'csv' in self.export_formats:
                csv_files = self._export_csv(evaluated_segments, high_value_segments, base_name)
                output_files.extend(csv_files)
            
            # Export JSON
            if 'json' in self.export_formats:
                json_files = self._export_json(input_data, base_name)
                output_files.extend(json_files)
            
            # Generate reports
            if 'report' in self.export_formats:
                report_files = self._generate_reports(input_data, base_name)
                output_files.extend(report_files)
            
            # Create export summary
            export_summary = {
                'output_directory': str(self.output_dir),
                'files_created': len(output_files),
                'export_formats': self.export_formats,
                'timestamp': timestamp,
                'total_segments_exported': len(evaluated_segments),
                'high_value_segments_exported': len(high_value_segments)
            }
            
            logger.info(f"Output generation completed: {len(output_files)} files created")
            
            return {
                'output_files': output_files,
                'export_summary': export_summary,
                'output_directory': str(self.output_dir)
            }
            
        except Exception as e:
            raise StageException(self.stage_name, f"Output generation failed: {str(e)}", e)
    
    def _export_csv(self, evaluated_segments: List[Segment], high_value_segments: List[Segment], base_name: str) -> List[str]:
        """Export segments to CSV format"""
        csv_files = []
        
        # Export all evaluated segments
        all_segments_file = self.output_dir / f"{base_name}_all_segments.csv"
        self._segments_to_csv(evaluated_segments, all_segments_file)
        csv_files.append(str(all_segments_file))
        
        # Export high-value segments only
        if high_value_segments:
            high_value_file = self.output_dir / f"{base_name}_high_value.csv"
            self._segments_to_csv(high_value_segments, high_value_file)
            csv_files.append(str(high_value_file))
        
        logger.info(f"Exported {len(csv_files)} CSV files")
        return csv_files
    
    def _segments_to_csv(self, segments: List[Segment], output_path: Path) -> None:
        """Convert segments to CSV format (clean version)"""
        data = []
        for i, segment in enumerate(segments):
            data.append({
                "index": i,
                "start_time": format_time(segment.start_time),
                "end_time": format_time(segment.end_time),
                "duration": f"{segment.duration():.1f}s",
                "text": segment.text,
                "confidence": f"{segment.confidence:.3f}",
                "value_score": f"{segment.value_score:.2f}" if segment.value_score else "",
                "reasoning": segment.reasoning or ""
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.debug(f"CSV exported: {output_path}")
    
    def _export_json(self, input_data: Dict[str, Any], base_name: str) -> List[str]:
        """Export complete results to JSON format"""
        json_files = []
        
        # Create comprehensive results
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'audio_path': input_data.get('audio_path', ''),
                'duration': input_data.get('duration', 0),
                'processing_profile': getattr(self.config, 'processing_profile', 'unknown')
            },
            'segments': {
                'evaluated': [self._segment_to_dict(s) for s in input_data.get('evaluated_segments', [])],
                'high_value': [self._segment_to_dict(s) for s in input_data.get('high_value_segments', [])]
            },
            'summaries': {
                'evaluation': input_data.get('evaluation_summary', {}),
                'segmentation': input_data.get('segmentation_summary', {}),
                'transcription': input_data.get('transcription_summary', {})
            }
        }
        
        # Export complete results
        json_file = self.output_dir / f"{base_name}_complete.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        json_files.append(str(json_file))
        
        logger.info(f"Exported {len(json_files)} JSON files")
        return json_files
    
    def _segment_to_dict(self, segment: Segment) -> Dict[str, Any]:
        """Convert segment to dictionary for JSON export"""
        return {
            'start_time': segment.start_time,
            'end_time': segment.end_time,
            'duration': segment.duration(),
            'text': segment.text,
            'confidence': segment.confidence,
            'value_score': segment.value_score,
            'reasoning': segment.reasoning,
            'technical_terms': getattr(segment, 'technical_terms', [])
        }
    
    def _generate_reports(self, input_data: Dict[str, Any], base_name: str) -> List[str]:
        """Generate human-readable reports"""
        report_files = []
        
        # Summary report
        summary_file = self.output_dir / f"{base_name}_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(self._create_summary_report(input_data))
        report_files.append(str(summary_file))
        
        # Detailed analysis report
        analysis_file = self.output_dir / f"{base_name}_analysis.txt"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(self._create_analysis_report(input_data))
        report_files.append(str(analysis_file))
        
        logger.info(f"Generated {len(report_files)} report files")
        return report_files
    
    def _create_summary_report(self, input_data: Dict[str, Any]) -> str:
        """Create a summary report"""
        lines = []
        lines.append("=== REELS CONTENT EXTRACTION SUMMARY ===")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Configuration and command line
        lines.append("⚙️ CONFIGURATION USED")
        lines.append(f"Profile: {getattr(self.config, 'processing_profile', 'balanced')}")
        lines.append(f"Transcription Model: {getattr(self.config, 'transcription_model', 'auto')} (actual: {getattr(self.config, 'actual_model_used', 'N/A')})")
        lines.append(f"Force Model: {'Yes' if getattr(self.config, 'force_transcription_model', False) else 'No'}")
        lines.append(f"Segment Duration: {getattr(self.config, 'segment_duration', 45)}s")
        lines.append(f"Overlap Duration: {getattr(self.config, 'overlap_duration', 10)}s")
        lines.append(f"Language: {getattr(self.config, 'primary_language', 'he')}")
        lines.append(f"Evaluation Model: {getattr(self.config, 'evaluation_model', 'N/A')}")
        lines.append(f"Min Score Threshold: {getattr(self.config, 'min_score_threshold', 0.7)}")
        
        # Reconstruct command line
        command_parts = ["python -m src"]
        if hasattr(self.config, 'video_path'):
            command_parts.append(getattr(self.config, 'video_path', 'video.mp4'))
        
        # Add key arguments
        if getattr(self.config, 'processing_profile', 'balanced') != 'balanced':
            command_parts.append(f"--profile {self.config.processing_profile}")
        if getattr(self.config, 'transcription_model', 'auto') != 'auto':
            command_parts.append(f"--transcription-model {self.config.transcription_model}")
        if getattr(self.config, 'force_transcription_model', False):
            command_parts.append("--force-model")
        if getattr(self.config, 'segment_duration', 45) != 45:
            command_parts.append(f"--segment-duration {self.config.segment_duration}")
        if getattr(self.config, 'overlap_duration', 10) != 10:
            command_parts.append(f"--overlap-duration {self.config.overlap_duration}")
        if getattr(self.config, 'primary_language', 'he') != 'he':
            command_parts.append(f"--language {self.config.primary_language}")
        
        lines.append("")
        lines.append("🖥️ EQUIVALENT COMMAND LINE")
        lines.append(" ".join(command_parts))
        lines.append("")
        
        # Basic info
        lines.append("📹 INPUT INFORMATION")
        lines.append(f"Audio File: {input_data.get('audio_path', 'N/A')}")
        lines.append(f"Duration: {input_data.get('duration', 0):.1f} seconds")
        lines.append("")
        
        # Processing results
        evaluated = input_data.get('evaluated_segments', [])
        high_value = input_data.get('high_value_segments', [])
        
        lines.append("🎯 PROCESSING RESULTS")
        lines.append(f"Total Segments: {len(evaluated)}")
        lines.append(f"High-Value Segments: {len(high_value)}")
        if evaluated:
            success_rate = (len(high_value) / len(evaluated)) * 100
            lines.append(f"Success Rate: {success_rate:.1f}%")
        lines.append("")
        
        # Quality metrics
        eval_summary = input_data.get('evaluation_summary', {})
        if eval_summary:
            lines.append("📊 QUALITY METRICS")
            lines.append(f"Average Score: {eval_summary.get('average_score', 0):.2f}")
            lines.append(f"Score Range: {eval_summary.get('min_score', 0):.2f} - {eval_summary.get('max_score', 0):.2f}")
            lines.append(f"Evaluation Method: {eval_summary.get('evaluation_method', 'unknown')}")
        
        return "\n".join(lines)
    
    def _create_analysis_report(self, input_data: Dict[str, Any]) -> str:
        """Create detailed analysis report"""
        lines = []
        lines.append("=== DETAILED ANALYSIS REPORT ===")
        lines.append("")
        
        # Score distribution
        eval_summary = input_data.get('evaluation_summary', {})
        if 'score_distribution' in eval_summary:
            lines.append("📈 SCORE DISTRIBUTION")
            for range_name, count in eval_summary['score_distribution'].items():
                lines.append(f"  {range_name}: {count} segments")
            lines.append("")
        
        # Top segments
        high_value = input_data.get('high_value_segments', [])
        if high_value:
            lines.append("🏆 TOP SEGMENTS")
            # Sort by score descending
            top_segments = sorted(high_value, key=lambda s: s.value_score or 0, reverse=True)[:5]
            for i, segment in enumerate(top_segments, 1):
                lines.append(f"{i}. Score: {segment.value_score:.2f} | {format_time(segment.start_time)}-{format_time(segment.end_time)}")
                lines.append(f"   Text: {segment.text[:100]}...")
                lines.append("")
        
        return "\n".join(lines)