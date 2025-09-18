"""
Content Evaluation Stage - Score segments for Reels quality
"""

import logging
from typing import Dict, Any, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.base_stage import BaseStage
from shared.exceptions import StageException
from shared.models import Segment
from .evaluation import ContentEvaluator
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class ContentEvaluationStage(BaseStage):
    """
    Stage 5: Evaluate content quality and generate embeddings
    
    Input: {
        'reels_segments': List[Segment],
        ...
    }
    Output: {
        'evaluated_segments': List[Segment],
        'high_value_segments': List[Segment],
        'evaluation_summary': Dict
    }
    """
    
    def __init__(self, config):
        super().__init__(config, "ContentEvaluation")
        
        # Initialize content evaluator
        self.evaluator = ContentEvaluator(
            model_name=getattr(config, 'evaluation_model', 'microsoft/Phi-3-mini-4k-instruct'),
            batch_size=getattr(config, 'evaluation_batch_size', 5),
            use_rule_based=getattr(config, 'use_rule_based_scoring', True),
            enable_evaluation=getattr(config, 'enable_content_evaluation', True)
        )
        
        # Initialize embedding generator (optional)
        self.embedding_generator = None
        if not getattr(config, 'minimal_mode', False):
            self.embedding_generator = EmbeddingGenerator(
                model_name=getattr(config, 'embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            )
        
        self.min_score_threshold = getattr(config, 'min_score_threshold', 0.7)
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input contains segments to evaluate"""
        super().validate_input(input_data)
        
        if not isinstance(input_data, dict):
            raise StageException(self.stage_name, "Input must be a dictionary")
        
        if 'reels_segments' not in input_data:
            raise StageException(self.stage_name, "Input must contain 'reels_segments'")
        
        segments = input_data['reels_segments']
        if not isinstance(segments, list):
            raise StageException(self.stage_name, "reels_segments must be a list")
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate content quality and generate embeddings
        
        Args:
            input_data: Dictionary containing Reels segments
            
        Returns:
            Dictionary with evaluated segments
        """
        try:
            reels_segments = input_data['reels_segments']
            
            if not reels_segments:
                logger.warning("No segments to evaluate")
                return {
                    'evaluated_segments': [],
                    'high_value_segments': [],
                    'evaluation_summary': {'total_segments': 0, 'message': 'No segments to evaluate'}
                }
            
            logger.info(f"Evaluating {len(reels_segments)} segments")
            
            # Generate embeddings if not in minimal mode
            segments_with_embeddings = reels_segments
            if self.embedding_generator:
                logger.info("Generating embeddings...")
                segments_with_embeddings = self.embedding_generator.add_embeddings_to_segments(
                    reels_segments,
                    batch_size=getattr(self.config, 'embedding_batch_size', 8)
                )
            
            # Evaluate content
            logger.info("Evaluating content quality...")
            evaluated_segments = self.evaluator.evaluate_segments(segments_with_embeddings)
            
            # Filter high-value segments
            high_value_segments = self.evaluator.filter_high_value_segments(
                evaluated_segments,
                threshold=self.min_score_threshold
            )
            
            # Create evaluation summary
            evaluation_summary = self._create_evaluation_summary(evaluated_segments, high_value_segments)
            
            logger.info(f"Evaluation completed: {len(high_value_segments)}/{len(evaluated_segments)} high-value segments")
            
            return {
                'evaluated_segments': evaluated_segments,
                'high_value_segments': high_value_segments,
                'evaluation_summary': evaluation_summary,
                # Pass through previous data
                'reels_segments': reels_segments,
                'transcribed_segments': input_data.get('transcribed_segments', []),
                'audio_path': input_data.get('audio_path'),
                'duration': input_data.get('duration', 0)
            }
            
        except Exception as e:
            raise StageException(self.stage_name, f"Content evaluation failed: {str(e)}", e)
    
    def _create_evaluation_summary(self, evaluated_segments: List[Segment], high_value_segments: List[Segment]) -> Dict[str, Any]:
        """
        Create detailed evaluation summary
        
        Args:
            evaluated_segments: All evaluated segments
            high_value_segments: High-value segments only
            
        Returns:
            Evaluation summary dictionary
        """
        if not evaluated_segments:
            return {'total_segments': 0, 'message': 'No segments evaluated'}
        
        scores = [s.value_score for s in evaluated_segments if s.value_score is not None]
        
        # Score distribution
        score_ranges = {
            '0.0-0.2': 0, '0.2-0.4': 0, '0.4-0.6': 0, 
            '0.6-0.8': 0, '0.8-1.0': 0
        }
        
        for score in scores:
            if score < 0.2:
                score_ranges['0.0-0.2'] += 1
            elif score < 0.4:
                score_ranges['0.2-0.4'] += 1
            elif score < 0.6:
                score_ranges['0.4-0.6'] += 1
            elif score < 0.8:
                score_ranges['0.6-0.8'] += 1
            else:
                score_ranges['0.8-1.0'] += 1
        
        return {
            'total_segments': len(evaluated_segments),
            'high_value_segments': len(high_value_segments),
            'high_value_percentage': (len(high_value_segments) / len(evaluated_segments)) * 100,
            'average_score': sum(scores) / len(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'score_distribution': score_ranges,
            'threshold_used': self.min_score_threshold,
            'evaluation_method': 'rule_based' if self.evaluator.use_rule_based else 'llm_based',
            'embeddings_generated': self.embedding_generator is not None
        }