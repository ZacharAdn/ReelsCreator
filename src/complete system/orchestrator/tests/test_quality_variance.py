"""
Quality tests for segment score and length variance
"""

import pytest
import numpy as np
from typing import List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.shared.models import Segment
from src.stages.content_evaluation.code.evaluation import ContentEvaluator, MultiCriteriaEvaluator
from src.stages.content_segmentation.code.segmentation import SegmentProcessor


class TestSegmentQualityVariance:
    """Test segment quality variance to ensure proper distribution"""
    
    def create_test_segments(self, count: int = 10) -> List[Segment]:
        """Create test segments with varied content"""
        segments = []
        
        # Mix of different content types to ensure variance
        content_samples = [
            "This is a simple example of basic programming concepts.",
            "Amazing! Look at this cool function that demonstrates advanced algorithms.",
            "Let me explain how this works. First, we need to understand the theory.",
            "Quick tip: Use this powerful technique for better performance.",
            "The result shows interesting patterns in the data analysis.",
            "Now let's see what happens when we run this code example.",
            "Important: Remember this key concept for your implementation.",
            "Here's another method that works differently but achieves similar results.",
            "Notice how the output changes when we modify these parameters.",
            "This demonstration illustrates the practical application of the theory."
        ]
        
        for i in range(count):
            segment = Segment(
                start_time=i * 45.0,
                end_time=(i + 1) * 45.0,
                text=content_samples[i % len(content_samples)],
                confidence=0.8 + (i % 3) * 0.1  # Vary confidence 0.8, 0.9, 1.0
            )
            segments.append(segment)
        
        return segments
    
    def test_score_variance_rule_based(self):
        """Test that rule-based evaluation produces score variance"""
        evaluator = ContentEvaluator(use_rule_based=True, enable_evaluation=True)
        segments = self.create_test_segments(15)
        
        # Evaluate segments
        evaluated_segments = evaluator.evaluate_segments(segments)
        scores = [seg.value_score for seg in evaluated_segments if seg.value_score is not None]
        
        # Check that we have scores
        assert len(scores) > 0, "No scores were generated"
        
        # Calculate variance
        score_variance = np.var(scores)
        score_std = np.std(scores)
        unique_scores = len(set(scores))
        
        print(f"Score statistics:")
        print(f"  Mean: {np.mean(scores):.3f}")
        print(f"  Std Dev: {score_std:.3f}")
        print(f"  Variance: {score_variance:.3f}")
        print(f"  Range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"  Unique scores: {unique_scores}/{len(scores)}")
        
        # Assertions for variance
        assert score_variance > 0.01, f"Score variance too low: {score_variance:.4f}"
        assert score_std > 0.05, f"Score standard deviation too low: {score_std:.4f}"
        assert unique_scores >= len(scores) * 0.5, f"Too few unique scores: {unique_scores}/{len(scores)}"
        
        # Check score range
        assert min(scores) >= 0.0, "Scores below 0.0"
        assert max(scores) <= 1.0, "Scores above 1.0"
        assert max(scores) - min(scores) > 0.2, "Score range too narrow"
    
    def test_score_variance_fast_dynamic(self):
        """Test that fast dynamic scoring produces variance"""
        evaluator = ContentEvaluator(enable_evaluation=False)  # This triggers fast dynamic scoring
        segments = self.create_test_segments(20)
        
        # Evaluate segments
        evaluated_segments = evaluator.evaluate_segments(segments)
        scores = [seg.value_score for seg in evaluated_segments if seg.value_score is not None]
        
        # Check variance
        score_variance = np.var(scores)
        unique_scores = len(set(scores))
        
        print(f"Fast dynamic scoring statistics:")
        print(f"  Variance: {score_variance:.3f}")
        print(f"  Unique scores: {unique_scores}/{len(scores)}")
        print(f"  Range: {min(scores):.3f} - {max(scores):.3f}")
        
        # Should NOT be all 0.75 (the old bug)
        assert not all(score == 0.75 for score in scores), "All scores are 0.75 - bug not fixed!"
        assert score_variance > 0.005, f"Fast dynamic variance too low: {score_variance:.4f}"
        assert unique_scores >= 3, f"Too few unique scores in fast mode: {unique_scores}"
    
    def test_multi_criteria_evaluation(self):
        """Test multi-criteria evaluation produces detailed variance"""
        multi_evaluator = MultiCriteriaEvaluator()
        segments = self.create_test_segments(10)
        
        results = []
        for segment in segments:
            result = multi_evaluator.evaluate_segment(segment)
            results.append(result)
        
        # Extract scores and criteria
        overall_scores = [r["score"] for r in results]
        criteria_scores = {
            criterion: [r["criteria_scores"][criterion] for r in results]
            for criterion in ["clarity", "interest", "educational_value", "technical_content", "engagement"]
        }
        
        # Test overall score variance
        overall_variance = np.var(overall_scores)
        assert overall_variance > 0.02, f"Multi-criteria overall variance too low: {overall_variance:.4f}"
        
        # Test individual criteria variance
        for criterion, scores in criteria_scores.items():
            criterion_variance = np.var(scores)
            print(f"{criterion} variance: {criterion_variance:.3f}")
            assert criterion_variance > 0.01, f"{criterion} variance too low: {criterion_variance:.4f}"
    
    def test_segment_length_variance(self):
        """Test that segments have length variance"""
        processor = SegmentProcessor(segment_duration=45, overlap_duration=10)
        
        # Create segments with different natural lengths
        base_segments = []
        texts = [
            "Short text.",
            "This is a medium length text with more words and content to process.",
            "This is a very long text segment that contains a lot of information and detailed explanations about complex topics that would naturally create longer segments with more comprehensive content coverage.",
            "Brief.",
            "Another moderate length segment with reasonable content.",
        ]
        
        for i, text in enumerate(texts):
            segment = Segment(
                start_time=i * 30.0,
                end_time=(i + 1) * 30.0 + len(text.split()) * 0.5,  # Variable length based on content
                text=text,
                confidence=0.9
            )
            base_segments.append(segment)
        
        # Process segments
        processed_segments = processor.create_overlapping_segments(base_segments)
        
        # Check length variance
        durations = [seg.duration() for seg in processed_segments]
        text_lengths = [len(seg.text.split()) for seg in processed_segments]
        
        duration_variance = np.var(durations)
        text_length_variance = np.var(text_lengths)
        
        print(f"Segment length statistics:")
        print(f"  Duration variance: {duration_variance:.3f}")
        print(f"  Text length variance: {text_length_variance:.3f}")
        print(f"  Duration range: {min(durations):.1f}s - {max(durations):.1f}s")
        print(f"  Text length range: {min(text_lengths)} - {max(text_lengths)} words")
        
        # Basic variance checks
        assert len(set(durations)) > 1, "All segments have identical duration"
        assert len(set(text_lengths)) > 1, "All segments have identical text length"
    
    def test_no_uniform_segments(self):
        """Test that we don't get the old uniformity bug (all segments identical)"""
        evaluator = ContentEvaluator(use_rule_based=True, enable_evaluation=True)
        segments = self.create_test_segments(12)
        
        # Process segments
        evaluated_segments = evaluator.evaluate_segments(segments)
        
        # Extract all properties
        scores = [seg.value_score for seg in evaluated_segments]
        durations = [seg.duration() for seg in evaluated_segments]
        text_lengths = [len(seg.text) for seg in evaluated_segments]
        
        # Check for uniformity (the old bug)
        unique_scores = len(set(scores))
        unique_durations = len(set(durations))
        unique_text_lengths = len(set(text_lengths))
        
        print(f"Uniformity check:")
        print(f"  Unique scores: {unique_scores}/{len(scores)}")
        print(f"  Unique durations: {unique_durations}/{len(durations)}")
        print(f"  Unique text lengths: {unique_text_lengths}/{len(text_lengths)}")
        
        # Fail if everything is too uniform
        assert unique_scores > 1, "All segments have identical scores - uniformity bug detected!"
        assert not (unique_scores == 1 and unique_durations == 1), "Segments are completely uniform!"
        
        # Check that we don't have the specific 0.75 bug
        if len(set(scores)) == 1:
            score_value = scores[0]
            assert score_value != 0.75, f"All segments scored 0.75 - old bug detected!"


class TestPerformanceRegression:
    """Test for performance regressions"""
    
    def test_evaluation_performance(self):
        """Test that evaluation doesn't hang or take too long"""
        import time
        
        evaluator = ContentEvaluator(use_rule_based=True, enable_evaluation=True)
        segments = []
        
        # Create larger segment set
        for i in range(50):
            segment = Segment(
                start_time=i * 30.0,
                end_time=(i + 1) * 30.0,
                text=f"Test segment {i} with some educational content about programming concepts.",
                confidence=0.8
            )
            segments.append(segment)
        
        # Time the evaluation
        start_time = time.time()
        evaluated_segments = evaluator.evaluate_segments(segments)
        elapsed = time.time() - start_time
        
        print(f"Evaluation performance:")
        print(f"  Segments: {len(segments)}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Rate: {len(segments)/elapsed:.1f} segments/s")
        
        # Performance assertions
        assert elapsed < 30.0, f"Evaluation too slow: {elapsed:.2f}s for {len(segments)} segments"
        assert len(evaluated_segments) == len(segments), "Lost segments during evaluation"
        assert all(seg.value_score is not None for seg in evaluated_segments), "Missing scores"
    
    def test_memory_usage(self):
        """Test that evaluation doesn't consume excessive memory"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        evaluator = ContentEvaluator(use_rule_based=True, enable_evaluation=True)
        segments = []
        
        # Create many segments
        for i in range(100):
            segment = Segment(
                start_time=i * 30.0,
                end_time=(i + 1) * 30.0,
                text=f"Test segment {i} with educational content about data science and machine learning.",
                confidence=0.8
            )
            segments.append(segment)
        
        # Evaluate
        evaluated_segments = evaluator.evaluate_segments(segments)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        
        # Memory assertions (reasonable limits)
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase:.1f} MB increase"
        assert len(evaluated_segments) == len(segments), "Lost segments during evaluation"


if __name__ == "__main__":
    # Run tests directly for development
    test_instance = TestSegmentQualityVariance()
    print("Testing score variance...")
    test_instance.test_score_variance_rule_based()
    test_instance.test_score_variance_fast_dynamic()
    test_instance.test_multi_criteria_evaluation()
    test_instance.test_no_uniform_segments()
    
    print("\nTesting performance...")
    perf_instance = TestPerformanceRegression()
    perf_instance.test_evaluation_performance()
    perf_instance.test_memory_usage()
    
    print("\n✅ All quality tests passed!")