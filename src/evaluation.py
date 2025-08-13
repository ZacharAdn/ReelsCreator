"""
Content evaluation module using open-source LLM
"""

import logging
import json
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from .models import Segment

logger = logging.getLogger(__name__)


class ContentEvaluator:
    """Handles content evaluation using open-source LLM"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        """
        Initialize content evaluator
        
        Args:
            model_name: Open-source LLM model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        logger.info(f"Initializing open-source LLM: {model_name}")
    
    def load_model(self):
        """Load open-source LLM model"""
        if self.model is None:
            logger.info(f"Loading LLM model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )

            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.eval()
            logger.info("LLM model loaded successfully")
    
    def evaluate_segment(self, segment: Segment) -> Dict[str, Any]:
        """
        Evaluate a single segment
        
        Args:
            segment: Segment to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        self.load_model()
        
        prompt = self._create_evaluation_prompt(segment.text)
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (after the prompt)
            generated_text = response_text[len(prompt):].strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(generated_text)
                return result
            except json.JSONDecodeError:
                # Attempt to extract a trailing JSON block
                import re
                match = re.search(r"\{[\s\S]*\}$", generated_text)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except Exception:
                        pass
                # Fallback: extract score from text
                return self._parse_text_response(generated_text)
                
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {"score": 0.5, "reasoning": f"Evaluation error: {str(e)}"}
    
    def _create_evaluation_prompt(self, text: str) -> str:
        """
        Create evaluation prompt for LLM
        
        Args:
            text: Text to evaluate
            
        Returns:
            Formatted prompt
        """
        return f"""
Please evaluate the following educational content segment for its value as short-form social media content.

Content: "{text}"

Evaluate based on these criteria:
1. Is there a clear insight or practical demonstration?
2. Is the content clear and understandable?
3. Is there significant educational value?
4. Would this work well as a short video clip?

Respond with a JSON object containing:
- "score": float between 0.0 and 1.0 (1.0 = excellent, 0.0 = poor)
- "reasoning": brief explanation of the score

Example response:
{{"score": 0.8, "reasoning": "Clear explanation of pandas concepts with practical value"}}
"""
    
    def _parse_text_response(self, text: str) -> Dict[str, Any]:
        """
        Parse text response when JSON parsing fails
        
        Args:
            text: Response text
            
        Returns:
            Parsed result
        """
        # Simple fallback parsing
        score = 0.5  # Default score
        reasoning = text
        
        # Try to extract score from text
        if "score" in text.lower():
            try:
                # Look for numbers in the text
                import re
                numbers = re.findall(r'\d+\.?\d*', text)
                if numbers:
                    score = min(1.0, max(0.0, float(numbers[0])))
            except:
                pass
        
        return {"score": score, "reasoning": reasoning}
    
    def evaluate_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Evaluate multiple segments
        
        Args:
            segments: List of segments to evaluate
            
        Returns:
            List of segments with evaluation results
        """
        if not segments:
            return segments
        
        logger.info(f"Evaluating {len(segments)} segments")
        
        # Add progress bar for segment evaluation
        from tqdm import tqdm
        progress_bar = tqdm(segments, desc="Evaluating segments", unit="segment")
        
        for i, segment in enumerate(progress_bar):
            progress_bar.set_description(f"Evaluating segment {i+1}/{len(segments)}")
            
            evaluation = self.evaluate_segment(segment)
            
            segment.value_score = evaluation.get("score", 0.0)
            segment.reasoning = evaluation.get("reasoning", "")
            
            progress_bar.set_postfix(score=f"{segment.value_score:.2f}")
        
        progress_bar.close()
        return segments
    
    def filter_high_value_segments(self, segments: List[Segment], threshold: float = 0.7) -> List[Segment]:
        """
        Filter segments by value score
        
        Args:
            segments: List of segments
            threshold: Minimum score threshold
            
        Returns:
            List of high-value segments
        """
        high_value_segments = [
            seg for seg in segments 
            if seg.value_score and seg.value_score >= threshold
        ]
        
        # Sort by score (highest first)
        high_value_segments.sort(key=lambda x: x.value_score or 0, reverse=True)
        
        logger.info(f"Filtered {len(segments)} segments to {len(high_value_segments)} high-value segments (threshold={threshold})")
        return high_value_segments
    
    def get_evaluation_summary(self, segments: List[Segment]) -> Dict[str, Any]:
        """
        Get summary of evaluation results
        
        Args:
            segments: List of evaluated segments
            
        Returns:
            Summary statistics
        """
        if not segments:
            return {
                "total_segments": 0,
                "average_score": 0.0,
                "score_distribution": {},
                "high_value_count": 0
            }
        
        scores = [seg.value_score for seg in segments if seg.value_score is not None]
        
        if not scores:
            return {
                "total_segments": len(segments),
                "average_score": 0.0,
                "score_distribution": {},
                "high_value_count": 0
            }
        
        # Score distribution
        score_ranges = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        for score in scores:
            if score < 0.2:
                score_ranges["0.0-0.2"] += 1
            elif score < 0.4:
                score_ranges["0.2-0.4"] += 1
            elif score < 0.6:
                score_ranges["0.4-0.6"] += 1
            elif score < 0.8:
                score_ranges["0.6-0.8"] += 1
            else:
                score_ranges["0.8-1.0"] += 1
        
        return {
            "total_segments": len(segments),
            "average_score": sum(scores) / len(scores),
            "score_distribution": score_ranges,
            "high_value_count": len([s for s in scores if s >= 0.7])
        } 