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
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct", batch_size: int = 5):
        """
        Initialize content evaluator
        
        Args:
            model_name: Open-source LLM model name
            batch_size: Number of segments to process in parallel
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.tokenizer = None
        self.model = None
        logger.info(f"Initializing open-source LLM: {model_name} with batch_size={batch_size}")
    
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

            # Add padding token if not present - use different token to avoid confusion
            if self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                else:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    self.model.resize_token_embeddings(len(self.tokenizer))

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
            # Tokenize input WITH attention mask
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True,
                padding=True,
                return_attention_mask=True
            )
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response WITH attention mask
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=100,
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
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
    
    def evaluate_batch(self, segments: List[Segment]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple segments in a single batch for better performance
        
        Args:
            segments: List of segments to evaluate (max batch_size)
            
        Returns:
            List of evaluation results
        """
        if not segments:
            return []
        
        self.load_model()
        
        # Create batch prompt
        batch_prompt = self._create_batch_evaluation_prompt([seg.text for seg in segments])
        
        try:
            # Tokenize batch input
            inputs = self.tokenizer(
                batch_prompt, 
                return_tensors="pt", 
                max_length=1024,  # Increased for batch processing
                truncation=True,
                padding=True,
                return_attention_mask=True
            )
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate batch response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=200,  # Increased for batch responses
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode batch response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response_text[len(batch_prompt):].strip()
            
            # Parse batch results
            return self._parse_batch_response(generated_text, len(segments))
            
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            # Fallback to individual evaluation
            return [self.evaluate_segment(seg) for seg in segments]
    
    def _create_batch_evaluation_prompt(self, texts: List[str]) -> str:
        """Create a batch evaluation prompt for multiple segments"""
        prompt = """
Please evaluate the following educational content segments for their value as short-form social media content.

For each segment, evaluate based on these criteria:
1. Is there a clear insight or practical demonstration?
2. Is the content clear and understandable?
3. Is there significant educational value?
4. Would this work well as a short video clip?

Respond with a JSON array where each element contains:
- "score": float between 0.0 and 1.0 (1.0 = excellent, 0.0 = poor)
- "reasoning": brief explanation of the score

Segments to evaluate:
"""
        
        for i, text in enumerate(texts, 1):
            prompt += f"\n{i}. \"{text}\"\n"
        
        prompt += "\nJSON Response (array format):\n"
        return prompt
    
    def _parse_batch_response(self, response_text: str, expected_count: int) -> List[Dict[str, Any]]:
        """Parse batch evaluation response"""
        try:
            # Try to extract JSON array
            import re
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                import json
                results = json.loads(json_match.group(0))
                if isinstance(results, list) and len(results) == expected_count:
                    return results
        except Exception as e:
            logger.warning(f"Failed to parse batch response: {e}")
        
        # Fallback: create default results
        return [{"score": 0.5, "reasoning": "Batch parsing failed"} for _ in range(expected_count)]
    
    def evaluate_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Evaluate multiple segments using batch processing for better performance
        
        Args:
            segments: List of segments to evaluate
            
        Returns:
            List of segments with evaluation results
        """
        if not segments:
            return segments
        
        logger.info(f"Evaluating {len(segments)} segments using batch processing (batch_size={self.batch_size})")
        
        # Process segments in batches
        from tqdm import tqdm
        total_batches = (len(segments) + self.batch_size - 1) // self.batch_size
        progress_bar = tqdm(total=len(segments), desc="Evaluating segments", unit="segment")
        
        for batch_start in range(0, len(segments), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(segments))
            batch_segments = segments[batch_start:batch_end]
            
            # Update progress description
            progress_bar.set_description(f"Evaluating batch {batch_start//self.batch_size + 1}/{total_batches}")
            
            # Evaluate batch
            batch_results = self.evaluate_batch(batch_segments)
            
            # Apply results to segments
            for segment, result in zip(batch_segments, batch_results):
                segment.value_score = result.get("score", 0.0)
                segment.reasoning = result.get("reasoning", "")
                progress_bar.update(1)
                progress_bar.set_postfix(score=f"{segment.value_score:.2f}")
        
        progress_bar.close()
        logger.info(f"✅ Completed evaluation of {len(segments)} segments in {total_batches} batches")
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