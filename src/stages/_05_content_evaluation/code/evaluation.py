"""
Content evaluation module using open-source LLM
"""

import logging
import json
import random
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.models import Segment
from shared.llm_manager import get_llm_manager
from shared.progress_monitor import get_progress_monitor

logger = logging.getLogger(__name__)


class ContentEvaluator:
    """Handles content evaluation using open-source LLM or rule-based scoring"""
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct", batch_size: int = 5, 
                 use_rule_based: bool = False, enable_evaluation: bool = True):
        """
        Initialize content evaluator
        
        Args:
            model_name: Open-source LLM model name
            batch_size: Number of segments to process in parallel
            use_rule_based: Use rule-based scoring instead of LLM
            enable_evaluation: Enable evaluation (False for draft mode)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_rule_based = use_rule_based
        self.enable_evaluation = enable_evaluation
        self.tokenizer = None
        self.model = None
        
        if not enable_evaluation:
            logger.info("Content evaluation disabled for maximum speed")
        elif use_rule_based:
            logger.info("Using rule-based scoring for fast evaluation")
        else:
            logger.info(f"Initializing open-source LLM: {model_name} with batch_size={batch_size}")
    
    def load_model(self):
        """Load open-source LLM model using LLMManager with timeout support"""
        if self.model is None:
            llm_manager = get_llm_manager()
            llm_manager.log_system_info()
            
            try:
                self.tokenizer, self.model = llm_manager.load_model(self.model_name)
                logger.info(f"✅ LLM model loaded successfully: {self.model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to load LLM model: {e}")
                raise
    
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
        Evaluate multiple segments using batch processing, rule-based, or no evaluation
        
        Args:
            segments: List of segments to evaluate
            
        Returns:
            List of segments with evaluation results
        """
        if not segments:
            return segments
        
        # Handle no evaluation mode (draft profile) - use dynamic scoring for variance
        if not self.enable_evaluation:
            logger.info(f"⚡ Using fast dynamic scoring for {len(segments)} segments (evaluation disabled)")
            return self._evaluate_segments_fast_dynamic(segments)
        
        # Handle rule-based evaluation (fast profile)
        if self.use_rule_based:
            logger.info(f"🚀 Using rule-based evaluation for {len(segments)} segments")
            return self._evaluate_segments_rule_based(segments)
        
        # Handle LLM evaluation (balanced/quality profiles)
        logger.info(f"🧠 Evaluating {len(segments)} segments using LLM batch processing (batch_size={self.batch_size})")
        
        # Process segments in batches with Rich progress monitoring
        progress_monitor = get_progress_monitor()
        total_batches = (len(segments) + self.batch_size - 1) // self.batch_size
        
        with progress_monitor.track_stage("LLM Evaluation", len(segments), "Evaluating segments with AI") as tracker:
            for batch_start in range(0, len(segments), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(segments))
                batch_segments = segments[batch_start:batch_end]
                
                # Update progress description
                batch_num = batch_start // self.batch_size + 1
                tracker.set_status(f"Processing batch {batch_num}/{total_batches}")
                
                # Evaluate batch
                batch_results = self.evaluate_batch(batch_segments)
                
                # Apply results to segments
                for segment, result in zip(batch_segments, batch_results):
                    segment.value_score = result.get("score", 0.0)
                    segment.reasoning = result.get("reasoning", "")
                    tracker.update(1)
                    tracker.set_postfix(score=f"{segment.value_score:.2f}")
        
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
    
    def _evaluate_segments_rule_based(self, segments: List[Segment]) -> List[Segment]:
        """
        Fast rule-based evaluation without LLM inference
        
        Args:
            segments: List of segments to evaluate
            
        Returns:
            List of segments with rule-based scores
        """
        progress_monitor = get_progress_monitor()
        
        with progress_monitor.track_stage("Rule-based Evaluation", len(segments), "Fast heuristic scoring") as tracker:
            for segment in segments:
                result = self._rule_based_score(segment)
                segment.value_score = result["score"]
                segment.reasoning = result["reasoning"]
                tracker.update(1)
                tracker.set_postfix(score=f"{segment.value_score:.2f}")
        
        logger.info(f"✅ Completed rule-based evaluation of {len(segments)} segments")
        return segments
    
    def _evaluate_segments_fast_dynamic(self, segments: List[Segment]) -> List[Segment]:
        """
        Fast dynamic scoring to create score variance without full evaluation
        Fixes the 0.75 fixed score bug by using lightweight heuristics
        
        Args:
            segments: List of segments to evaluate
            
        Returns:
            List of segments with dynamic scores
        """
        progress_monitor = get_progress_monitor()
        
        with progress_monitor.track_stage("Fast Dynamic Scoring", len(segments), "Quick variance scoring") as tracker:
            for segment in segments:
                # Generate dynamic score based on content characteristics
                score = self._fast_dynamic_score(segment)
                segment.value_score = score
                segment.reasoning = "Fast dynamic scoring for speed"
                tracker.update(1)
                tracker.set_postfix(score=f"{segment.value_score:.2f}")
        
        logger.info(f"✅ Completed fast dynamic scoring of {len(segments)} segments")
        return segments
    
    def _fast_dynamic_score(self, segment: Segment) -> float:
        """
        Generate a dynamic score based on lightweight content analysis
        Ensures score variance without expensive LLM inference
        
        Args:
            segment: Segment to score
            
        Returns:
            Dynamic score between 0.3 and 0.9
        """
        text = segment.text.strip().lower()
        
        # Base score with slight randomization for variance
        base_score = 0.6 + (random.random() * 0.2 - 0.1)  # 0.5 to 0.7 range
        
        # Length-based adjustment
        word_count = len(text.split())
        if 20 <= word_count <= 80:  # Optimal length
            base_score += 0.1
        elif word_count < 10 or word_count > 120:  # Too short or long
            base_score -= 0.15
        
        # Simple keyword detection
        valuable_keywords = ['example', 'show', 'demonstrate', 'important', 'notice', 'see']
        tech_keywords = ['function', 'data', 'code', 'variable', 'method', 'algorithm']
        
        keyword_boost = 0
        for keyword in valuable_keywords:
            if keyword in text:
                keyword_boost += 0.05
        
        for keyword in tech_keywords:
            if keyword in text:
                keyword_boost += 0.03
        
        # Cap keyword boost
        keyword_boost = min(0.2, keyword_boost)
        
        # Confidence adjustment
        if segment.confidence:
            if segment.confidence > 0.9:
                base_score += 0.05
            elif segment.confidence < 0.7:
                base_score -= 0.1
        
        # Add slight position-based variance (segments in middle tend to be better)
        position_factor = 0.02 * np.sin(len(text) / 50)  # Creates gentle wave
        
        final_score = base_score + keyword_boost + position_factor
        
        # Ensure score is in reasonable range with good variance
        return max(0.3, min(0.9, final_score))
    
    def _rule_based_score(self, segment: Segment) -> Dict[str, Any]:
        """
        Calculate rule-based score for a segment based on various heuristics
        Enhanced to create better score distribution and variance
        
        Args:
            segment: Segment to score
            
        Returns:
            Dictionary with score and reasoning
        """
        text = segment.text.strip()
        # More dynamic base score for better distribution
        base_variation = random.random() * 0.3 + 0.3  # 0.3 to 0.6 range
        score = base_variation
        reasoning_parts = []
        
        # Length-based scoring (optimized for Reels 15-45s segments)
        word_count = len(text.split())
        if 20 <= word_count <= 100:  # Good length for Reels content  
            score += 0.25
            reasoning_parts.append("Optimal length for Reels")
        elif 10 <= word_count <= 19:  # Decent length
            score += 0.15
            reasoning_parts.append("Good length")
        elif word_count < 5:  # Too short
            score -= 0.3
            reasoning_parts.append("Too short")
        elif word_count > 150:  # Too long
            score -= 0.2
            reasoning_parts.append("Too long for short-form")
        
        # Educational content indicators (enhanced)
        educational_keywords = [
            "example", "demonstrate", "show", "explain", "understand", "learn",
            "concept", "important", "key", "remember", "notice", "see", "look",
            "step", "process", "method", "technique", "approach", "solution"
        ]
        
        technical_keywords = [
            "function", "method", "variable", "data", "code", "programming",
            "algorithm", "syntax", "error", "debug", "output", "input",
            "class", "object", "array", "loop", "condition", "parameter"
        ]
        
        engagement_keywords = [
            "amazing", "cool", "interesting", "powerful", "simple", "easy",
            "quick", "fast", "effective", "useful", "practical", "tip", "trick"
        ]
        
        text_lower = text.lower()
        
        # Educational content scoring (more nuanced)
        edu_matches = sum(1 for keyword in educational_keywords if keyword in text_lower)
        tech_matches = sum(1 for keyword in technical_keywords if keyword in text_lower)
        engagement_matches = sum(1 for keyword in engagement_keywords if keyword in text_lower)
        
        if edu_matches >= 3:
            score += 0.2
            reasoning_parts.append("Rich educational content")
        elif edu_matches >= 2:
            score += 0.15
            reasoning_parts.append("Strong educational content")
        elif edu_matches >= 1:
            score += 0.1
            reasoning_parts.append("Educational content")
        
        if tech_matches >= 3:
            score += 0.15
            reasoning_parts.append("Strong technical content")
        elif tech_matches >= 1:
            score += 0.1
            reasoning_parts.append("Technical content")
        
        # Engagement scoring
        if engagement_matches >= 2:
            score += 0.15
            reasoning_parts.append("Highly engaging language")
        elif engagement_matches >= 1:
            score += 0.1
            reasoning_parts.append("Engaging language")
        
        # Question indicators (engagement)
        question_count = text.count("?")
        question_words = sum(1 for q in ["what", "how", "why", "when", "where"] if q in text_lower)
        
        if question_count >= 2 or question_words >= 2:
            score += 0.15
            reasoning_parts.append("Multiple engaging questions")
        elif question_count >= 1 or question_words >= 1:
            score += 0.1
            reasoning_parts.append("Engaging questions")
        
        # Practical demonstration indicators
        practical_keywords = ["result", "output", "works", "example", "demo", "practice", "try", "test", "run"]
        practical_matches = sum(1 for keyword in practical_keywords if keyword in text_lower)
        
        if practical_matches >= 2:
            score += 0.15
            reasoning_parts.append("Strong practical demonstration")
        elif practical_matches >= 1:
            score += 0.1
            reasoning_parts.append("Practical demonstration")
        
        # Confidence-based adjustment (more impact)
        if segment.confidence and segment.confidence < 0.6:
            score -= 0.2
            reasoning_parts.append("Very low confidence")
        elif segment.confidence and segment.confidence < 0.8:
            score -= 0.1
            reasoning_parts.append("Low confidence")
        elif segment.confidence and segment.confidence > 0.9:
            score += 0.1
            reasoning_parts.append("High confidence")
        
        # Clamp score to valid range
        score = max(0.0, min(1.0, score))
        
        # Create reasoning
        if reasoning_parts:
            reasoning = "Rule-based: " + ", ".join(reasoning_parts)
        else:
            reasoning = "Rule-based: Standard content"
        
        return {"score": score, "reasoning": reasoning}


class MultiCriteriaEvaluator:
    """Advanced multi-criteria evaluation for higher quality scoring"""
    
    def __init__(self, enable_embeddings: bool = False):
        self.enable_embeddings = enable_embeddings
        self.weights = {
            "clarity": 0.25,
            "interest": 0.25, 
            "educational_value": 0.25,
            "technical_content": 0.15,
            "engagement": 0.10
        }
    
    def evaluate_segment(self, segment: Segment, context_segments: List[Segment] = None) -> Dict[str, Any]:
        """
        Multi-criteria evaluation of a segment
        
        Args:
            segment: Segment to evaluate
            context_segments: Other segments for relative scoring
            
        Returns:
            Detailed evaluation results
        """
        text = segment.text.strip()
        
        # Individual criteria scores
        clarity_score = self._evaluate_clarity(text)
        interest_score = self._evaluate_interest(text)
        educational_score = self._evaluate_educational_value(text)
        technical_score = self._evaluate_technical_content(text)
        engagement_score = self._evaluate_engagement(text)
        
        # Calculate weighted score
        weighted_score = (
            clarity_score * self.weights["clarity"] +
            interest_score * self.weights["interest"] + 
            educational_score * self.weights["educational_value"] +
            technical_score * self.weights["technical_content"] +
            engagement_score * self.weights["engagement"]
        )
        
        # Apply relative scoring if context is available
        if context_segments:
            weighted_score = self._apply_relative_scoring(weighted_score, segment, context_segments)
        
        # Apply confidence adjustment
        if segment.confidence:
            confidence_factor = max(0.8, min(1.2, segment.confidence + 0.1))
            weighted_score *= confidence_factor
        
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, weighted_score))
        
        # Create detailed reasoning
        reasoning = self._create_detailed_reasoning(
            clarity_score, interest_score, educational_score, 
            technical_score, engagement_score, final_score
        )
        
        return {
            "score": final_score,
            "reasoning": reasoning,
            "criteria_scores": {
                "clarity": clarity_score,
                "interest": interest_score,
                "educational_value": educational_score,
                "technical_content": technical_score,
                "engagement": engagement_score
            }
        }
    
    def _evaluate_clarity(self, text: str) -> float:
        """Evaluate text clarity and understandability"""
        text_lower = text.lower()
        score = 0.5
        
        # Clear indicators
        clear_words = ["clear", "simple", "easy", "understand", "obvious", "notice"]
        unclear_words = ["confusing", "unclear", "complicated", "hard", "difficult"]
        
        for word in clear_words:
            if word in text_lower:
                score += 0.1
        
        for word in unclear_words:
            if word in text_lower:
                score -= 0.15
        
        # Sentence structure analysis
        sentences = text.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        if 8 <= avg_sentence_length <= 20:  # Optimal sentence length
            score += 0.15
        elif avg_sentence_length > 30:  # Too long
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_interest(self, text: str) -> float:
        """Evaluate content interest and engagement potential"""
        text_lower = text.lower()
        score = 0.4
        
        # Interest indicators
        interest_words = ["amazing", "interesting", "cool", "wow", "look", "see", "check"]
        question_words = ["what", "how", "why", "when", "where"]
        
        for word in interest_words:
            if word in text_lower:
                score += 0.12
        
        # Questions boost engagement
        question_count = text.count("?")
        question_word_count = sum(1 for word in question_words if word in text_lower)
        
        score += min(0.3, (question_count + question_word_count) * 0.1)
        
        # Exclamations indicate enthusiasm
        exclamation_count = text.count("!")
        score += min(0.15, exclamation_count * 0.05)
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_educational_value(self, text: str) -> float:
        """Evaluate educational content value"""
        text_lower = text.lower()
        score = 0.3
        
        # Educational keywords
        edu_words = [
            "learn", "teach", "explain", "understand", "concept", "important",
            "remember", "key", "principle", "theory", "practice", "example",
            "demonstrate", "show", "illustrate", "method", "technique", "approach"
        ]
        
        for word in edu_words:
            if word in text_lower:
                score += 0.08
        
        # Learning objectives
        objective_phrases = ["learn how to", "understand", "we will", "let's see", "notice that"]
        for phrase in objective_phrases:
            if phrase in text_lower:
                score += 0.15
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_technical_content(self, text: str) -> float:
        """Evaluate technical content depth"""
        text_lower = text.lower()
        score = 0.2
        
        # Technical terms
        tech_words = [
            "function", "method", "variable", "data", "algorithm", "code",
            "programming", "syntax", "parameter", "argument", "class", "object",
            "array", "list", "loop", "condition", "import", "library", "module"
        ]
        
        tech_count = sum(1 for word in tech_words if word in text_lower)
        score += min(0.6, tech_count * 0.1)
        
        # Code-like patterns (simple detection)
        if any(pattern in text for pattern in ["()", "[]", "==", "!="]):
            score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_engagement(self, text: str) -> float:
        """Evaluate engagement and social media readiness"""
        text_lower = text.lower()
        score = 0.4
        
        # Engagement words
        engaging_words = ["tip", "trick", "hack", "secret", "quick", "fast", "powerful"]
        
        for word in engaging_words:
            if word in text_lower:
                score += 0.15
        
        # Call to action
        action_words = ["try", "test", "practice", "use", "apply", "implement"]
        for word in action_words:
            if word in text_lower:
                score += 0.1
        
        # Length optimization for social media
        word_count = len(text.split())
        if 15 <= word_count <= 60:  # Perfect for reels
            score += 0.2
        elif 60 < word_count <= 100:  # Still good
            score += 0.1
        elif word_count > 150:  # Too long
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _apply_relative_scoring(self, base_score: float, segment: Segment, context_segments: List[Segment]) -> float:
        """Apply relative scoring within the video context"""
        if not context_segments or len(context_segments) < 3:
            return base_score
        
        # Calculate relative position (beginning, middle, end segments often differ in quality)
        segment_index = context_segments.index(segment) if segment in context_segments else 0
        relative_position = segment_index / len(context_segments)
        
        # Middle segments often have better content
        position_bonus = 0.1 * np.sin(relative_position * np.pi)
        
        return base_score + position_bonus
    
    def _create_detailed_reasoning(self, clarity: float, interest: float, educational: float, 
                                 technical: float, engagement: float, final_score: float) -> str:
        """Create human-readable reasoning for the score"""
        reasoning_parts = []
        
        if clarity > 0.7:
            reasoning_parts.append("excellent clarity")
        elif clarity > 0.5:
            reasoning_parts.append("good clarity")
        else:
            reasoning_parts.append("needs clearer explanation")
        
        if interest > 0.7:
            reasoning_parts.append("highly engaging")
        elif interest > 0.5:
            reasoning_parts.append("moderately interesting")
        
        if educational > 0.7:
            reasoning_parts.append("strong educational value")
        elif educational > 0.5:
            reasoning_parts.append("good learning content")
        
        if technical > 0.6:
            reasoning_parts.append("rich technical content")
        elif technical > 0.3:
            reasoning_parts.append("some technical depth")
        
        if engagement > 0.6:
            reasoning_parts.append("social media ready")
        
        base_reasoning = "Multi-criteria: " + ", ".join(reasoning_parts)
        
        # Add score interpretation
        if final_score >= 0.8:
            base_reasoning += " (excellent segment)"
        elif final_score >= 0.6:
            base_reasoning += " (good segment)"
        elif final_score >= 0.4:
            base_reasoning += " (average segment)"
        else:
            base_reasoning += " (needs improvement)"
        
        return base_reasoning 