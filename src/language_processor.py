"""
Language processing module for multilingual content and technical term preservation
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException as LangDetectError

logger = logging.getLogger(__name__)


class LanguageProcessor:
    """Handles multilingual processing and technical term preservation"""
    
    def __init__(self, primary_language: str = "he", technical_language: str = "en"):
        """
        Initialize language processor
        
        Args:
            primary_language: Primary language code (Hebrew)
            technical_language: Technical terms language (English)
        """
        self.primary_language = primary_language
        self.technical_language = technical_language
        self.technical_terms = self._load_technical_vocabulary()
        logger.info(f"Language processor initialized: {primary_language} + {technical_language}")
    
    def _load_technical_vocabulary(self) -> Dict[str, List[str]]:
        """Load data science technical vocabulary"""
        # Data science terms that should be preserved
        technical_vocab = {
            "machine_learning": [
                "machine learning", "deep learning", "neural network", "algorithm",
                "regression", "classification", "clustering", "supervised", "unsupervised",
                "overfitting", "underfitting", "cross-validation", "feature", "dataset",
                "training", "testing", "validation", "model", "prediction", "accuracy"
            ],
            "data_science": [
                "data science", "data analysis", "big data", "analytics", "visualization",
                "pandas", "numpy", "matplotlib", "seaborn", "jupyter", "notebook",
                "dataframe", "series", "array", "vector", "matrix", "statistics"
            ],
            "programming": [
                "python", "code", "function", "variable", "loop", "condition", "class",
                "object", "method", "import", "library", "package", "API", "framework",
                "debugging", "syntax", "parameter", "argument", "return", "print"
            ],
            "statistics": [
                "mean", "median", "mode", "standard deviation", "variance", "correlation",
                "distribution", "probability", "hypothesis", "test", "p-value", "confidence",
                "interval", "significance", "sample", "population", "bias"
            ]
        }
        
        # Flatten into single list
        all_terms = []
        for category_terms in technical_vocab.values():
            all_terms.extend(category_terms)
        
        logger.info(f"Loaded {len(all_terms)} technical terms")
        return {"all": all_terms, **technical_vocab}
    
    def detect_language(self, text: str) -> str:
        """
        Detect primary language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (he/en/mixed)
        """
        if not text.strip():
            return self.primary_language
        
        try:
            # Remove common English technical terms for better Hebrew detection
            text_for_detection = text
            for term in self.technical_terms["all"][:20]:  # Check top terms only
                text_for_detection = re.sub(r'\b' + re.escape(term) + r'\b', '', 
                                          text_for_detection, flags=re.IGNORECASE)
            
            detected = detect(text_for_detection.strip())
            
            # Check if it's mixed language
            english_words = len(re.findall(r'[a-zA-Z]+', text))
            hebrew_words = len(re.findall(r'[\u0590-\u05FF]+', text))
            
            if english_words > 0 and hebrew_words > 0:
                if english_words / (english_words + hebrew_words) > 0.3:
                    return "mixed"
            
            return detected if detected in ["he", "en"] else self.primary_language
            
        except LangDetectError:
            return self.primary_language
    
    def identify_technical_terms(self, text: str) -> List[str]:
        """
        Identify technical terms in text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of found technical terms
        """
        found_terms = []
        text_lower = text.lower()
        
        for term in self.technical_terms["all"]:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            if re.search(pattern, text_lower):
                found_terms.append(term)
        
        return found_terms
    
    def calculate_technical_density(self, text: str) -> float:
        """
        Calculate density of technical terms in text
        
        Args:
            text: Text to analyze
            
        Returns:
            Technical term density (0-1)
        """
        if not text.strip():
            return 0.0
        
        words = text.split()
        if not words:
            return 0.0
        
        technical_terms = self.identify_technical_terms(text)
        technical_word_count = sum(len(term.split()) for term in technical_terms)
        
        return min(1.0, technical_word_count / len(words))
    
    def enhance_transcription(self, text: str) -> Dict[str, any]:
        """
        Enhance transcription with language analysis
        
        Args:
            text: Original transcription text
            
        Returns:
            Enhanced transcription data
        """
        language = self.detect_language(text)
        technical_terms = self.identify_technical_terms(text)
        technical_density = self.calculate_technical_density(text)
        
        # Quality scoring based on language characteristics
        language_score = 1.0
        
        if language == "mixed":
            # Mixed language is good for educational content
            language_score = 0.9
        elif language == self.primary_language:
            # Pure Hebrew is good
            language_score = 0.8
        elif language == self.technical_language:
            # Pure English might be less suitable for Hebrew audience
            language_score = 0.6
        
        # Boost score for technical content
        technical_boost = min(0.2, technical_density * 0.5)
        final_score = min(1.0, language_score + technical_boost)
        
        return {
            "enhanced_text": text,  # Could add term highlighting here
            "primary_language": language,
            "technical_terms": technical_terms,
            "technical_density": technical_density,
            "language_score": final_score,
            "is_mixed_language": language == "mixed",
            "has_technical_content": len(technical_terms) > 0
        }
    
    def extract_technical_terms(self, text: str) -> List[str]:
        """
        Extract technical terms from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of technical terms found
        """
        if not text:
            return []
        
        # Normalize text for matching
        text_lower = text.lower()
        found_terms = []
        
        # Check for technical terms in our vocabulary
        for term in self.technical_terms["all"]:
            # Look for exact matches (case-insensitive)
            if term.lower() in text_lower:
                found_terms.append(term)
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in found_terms:
            if term.lower() not in seen:
                unique_terms.append(term)
                seen.add(term.lower())
        
        return unique_terms
    
    def process_segment(self, segment) -> None:
        """
        Process a segment with language analysis
        
        Args:
            segment: Segment object to enhance
        """
        enhancement = self.enhance_transcription(segment.text)
        
        # Add language metadata to segment
        segment.primary_language = enhancement["primary_language"]
        segment.technical_terms = enhancement["technical_terms"]
        
        # Boost value score for technical content
        if segment.value_score is not None:
            language_boost = (enhancement["language_score"] - 0.5) * 0.2
            segment.value_score = min(1.0, segment.value_score + language_boost)
    
    def filter_by_language_quality(self, segments: List, min_language_score: float = 0.6) -> List:
        """
        Filter segments by language quality
        
        Args:
            segments: List of segments to filter
            min_language_score: Minimum language quality score
            
        Returns:
            Filtered segments
        """
        filtered = []
        
        for segment in segments:
            enhancement = self.enhance_transcription(segment.text)
            
            if enhancement["language_score"] >= min_language_score:
                self.process_segment(segment)
                filtered.append(segment)
        
        logger.info(f"Language filtering: {len(segments)} -> {len(filtered)} segments")
        return filtered
