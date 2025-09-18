"""
Embeddings module using sentence-transformers
"""

import logging
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from shared.models import Segment

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Handles embedding generation for segments"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model_name = model_name
        self.model = None
        logger.info(f"Initializing embedding model: {model_name}")
    
    def load_model(self):
        """Load sentence transformer model"""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing (to reduce memory usage)
            
        Returns:
            List of embedding vectors
        """
        self.load_model()
        
        if not texts:
            return []
        
        logger.info(f"Generating embeddings for {len(texts)} texts with batch_size={batch_size}")
        
        try:
            embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_tensor=False)
            
            # Convert to list of lists
            embedding_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
            
            logger.info(f"Generated {len(embedding_list)} embeddings")
            return embedding_list
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def add_embeddings_to_segments(self, segments: List[Segment], batch_size: int = 32) -> List[Segment]:
        """
        Add embeddings to segments
        
        Args:
            segments: List of segments
            batch_size: Batch size for processing
            
        Returns:
            List of segments with embeddings added
        """
        if not segments:
            return segments
        
        # Extract texts
        texts = [segment.text for segment in segments]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts, batch_size=batch_size)
        
        # Add embeddings to segments
        for segment, embedding in zip(segments, embeddings):
            segment.embedding = embedding
        
        logger.info(f"Added embeddings to {len(segments)} segments")
        return segments
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Similarity score between 0 and 1
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_similar_segments(self, segments: List[Segment], threshold: float = 0.7) -> List[List[int]]:
        """
        Find groups of similar segments
        
        Args:
            segments: List of segments with embeddings
            threshold: Similarity threshold
            
        Returns:
            List of segment group indices
        """
        if not segments or len(segments) < 2:
            return []
        
        # Check if segments have embeddings
        segments_with_embeddings = [seg for seg in segments if seg.embedding is not None]
        
        if len(segments_with_embeddings) < 2:
            logger.warning("Not enough segments with embeddings for similarity analysis")
            return []
        
        similar_groups = []
        used_indices = set()
        
        for i, segment1 in enumerate(segments_with_embeddings):
            if i in used_indices:
                continue
            
            group = [i]
            used_indices.add(i)
            
            for j, segment2 in enumerate(segments_with_embeddings[i+1:], i+1):
                if j in used_indices:
                    continue
                
                similarity = self.calculate_similarity(segment1.embedding, segment2.embedding)
                
                if similarity >= threshold:
                    group.append(j)
                    used_indices.add(j)
            
            if len(group) > 1:  # Only include groups with multiple segments
                similar_groups.append(group)
        
        logger.info(f"Found {len(similar_groups)} groups of similar segments")
        return similar_groups 