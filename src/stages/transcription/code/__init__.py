"""
Transcription Stage - Convert audio to text using Whisper
"""

import logging
from typing import Dict, Any, List, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.base_stage import BaseStage
from shared.exceptions import StageException
from shared.models import Segment
from .transcription import WhisperTranscriber
from .language_processor import LanguageProcessor

logger = logging.getLogger(__name__)


class TranscriptionStage(BaseStage):
    """
    Stage 3: Audio transcription using Whisper
    
    Input: {
        'audio_path': str,
        'duration': float,
        'speaker_segments': List[Dict] (optional)
    }
    Output: {
        'transcribed_segments': List[Segment],
        'total_segments': int,
        'transcription_summary': Dict
    }
    """
    
    def __init__(self, config):
        super().__init__(config, "Transcription")
        
        # Initialize Whisper transcriber
        self.transcriber = WhisperTranscriber(
            model_name=getattr(config, 'whisper_model', 'base'),
            primary_language=getattr(config, 'primary_language', 'he')
        )
        
        # Initialize language processor (optional)
        self.language_processor = None
        if getattr(config, 'preserve_technical_terms', False):
            self.language_processor = LanguageProcessor(
                primary_language=getattr(config, 'primary_language', 'he'),
                technical_language=getattr(config, 'technical_language', 'en')
            )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input contains audio path"""
        super().validate_input(input_data)
        
        if not isinstance(input_data, dict):
            raise StageException(self.stage_name, "Input must be a dictionary")
        
        if 'audio_path' not in input_data:
            raise StageException(self.stage_name, "Input must contain 'audio_path'")
        
        audio_path = input_data['audio_path']
        if not audio_path or not isinstance(audio_path, str):
            raise StageException(self.stage_name, "audio_path must be a valid string")
        
        if not Path(audio_path).exists():
            raise StageException(self.stage_name, f"Audio file not found: {audio_path}")
        
        return True
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform audio transcription
        
        Args:
            input_data: Dictionary containing audio_path and metadata
            
        Returns:
            Dictionary with transcribed segments
        """
        try:
            audio_path = input_data['audio_path']
            
            # Perform transcription
            logger.info(f"Transcribing audio: {audio_path}")
            transcribed_segments = self.transcriber.process_audio_file(audio_path)
            
            if not transcribed_segments:
                raise StageException(self.stage_name, "No segments were transcribed")
            
            # Process technical terms if enabled
            technical_terms_count = 0
            if self.language_processor:
                logger.info("Processing technical terms...")
                for segment in transcribed_segments:
                    technical_terms = self.language_processor.extract_technical_terms(segment.text)
                    if technical_terms:
                        technical_terms_count += len(technical_terms)
                        segment.technical_terms = technical_terms
            
            # Create summary
            transcription_summary = {
                'total_segments': len(transcribed_segments),
                'total_duration': input_data.get('duration', 0),
                'average_confidence': sum(s.confidence for s in transcribed_segments) / len(transcribed_segments),
                'technical_terms_found': technical_terms_count,
                'whisper_model_used': self.transcriber.model_name,
                'primary_language': self.transcriber.primary_language
            }
            
            logger.info(f"Transcription completed: {len(transcribed_segments)} segments")
            
            return {
                'transcribed_segments': transcribed_segments,
                'total_segments': len(transcribed_segments),
                'transcription_summary': transcription_summary,
                'audio_path': audio_path,  # Pass through
                'duration': input_data.get('duration', 0),
                # Pass through speaker data if available
                'speaker_segments': input_data.get('speaker_segments', [])
            }
            
        except Exception as e:
            raise StageException(self.stage_name, f"Transcription failed: {str(e)}", e)