"""
Video processing module for audio extraction
"""

import logging
import tempfile
import os
from pathlib import Path
from typing import Optional
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy import VideoFileClip

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video processing and audio extraction"""
    
    def __init__(self):
        """Initialize video processor"""
        logger.info("Initializing Video Processor")
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to video file
            output_path: Optional path for extracted audio (if None, creates temp file)
            
        Returns:
            Path to extracted audio file
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Generate output path if not provided
        if output_path is None:
            video_name = Path(video_path).stem
            output_path = tempfile.mktemp(suffix=f"_{video_name}.wav")
        
        logger.info(f"Extracting audio from: {video_path}")
        
        try:
            # Load video file
            video = VideoFileClip(video_path)
            
            # Extract audio
            audio = video.audio
            
            if audio is None:
                raise ValueError(f"No audio track found in video: {video_path}")
            
            # Save audio
            audio.write_audiofile(output_path)
            
            # Clean up
            video.close()
            audio.close()
            
            logger.info(f"Audio extracted to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            raise
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video file information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            video = VideoFileClip(video_path)
            
            info = {
                "duration": video.duration,
                "fps": video.fps,
                "size": video.size,
                "has_audio": video.audio is not None,
                "audio_fps": video.audio.fps if video.audio else None
            }
            
            video.close()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise
    
    def process_video_file(self, video_path: str, keep_audio: bool = False) -> str:
        """
        Complete video processing pipeline
        
        Args:
            video_path: Path to video file
            keep_audio: Whether to keep the extracted audio file
            
        Returns:
            Path to extracted audio file
        """
        # Extract audio
        audio_path = self.extract_audio(video_path)
        
        # Clean up audio file if not keeping it
        if not keep_audio:
            # Schedule cleanup when process ends
            import atexit
            atexit.register(lambda: self._cleanup_file(audio_path))
        
        return audio_path
    
    def _cleanup_file(self, file_path: str) -> None:
        """Clean up temporary file"""
        try:
            if Path(file_path).exists():
                os.remove(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {e}")
    
    def supported_formats(self) -> list:
        """Get list of supported video formats"""
        return ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm']
    
    def is_supported_format(self, file_path: str) -> bool:
        """
        Check if video format is supported
        
        Args:
            file_path: Path to video file
            
        Returns:
            True if format is supported
        """
        return Path(file_path).suffix.lower() in self.supported_formats() 