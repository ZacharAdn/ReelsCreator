"""
Shared utilities for the pipeline
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional


def format_time(seconds: float) -> str:
    """
    Convert seconds to MM:SS format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"


def ensure_directory_exists(path: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in MB
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    return Path(file_path).stat().st_size / (1024 * 1024)


def setup_logging(level=logging.INFO) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Logging level
        
    Returns:
        Configured logger
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def calculate_processing_speed(duration: float, processing_time: float) -> float:
    """
    Calculate processing speed as realtime factor
    
    Args:
        duration: Content duration in seconds
        processing_time: Processing time in seconds
        
    Returns:
        Realtime factor (>1 means faster than realtime)
    """
    if processing_time <= 0:
        return float('inf')
    return duration / processing_time