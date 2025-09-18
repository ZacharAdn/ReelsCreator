"""
Base stage class for all pipeline stages
"""

import logging
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseStage(ABC):
    """
    Base class for all pipeline stages
    
    Provides common functionality for:
    - Performance monitoring
    - Input validation
    - Error handling
    - Logging
    """
    
    def __init__(self, config: Any, stage_name: str):
        """
        Initialize stage
        
        Args:
            config: Configuration object
            stage_name: Name of this stage for logging
        """
        self.config = config
        self.stage_name = stage_name
        self.performance_metrics = {}
        self.logger = logging.getLogger(f"{__name__}.{stage_name}")
        
        self.logger.info(f"Initialized {stage_name} stage")
    
    @abstractmethod
    def execute(self, input_data: Any) -> Any:
        """
        Execute the stage logic
        
        Args:
            input_data: Input data for this stage
            
        Returns:
            Output data from this stage
        """
        raise NotImplementedError(f"Stage {self.stage_name} must implement execute()")
    
    def validate_input(self, input_data: Any) -> bool:
        """
        Validate input data for this stage
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, raises exception if not
        """
        if input_data is None:
            raise ValueError(f"Stage {self.stage_name} received None input")
        return True
    
    def run(self, input_data: Any) -> Any:
        """
        Run the stage with performance monitoring and error handling
        
        Args:
            input_data: Input data for this stage
            
        Returns:
            Output data from this stage
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting {self.stage_name} stage")
            
            # Validate input
            self.validate_input(input_data)
            
            # Execute stage
            result = self.execute(input_data)
            
            # Record performance
            execution_time = time.time() - start_time
            self.performance_metrics = {
                'execution_time': execution_time,
                'stage_name': self.stage_name,
                'success': True,
                'timestamp': time.time()
            }
            
            self.logger.info(f"Completed {self.stage_name} stage in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.performance_metrics = {
                'execution_time': execution_time,
                'stage_name': self.stage_name,
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }
            
            self.logger.error(f"Stage {self.stage_name} failed after {execution_time:.2f}s: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this stage
        
        Returns:
            Dictionary containing performance metrics
        """
        return self.performance_metrics.copy()
    
    def get_stage_info(self) -> Dict[str, Any]:
        """
        Get information about this stage
        
        Returns:
            Dictionary with stage information
        """
        return {
            'stage_name': self.stage_name,
            'config': str(self.config),
            'metrics': self.get_metrics()
        }