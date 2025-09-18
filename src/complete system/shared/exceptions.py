"""
Custom exceptions for the pipeline
"""


class PipelineException(Exception):
    """Base exception for all pipeline errors"""
    pass


class StageException(PipelineException):
    """Exception raised by individual stages"""
    
    def __init__(self, stage_name: str, message: str, original_exception=None):
        self.stage_name = stage_name
        self.original_exception = original_exception
        super().__init__(f"Stage '{stage_name}': {message}")


class ValidationException(StageException):
    """Exception raised during input validation"""
    pass


class ConfigurationException(PipelineException):
    """Exception raised for configuration issues"""
    pass


class ResourceException(PipelineException):
    """Exception raised for resource issues (files, memory, etc.)"""
    pass