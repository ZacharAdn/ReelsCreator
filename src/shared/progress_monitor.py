"""
Advanced progress monitoring with Rich library
"""

import logging
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn, 
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)
from rich.panel import Panel
from rich.text import Text
import psutil

logger = logging.getLogger(__name__)


class ProgressMonitor:
    """Advanced progress monitoring with Rich UI and system metrics"""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.progress = None
        self.current_tasks = {}
        self.stage_start_times = {}
        
    def create_progress_display(self):
        """Create Rich progress display with multiple columns"""
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True
        )
    
    @contextmanager
    def track_stage(self, stage_name: str, total: int = 100, description: str = None):
        """
        Context manager for tracking a processing stage
        
        Args:
            stage_name: Name of the stage
            total: Total number of items to process
            description: Optional detailed description
        """
        if self.progress is None:
            self.progress = self.create_progress_display()
            self.progress.start()
        
        display_name = description or stage_name
        task_id = self.progress.add_task(
            f"[cyan]{display_name}",
            total=total,
            visible=True
        )
        
        self.current_tasks[stage_name] = task_id
        self.stage_start_times[stage_name] = time.time()
        
        # Log stage start with system info
        self._log_stage_start(stage_name, total)
        
        try:
            yield TaskTracker(self, stage_name, task_id)
        finally:
            # Mark task as completed
            self.progress.update(task_id, completed=total)
            elapsed = time.time() - self.stage_start_times[stage_name]
            
            # Log completion
            self._log_stage_completion(stage_name, elapsed)
            
            # Clean up
            del self.current_tasks[stage_name]
            del self.stage_start_times[stage_name]
    
    def update_progress(self, stage_name: str, advance: int = 1, description: str = None):
        """Update progress for a specific stage"""
        if stage_name in self.current_tasks and self.progress:
            task_id = self.current_tasks[stage_name]
            update_kwargs = {"advance": advance}
            
            if description:
                update_kwargs["description"] = f"[cyan]{description}"
            
            self.progress.update(task_id, **update_kwargs)
    
    def set_stage_status(self, stage_name: str, status: str, **kwargs):
        """Set detailed status for a stage"""
        if stage_name in self.current_tasks and self.progress:
            task_id = self.current_tasks[stage_name]
            description = f"[cyan]{stage_name}[/cyan] - [yellow]{status}[/yellow]"
            self.progress.update(task_id, description=description, **kwargs)
    
    def display_system_info(self):
        """Display current system information"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        info_text = Text()
        info_text.append("System Status: ", style="bold")
        info_text.append(f"CPU: {cpu_percent:.1f}% ", style="green" if cpu_percent < 80 else "red")
        info_text.append(f"Memory: {memory.percent:.1f}% ", style="green" if memory.percent < 80 else "red")
        info_text.append(f"({memory.available / 1024**3:.1f}GB free)", style="dim")
        
        panel = Panel(info_text, title="🖥️ System", border_style="blue", padding=(0, 1))
        self.console.print(panel)
    
    def display_stage_summary(self, stage_results: Dict[str, Any]):
        """Display summary of stage results"""
        summary_text = Text()
        
        for key, value in stage_results.items():
            summary_text.append(f"{key}: ", style="bold")
            summary_text.append(f"{value}\n", style="cyan")
        
        panel = Panel(summary_text, title="📊 Stage Summary", border_style="green")
        self.console.print(panel)
    
    def close(self):
        """Close the progress display"""
        if self.progress:
            self.progress.stop()
            self.progress = None
    
    def _log_stage_start(self, stage_name: str, total: int):
        """Log stage start with system information"""
        memory = psutil.virtual_memory()
        logger.info(f"🚀 Starting stage: {stage_name}")
        logger.info(f"   Total items: {total}")
        logger.info(f"   Memory usage: {memory.percent:.1f}% ({memory.available / 1024**3:.1f}GB available)")
    
    def _log_stage_completion(self, stage_name: str, elapsed_time: float):
        """Log stage completion with timing information"""
        logger.info(f"✅ Completed stage: {stage_name}")
        logger.info(f"   Elapsed time: {elapsed_time:.2f} seconds")
        logger.info(f"   Rate: {elapsed_time:.3f}s per item" if elapsed_time > 0 else "   Very fast completion")


class TaskTracker:
    """Helper class for tracking individual tasks within a stage"""
    
    def __init__(self, monitor: ProgressMonitor, stage_name: str, task_id: int):
        self.monitor = monitor
        self.stage_name = stage_name
        self.task_id = task_id
        self._current_item = 0
    
    def update(self, advance: int = 1, description: str = None, **kwargs):
        """Update task progress"""
        self._current_item += advance
        self.monitor.update_progress(self.stage_name, advance, description)
        
        # Add additional metadata to progress if provided
        if kwargs and self.monitor.progress:
            self.monitor.progress.update(self.task_id, **kwargs)
    
    def set_status(self, status: str, **kwargs):
        """Set current status of the task"""
        self.monitor.set_stage_status(self.stage_name, status, **kwargs)
    
    def set_postfix(self, **kwargs):
        """Set postfix information (like current score, etc.)"""
        if self.monitor.progress:
            # Create postfix string from kwargs
            postfix_items = [f"{k}={v}" for k, v in kwargs.items()]
            postfix = " | ".join(postfix_items)
            
            # Update description with postfix
            description = f"[cyan]{self.stage_name}[/cyan] [{postfix}]"
            self.monitor.progress.update(self.task_id, description=description)


# Global instance for reuse
_progress_monitor = None

def get_progress_monitor() -> ProgressMonitor:
    """Get the global progress monitor instance"""
    global _progress_monitor
    if _progress_monitor is None:
        _progress_monitor = ProgressMonitor()
    return _progress_monitor

@contextmanager
def track_stage(stage_name: str, total: int = 100, description: str = None):
    """Convenience function for tracking stages"""
    monitor = get_progress_monitor()
    with monitor.track_stage(stage_name, total, description) as tracker:
        yield tracker