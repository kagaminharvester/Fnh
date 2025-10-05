"""
Enhanced Profiling and Metrics System
Author: FunGen AI System
Version: 1.0.0

Provides detailed performance profiling with GPU timing support:
- Per-stage timing
- GPU utilization and VRAM tracking
- Quality metrics
- JSON export
"""

import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager


@dataclass
class StageMetrics:
    """Metrics for a processing stage."""
    name: str
    total_ms: float = 0.0
    count: int = 0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    
    @property
    def avg_ms(self) -> float:
        """Average time per operation."""
        return self.total_ms / self.count if self.count > 0 else 0.0
        
    def add_sample(self, duration_ms: float):
        """Add a timing sample."""
        self.total_ms += duration_ms
        self.count += 1
        self.min_ms = min(self.min_ms, duration_ms)
        self.max_ms = max(self.max_ms, duration_ms)


@dataclass
class GPUMetrics:
    """GPU utilization and memory metrics."""
    available: bool = False
    util_avg: float = 0.0
    mem_peak_mb: float = 0.0
    mem_allocated_mb: float = 0.0
    mem_reserved_mb: float = 0.0
    device_name: str = ""


@dataclass
class QualityMetrics:
    """Quality metrics for output."""
    total_actions: int = 0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    smoothness: float = 0.0
    range_used: float = 0.0


@dataclass
class ProfileSession:
    """Main profiling session container."""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    # Stage timings
    stages: Dict[str, StageMetrics] = field(default_factory=dict)
    
    # GPU metrics
    gpu: GPUMetrics = field(default_factory=GPUMetrics)
    
    # Quality metrics
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    
    # Additional metrics
    throughput_fps: float = 0.0
    total_frames: int = 0
    video_duration_s: float = 0.0
    
    # Feature-specific metrics
    tracking_metrics: Dict[str, Any] = field(default_factory=dict)
    projection_metrics: Dict[str, Any] = field(default_factory=dict)
    filter_optimizer_metrics: Dict[str, Any] = field(default_factory=dict)
    cache_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize CUDA if available."""
        self._init_gpu()
        
    def _init_gpu(self):
        """Initialize GPU monitoring."""
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu.available = True
                self.gpu.device_name = torch.cuda.get_device_name(0)
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
            
    def start_stage(self, stage_name: str) -> float:
        """
        Start timing a stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Start timestamp
        """
        return time.time()
        
    def end_stage(self, stage_name: str, start_time: float):
        """
        End timing a stage.
        
        Args:
            stage_name: Name of the stage
            start_time: Start timestamp from start_stage()
        """
        duration_ms = (time.time() - start_time) * 1000.0
        
        if stage_name not in self.stages:
            self.stages[stage_name] = StageMetrics(name=stage_name)
            
        self.stages[stage_name].add_sample(duration_ms)
        
    @contextmanager
    def stage(self, stage_name: str):
        """
        Context manager for timing a stage.
        
        Usage:
            with session.stage("inference"):
                # do work
                pass
        """
        start = self.start_stage(stage_name)
        try:
            yield
        finally:
            self.end_stage(stage_name, start)
            
    def record_gpu_interval(self, label: str):
        """Record GPU event timing (CUDA events)."""
        if not self.gpu.available:
            return
            
        try:
            import torch
            
            # Record current GPU memory
            if torch.cuda.is_available():
                self.gpu.mem_allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                self.gpu.mem_reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)
                self.gpu.mem_peak_mb = max(
                    self.gpu.mem_peak_mb,
                    torch.cuda.max_memory_allocated() / (1024 * 1024)
                )
        except Exception:
            pass
            
    def update_gpu_metrics(self):
        """Update GPU metrics."""
        if not self.gpu.available:
            return
            
        try:
            import torch
            
            if torch.cuda.is_available():
                self.gpu.mem_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                
                # Try to get GPU utilization via pynvml
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.gpu.util_avg = util.gpu
                    pynvml.nvmlShutdown()
                except ImportError:
                    pass
        except Exception:
            pass
            
    def finalize(self, total_frames: int, duration_s: float):
        """
        Finalize profiling session.
        
        Args:
            total_frames: Total number of frames processed
            duration_s: Total duration in seconds
        """
        self.end_time = time.time()
        self.total_frames = total_frames
        self.video_duration_s = duration_s
        
        # Calculate throughput
        if duration_s > 0:
            self.throughput_fps = total_frames / duration_s
            
        # Update GPU metrics one final time
        self.update_gpu_metrics()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        result = {
            'timestamp': self.start_time,
            'duration_s': (self.end_time or time.time()) - self.start_time,
            'throughput_fps': self.throughput_fps,
            'total_frames': self.total_frames,
            'video_duration_s': self.video_duration_s,
        }
        
        # Add stage metrics
        result['stages'] = {}
        for name, metrics in self.stages.items():
            result['stages'][name] = {
                'avg_ms': metrics.avg_ms,
                'min_ms': metrics.min_ms,
                'max_ms': metrics.max_ms,
                'count': metrics.count,
                'total_ms': metrics.total_ms
            }
            
        # Add GPU metrics
        if self.gpu.available:
            result['gpu'] = {
                'device': self.gpu.device_name,
                'util_avg': self.gpu.util_avg,
                'mem_peak_mb': self.gpu.mem_peak_mb,
                'mem_allocated_mb': self.gpu.mem_allocated_mb,
                'mem_reserved_mb': self.gpu.mem_reserved_mb
            }
            
        # Add quality metrics
        if self.quality.total_actions > 0:
            result['quality'] = asdict(self.quality)
            
        # Add feature metrics
        if self.tracking_metrics:
            result['tracking'] = self.tracking_metrics
        if self.projection_metrics:
            result['projection'] = self.projection_metrics
        if self.filter_optimizer_metrics:
            result['filter_optimizer'] = self.filter_optimizer_metrics
        if self.cache_metrics:
            result['cache'] = self.cache_metrics
            
        return result
        
    def save(self, output_path: Path, logger: Optional[logging.Logger] = None):
        """
        Save profile to JSON file.
        
        Args:
            output_path: Path to output JSON file
            logger: Optional logger
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
                
            if logger:
                logger.info(f"Profile saved to {output_path}")
        except Exception as e:
            if logger:
                logger.error(f"Failed to save profile: {e}")
                
    def print_summary(self, logger: Optional[logging.Logger] = None):
        """Print profiling summary."""
        logger = logger or logging.getLogger(__name__)
        
        logger.info("=" * 60)
        logger.info("PROFILING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Throughput: {self.throughput_fps:.1f} FPS")
        logger.info(f"Total frames: {self.total_frames}")
        logger.info(f"Duration: {self.video_duration_s:.1f}s")
        
        if self.stages:
            logger.info("\nStage Timings:")
            for name, metrics in sorted(self.stages.items()):
                logger.info(f"  {name}: {metrics.avg_ms:.2f}ms avg ({metrics.count} samples)")
                
        if self.gpu.available:
            logger.info(f"\nGPU: {self.gpu.device_name}")
            logger.info(f"  Peak VRAM: {self.gpu.mem_peak_mb:.1f} MB")
            if self.gpu.util_avg > 0:
                logger.info(f"  Avg Utilization: {self.gpu.util_avg:.1f}%")
                
        logger.info("=" * 60)


class ProfilerManager:
    """Manages profiling sessions."""
    
    def __init__(self, enabled: bool = False, logger: Optional[logging.Logger] = None):
        """
        Initialize profiler manager.
        
        Args:
            enabled: Enable profiling
            logger: Optional logger
        """
        self.enabled = enabled
        self.logger = logger or logging.getLogger(__name__)
        self.current_session: Optional[ProfileSession] = None
        
    def start_session(self) -> Optional[ProfileSession]:
        """Start a new profiling session."""
        if not self.enabled:
            return None
            
        self.current_session = ProfileSession()
        self.logger.info("Profiling session started")
        return self.current_session
        
    def end_session(self, output_path: Optional[Path] = None):
        """End current profiling session."""
        if not self.enabled or self.current_session is None:
            return
            
        self.current_session.print_summary(self.logger)
        
        if output_path:
            self.current_session.save(output_path, self.logger)
            
        self.current_session = None
        
    def get_session(self) -> Optional[ProfileSession]:
        """Get current profiling session."""
        return self.current_session if self.enabled else None
