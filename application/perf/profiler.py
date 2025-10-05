"""
Performance profiling infrastructure for FunGen.

Provides timing instrumentation for various processing stages.
"""
import time
import threading
from typing import Dict, Optional, List
from dataclasses import dataclass, field
import json


@dataclass
class TimingRecord:
    """Record of a single timing measurement."""
    label: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    def finish(self):
        """Mark the timing record as finished."""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time


class PerfSession:
    """
    Performance profiling session manager.
    
    Tracks timing information for various labeled operations.
    Thread-safe for concurrent access.
    """
    
    _instance: Optional['PerfSession'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._timings: Dict[str, List[TimingRecord]] = {}
        self._active_timings: Dict[str, TimingRecord] = {}
        self._session_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> 'PerfSession':
        """Get or create the global profiling session."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the global profiling session."""
        with cls._lock:
            cls._instance = cls()
    
    @classmethod
    def start(cls, label: str):
        """
        Start timing for the given label.
        
        Args:
            label: Identifier for the operation being timed
        """
        instance = cls.get_instance()
        with instance._session_lock:
            record = TimingRecord(label=label, start_time=time.time())
            instance._active_timings[label] = record
    
    @classmethod
    def stop(cls, label: str):
        """
        Stop timing for the given label.
        
        Args:
            label: Identifier for the operation that finished
        """
        instance = cls.get_instance()
        with instance._session_lock:
            if label in instance._active_timings:
                record = instance._active_timings.pop(label)
                record.finish()
                
                if label not in instance._timings:
                    instance._timings[label] = []
                instance._timings[label].append(record)
    
    @classmethod
    def summary_dict(cls) -> Dict[str, any]:
        """
        Get summary of all timing measurements.
        
        Returns:
            Dictionary with timing statistics per label
        """
        instance = cls.get_instance()
        summary = {}
        
        with instance._session_lock:
            for label, records in instance._timings.items():
                durations = [r.duration for r in records if r.duration is not None]
                
                if durations:
                    summary[label] = {
                        "count": len(durations),
                        "total_seconds": sum(durations),
                        "avg_seconds": sum(durations) / len(durations),
                        "min_seconds": min(durations),
                        "max_seconds": max(durations),
                    }
        
        return summary
    
    @classmethod
    def export_json(cls, output_path: str):
        """
        Export timing summary to a JSON file.
        
        Args:
            output_path: Path where JSON file will be written
        """
        summary = cls.summary_dict()
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)


class PerfContext:
    """Context manager for performance timing."""
    
    def __init__(self, label: str):
        self.label = label
    
    def __enter__(self):
        PerfSession.start(self.label)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        PerfSession.stop(self.label)
        return False
