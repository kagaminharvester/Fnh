"""
Multi-object tracking module for advanced tracking scenarios.
"""

from .multi_tracker import MultiTracker, TrackerAdapter, ByteTrackAdapter, OCSortAdapter, SimpleTrackerFallback

__all__ = ['MultiTracker', 'TrackerAdapter', 'ByteTrackAdapter', 'OCSortAdapter', 'SimpleTrackerFallback']
