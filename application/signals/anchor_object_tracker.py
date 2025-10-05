"""
Anchor Object Tracker interface and implementations.

Provides abstraction for tracking anchor detections across frames,
with optional integration of advanced trackers like BYTETrack or OC-SORT.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod


class AnchorObjectTracker(ABC):
    """
    Abstract interface for tracking anchor objects across frames.
    
    Implementations can range from simple center-based continuity
    to sophisticated multi-object tracking algorithms.
    """
    
    @abstractmethod
    def update(self, detections: List[Dict[str, Any]], frame_id: int) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dicts with keys:
                - 'bbox': [x1, y1, x2, y2]
                - 'confidence': float
                - 'class_name': str
            frame_id: Current frame number
            
        Returns:
            List of tracked detections with added 'track_id' field
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
        pass
    
    @abstractmethod
    def get_id_switches(self) -> int:
        """Get count of ID switches (for metrics)."""
        pass


class SimpleContinuityTracker(AnchorObjectTracker):
    """
    Simple center-based continuity tracker.
    
    Matches detections across frames based on spatial proximity
    of bounding box centers. No ML-based tracking.
    """
    
    def __init__(self, max_distance: float = 100.0, max_age: int = 10,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize simple continuity tracker.
        
        Args:
            max_distance: Maximum pixel distance for matching centers
            max_age: Maximum frames to keep track without detection
            logger: Optional logger instance
        """
        self.max_distance = max_distance
        self.max_age = max_age
        self.logger = logger or logging.getLogger(__name__)
        
        self.next_track_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}  # track_id -> track info
        self.id_switches = 0
        
    def update(self, detections: List[Dict[str, Any]], frame_id: int) -> List[Dict[str, Any]]:
        """Update tracker with new detections."""
        tracked_detections = []
        
        # Convert detections to centers for matching
        det_centers = []
        for det in detections:
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            det_centers.append((cx, cy))
        
        # Age out old tracks
        aged_out = []
        for track_id, track_info in self.tracks.items():
            if frame_id - track_info['last_frame'] > self.max_age:
                aged_out.append(track_id)
        for track_id in aged_out:
            del self.tracks[track_id]
        
        # Match detections to existing tracks
        matched_tracks = set()
        unmatched_detections = []
        
        for det_idx, (det, center) in enumerate(zip(detections, det_centers)):
            best_track_id = None
            best_distance = float('inf')
            
            # Find closest track
            for track_id, track_info in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                track_center = track_info['center']
                distance = np.sqrt((center[0] - track_center[0])**2 + 
                                  (center[1] - track_center[1])**2)
                
                if distance < best_distance and distance < self.max_distance:
                    best_distance = distance
                    best_track_id = track_id
            
            if best_track_id is not None:
                # Match found
                matched_tracks.add(best_track_id)
                self.tracks[best_track_id]['center'] = center
                self.tracks[best_track_id]['last_frame'] = frame_id
                self.tracks[best_track_id]['bbox'] = det['bbox']
                
                tracked_det = det.copy()
                tracked_det['track_id'] = best_track_id
                tracked_detections.append(tracked_det)
            else:
                # No match, will create new track
                unmatched_detections.append((det_idx, det, center))
        
        # Create new tracks for unmatched detections
        for det_idx, det, center in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            self.tracks[track_id] = {
                'center': center,
                'last_frame': frame_id,
                'bbox': det['bbox']
            }
            
            tracked_det = det.copy()
            tracked_det['track_id'] = track_id
            tracked_detections.append(tracked_det)
        
        return tracked_detections
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.tracks.clear()
        self.next_track_id = 1
        self.id_switches = 0
    
    def get_id_switches(self) -> int:
        """Get count of ID switches."""
        # Simple tracker doesn't track ID switches explicitly
        return 0


# Guarded imports for optional advanced trackers
_BYTETRACK_AVAILABLE = False
_OCSORT_AVAILABLE = False

try:
    # Attempt to import BYTETrack
    # This is a placeholder - actual import would depend on the installed package
    # from bytetrack import BYTETracker
    # _BYTETRACK_AVAILABLE = True
    pass
except ImportError:
    pass

try:
    # Attempt to import OC-SORT
    # This is a placeholder - actual import would depend on the installed package
    # from ocsort import OCSort
    # _OCSORT_AVAILABLE = True
    pass
except ImportError:
    pass


def create_anchor_tracker(tracker_type: str = 'none', 
                          logger: Optional[logging.Logger] = None,
                          **kwargs) -> AnchorObjectTracker:
    """
    Factory function to create an anchor tracker instance.
    
    Args:
        tracker_type: Type of tracker ('none', 'bytetrack', 'ocs')
        logger: Optional logger instance
        **kwargs: Additional parameters for tracker initialization
        
    Returns:
        AnchorObjectTracker instance
    """
    logger = logger or logging.getLogger(__name__)
    
    if tracker_type == 'none' or tracker_type == 'simple':
        logger.info("Using SimpleContinuityTracker for anchor tracking")
        return SimpleContinuityTracker(logger=logger, **kwargs)
    
    elif tracker_type == 'bytetrack':
        if not _BYTETRACK_AVAILABLE:
            logger.warning("BYTETrack requested but not available. Falling back to SimpleContinuityTracker.")
            logger.warning("Install BYTETrack for advanced tracking: pip install bytetrack")
            return SimpleContinuityTracker(logger=logger, **kwargs)
        # Would create BYTETrack adapter here if available
        logger.info("Using BYTETrack for anchor tracking")
        # return ByteTrackAdapter(logger=logger, **kwargs)
        return SimpleContinuityTracker(logger=logger, **kwargs)
    
    elif tracker_type == 'ocs' or tracker_type == 'ocsort':
        if not _OCSORT_AVAILABLE:
            logger.warning("OC-SORT requested but not available. Falling back to SimpleContinuityTracker.")
            logger.warning("Install OC-SORT for advanced tracking: pip install ocsort")
            return SimpleContinuityTracker(logger=logger, **kwargs)
        # Would create OC-SORT adapter here if available
        logger.info("Using OC-SORT for anchor tracking")
        # return OCSortAdapter(logger=logger, **kwargs)
        return SimpleContinuityTracker(logger=logger, **kwargs)
    
    else:
        logger.warning(f"Unknown tracker type '{tracker_type}'. Using SimpleContinuityTracker.")
        return SimpleContinuityTracker(logger=logger, **kwargs)
