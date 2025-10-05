"""
Multi-Object Tracking Layer
Author: FunGen AI System
Version: 1.0.0

Provides abstract interface for multi-object tracking with support for:
- BYTETrack integration
- OC-SORT integration  
- Simple fallback tracker
- Auto-selection based on availability
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TrackedObject:
    """Represents a tracked object with persistent ID."""
    track_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    frame_idx: int
    disappeared_frames: int = 0


class TrackerAdapter(ABC):
    """Abstract base class for tracking adapters."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.id_switches = 0
        self.active_tracks: Dict[int, TrackedObject] = {}
        
    @abstractmethod
    def update(self, frame_idx: int, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        
        Args:
            frame_idx: Current frame index
            detections: List of detections with 'bbox', 'confidence', 'class_id'
            
        Returns:
            List of tracked objects with persistent IDs
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset tracker state."""
        pass


class ByteTrackAdapter(TrackerAdapter):
    """Adapter for BYTETrack multi-object tracker."""
    
    def __init__(self, track_thresh: float = 0.5, track_buffer: int = 30, 
                 match_thresh: float = 0.8, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.tracker = None
        self._init_tracker()
        
    def _init_tracker(self):
        """Initialize BYTETrack if available."""
        try:
            # Try to import ByteTrack (soft dependency)
            from yolox.tracker.byte_tracker import BYTETracker
            
            # BYTETrack expects args object
            class Args:
                def __init__(self):
                    self.track_thresh = 0.5
                    self.track_buffer = 30
                    self.match_thresh = 0.8
                    self.mot20 = False
            
            self.tracker = BYTETracker(Args(), frame_rate=30)
            self.logger.info("BYTETrack initialized successfully")
            self.available = True
        except ImportError:
            self.logger.warning("BYTETrack not available, falling back")
            self.available = False
            self.tracker = None
            
    def update(self, frame_idx: int, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """Update with BYTETrack."""
        if not self.available or self.tracker is None:
            return self._simple_update(frame_idx, detections)
            
        try:
            # Convert detections to BYTETrack format: [x1, y1, x2, y2, conf, class]
            if not detections:
                return []
                
            dets = np.array([
                [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 
                 d['confidence'], d['class_id']]
                for d in detections
            ])
            
            # Update tracker
            online_targets = self.tracker.update(dets, [1080, 1920], [1080, 1920])
            
            # Convert to TrackedObject
            tracked_objects = []
            for track in online_targets:
                tlbr = track.tlbr
                tracked_obj = TrackedObject(
                    track_id=track.track_id,
                    bbox=(float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])),
                    confidence=float(track.score),
                    class_id=int(track.class_id) if hasattr(track, 'class_id') else 0,
                    frame_idx=frame_idx,
                    disappeared_frames=0
                )
                tracked_objects.append(tracked_obj)
                
            return tracked_objects
            
        except Exception as e:
            self.logger.error(f"BYTETrack update error: {e}, falling back")
            return self._simple_update(frame_idx, detections)
    
    def _simple_update(self, frame_idx: int, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """Simple fallback update without tracking library."""
        tracked_objects = []
        for idx, det in enumerate(detections):
            tracked_obj = TrackedObject(
                track_id=idx,
                bbox=det['bbox'],
                confidence=det['confidence'],
                class_id=det['class_id'],
                frame_idx=frame_idx
            )
            tracked_objects.append(tracked_obj)
        return tracked_objects
        
    def reset(self):
        """Reset tracker."""
        if self.available and self.tracker is not None:
            self._init_tracker()
        self.active_tracks.clear()
        self.id_switches = 0


class OCSortAdapter(TrackerAdapter):
    """Adapter for OC-SORT multi-object tracker."""
    
    def __init__(self, det_thresh: float = 0.5, max_age: int = 30,
                 iou_threshold: float = 0.3, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.tracker = None
        self._init_tracker()
        
    def _init_tracker(self):
        """Initialize OC-SORT if available."""
        try:
            # Try to import OC-SORT (soft dependency)
            from ocsort.ocsort import OCSort
            
            self.tracker = OCSort(
                det_thresh=self.det_thresh,
                max_age=self.max_age,
                iou_threshold=self.iou_threshold,
                use_byte=False
            )
            self.logger.info("OC-SORT initialized successfully")
            self.available = True
        except ImportError:
            self.logger.warning("OC-SORT not available, falling back")
            self.available = False
            self.tracker = None
            
    def update(self, frame_idx: int, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """Update with OC-SORT."""
        if not self.available or self.tracker is None:
            return self._simple_update(frame_idx, detections)
            
        try:
            if not detections:
                # Update with empty detections
                tracks = self.tracker.update(np.empty((0, 5)))
                return []
                
            # Convert detections to OC-SORT format: [x1, y1, x2, y2, conf]
            dets = np.array([
                [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['confidence']]
                for d in detections
            ])
            
            # Update tracker - returns [x1, y1, x2, y2, track_id]
            tracks = self.tracker.update(dets)
            
            # Convert to TrackedObject
            tracked_objects = []
            for track in tracks:
                tracked_obj = TrackedObject(
                    track_id=int(track[4]),
                    bbox=(float(track[0]), float(track[1]), float(track[2]), float(track[3])),
                    confidence=1.0,  # OC-SORT doesn't return confidence
                    class_id=0,  # Default class
                    frame_idx=frame_idx,
                    disappeared_frames=0
                )
                tracked_objects.append(tracked_obj)
                
            return tracked_objects
            
        except Exception as e:
            self.logger.error(f"OC-SORT update error: {e}, falling back")
            return self._simple_update(frame_idx, detections)
    
    def _simple_update(self, frame_idx: int, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """Simple fallback update without tracking library."""
        tracked_objects = []
        for idx, det in enumerate(detections):
            tracked_obj = TrackedObject(
                track_id=idx,
                bbox=det['bbox'],
                confidence=det['confidence'],
                class_id=det['class_id'],
                frame_idx=frame_idx
            )
            tracked_objects.append(tracked_obj)
        return tracked_objects
        
    def reset(self):
        """Reset tracker."""
        if self.available and self.tracker is not None:
            self._init_tracker()
        self.active_tracks.clear()
        self.id_switches = 0


class SimpleTrackerFallback(TrackerAdapter):
    """Simple IOU-based tracker as fallback."""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30, 
                 logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 0
        self.tracks: Dict[int, TrackedObject] = {}
        
    def _compute_iou(self, bbox1: Tuple[float, float, float, float], 
                     bbox2: Tuple[float, float, float, float]) -> float:
        """Compute IOU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Compute intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Compute union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
            
        return intersection / union
        
    def update(self, frame_idx: int, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """Simple IOU-based tracking."""
        # Mark all tracks as potentially disappeared
        for track in self.tracks.values():
            track.disappeared_frames += 1
            
        # Match detections to existing tracks
        matched_tracks = set()
        tracked_objects = []
        
        for det in detections:
            best_iou = 0.0
            best_track_id = None
            
            # Find best matching track
            for track_id, track in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                    
                iou = self._compute_iou(det['bbox'], track.bbox)
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
                    
            if best_track_id is not None:
                # Update existing track
                track = self.tracks[best_track_id]
                track.bbox = det['bbox']
                track.confidence = det['confidence']
                track.frame_idx = frame_idx
                track.disappeared_frames = 0
                matched_tracks.add(best_track_id)
                tracked_objects.append(track)
            else:
                # Create new track
                new_track = TrackedObject(
                    track_id=self.next_id,
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    class_id=det['class_id'],
                    frame_idx=frame_idx,
                    disappeared_frames=0
                )
                self.tracks[self.next_id] = new_track
                tracked_objects.append(new_track)
                self.next_id += 1
                
        # Remove old tracks
        tracks_to_remove = [
            track_id for track_id, track in self.tracks.items()
            if track.disappeared_frames > self.max_age
        ]
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            
        return tracked_objects
        
    def reset(self):
        """Reset tracker."""
        self.tracks.clear()
        self.active_tracks.clear()
        self.next_id = 0
        self.id_switches = 0


class MultiTracker:
    """
    Main multi-tracker interface with auto-selection and fallback.
    """
    
    def __init__(self, tracker_type: str = 'auto', gap_tolerance: int = 5,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize multi-tracker.
        
        Args:
            tracker_type: 'auto', 'bytetrack', 'ocsort', or 'none'
            gap_tolerance: Max frames to interpolate for disappeared objects
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.gap_tolerance = gap_tolerance
        self.tracker_type = tracker_type
        self.adapter: Optional[TrackerAdapter] = None
        self._init_tracker()
        
    def _init_tracker(self):
        """Initialize tracker based on type."""
        if self.tracker_type == 'none':
            self.logger.info("Multi-tracking disabled")
            self.adapter = None
            return
            
        if self.tracker_type == 'auto':
            # Try BYTETrack first, then OC-SORT, then fallback
            bytetrack = ByteTrackAdapter(logger=self.logger)
            if bytetrack.available:
                self.adapter = bytetrack
                self.logger.info("Using BYTETrack for multi-object tracking")
                return
                
            ocsort = OCSortAdapter(logger=self.logger)
            if ocsort.available:
                self.adapter = ocsort
                self.logger.info("Using OC-SORT for multi-object tracking")
                return
                
            self.adapter = SimpleTrackerFallback(logger=self.logger)
            self.logger.info("Using simple fallback tracker")
            
        elif self.tracker_type == 'bytetrack':
            bytetrack = ByteTrackAdapter(logger=self.logger)
            if bytetrack.available:
                self.adapter = bytetrack
            else:
                self.logger.warning("BYTETrack requested but not available, using fallback")
                self.adapter = SimpleTrackerFallback(logger=self.logger)
                
        elif self.tracker_type == 'ocsort':
            ocsort = OCSortAdapter(logger=self.logger)
            if ocsort.available:
                self.adapter = ocsort
            else:
                self.logger.warning("OC-SORT requested but not available, using fallback")
                self.adapter = SimpleTrackerFallback(logger=self.logger)
                
        else:
            self.logger.warning(f"Unknown tracker type '{self.tracker_type}', using fallback")
            self.adapter = SimpleTrackerFallback(logger=self.logger)
            
    def update(self, frame_idx: int, detections: List[Dict[str, Any]]) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        
        Args:
            frame_idx: Current frame index
            detections: List of detections with 'bbox', 'confidence', 'class_id'
            
        Returns:
            List of tracked objects with persistent IDs
        """
        if self.adapter is None:
            # No tracking, just return detections as-is
            return [
                TrackedObject(
                    track_id=idx,
                    bbox=det['bbox'],
                    confidence=det['confidence'],
                    class_id=det['class_id'],
                    frame_idx=frame_idx
                )
                for idx, det in enumerate(detections)
            ]
            
        return self.adapter.update(frame_idx, detections)
        
    def reset(self):
        """Reset tracker state."""
        if self.adapter is not None:
            self.adapter.reset()
            
    def get_id_switches(self) -> int:
        """Get number of ID switches."""
        if self.adapter is not None:
            return self.adapter.id_switches
        return 0
