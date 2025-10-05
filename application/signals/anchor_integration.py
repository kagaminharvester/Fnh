"""
Phase A Anchor Tracking Integration for Stage 2.

This module provides integration between the anchor tracking system
and Stage 2 contact analysis, allowing anchor-based signal generation
as an alternative to legacy vertical displacement calculation.
"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np

# Import anchor tracking components
try:
    from application.detection.roles import RoleMapper, ROLE_PRIMARY_ANCHOR
    from application.signals.anchor_signal_builder import AnchorSignalBuilder, AnchorMetrics
    from application.signals.anchor_object_tracker import create_anchor_tracker
    _ANCHOR_TRACKING_AVAILABLE = True
except ImportError:
    _ANCHOR_TRACKING_AVAILABLE = False


class AnchorTrackingIntegration:
    """
    Integrates anchor tracking into Stage 2 processing.
    
    Provides anchor-based signal generation as an alternative to
    legacy vertical displacement calculation.
    """
    
    def __init__(self, 
                 app_settings,
                 logger: Optional[logging.Logger] = None,
                 enable_anchor_tracking: bool = True):
        """
        Initialize anchor tracking integration.
        
        Args:
            app_settings: Application settings object
            logger: Optional logger instance
            enable_anchor_tracking: Whether to enable anchor tracking
        """
        self.logger = logger or logging.getLogger(__name__)
        self.enabled = enable_anchor_tracking and _ANCHOR_TRACKING_AVAILABLE
        
        if not _ANCHOR_TRACKING_AVAILABLE:
            self.logger.warning("Anchor tracking modules not available, using legacy displacement")
            self.enabled = False
            return
        
        if not self.enabled:
            self.logger.info("Anchor tracking disabled, using legacy displacement")
            return
        
        # Get settings from app
        self.anchor_alpha = app_settings.get("anchor_alpha", 0.25)
        self.anchor_adaptive = app_settings.get("anchor_adaptive", False)
        self.anchor_gap_tolerance = app_settings.get("anchor_gap_tolerance", 10)
        self.anchor_outlier_mult = app_settings.get("anchor_outlier_mult", 4.0)
        self.anchor_percentile_window = app_settings.get("anchor_percentile_window", 120)
        self.anchor_tracker_type = app_settings.get("anchor_tracker", "none")
        self.anchor_min_warmup = app_settings.get("anchor_min_warmup", 80)
        
        # Initialize components
        self.role_mapper = RoleMapper(logger=self.logger)
        self.signal_builder = AnchorSignalBuilder(
            base_alpha=self.anchor_alpha,
            adaptive_smoothing=self.anchor_adaptive,
            gap_tolerance=self.anchor_gap_tolerance,
            outlier_mult=self.anchor_outlier_mult,
            percentile_window=self.anchor_percentile_window,
            min_warmup=self.anchor_min_warmup,
            logger=self.logger
        )
        self.object_tracker = create_anchor_tracker(
            tracker_type=self.anchor_tracker_type,
            logger=self.logger
        )
        
        self.metrics: Optional[AnchorMetrics] = None
        
        self.logger.info(f"Anchor tracking initialized: alpha={self.anchor_alpha}, "
                        f"adaptive={self.anchor_adaptive}, tracker={self.anchor_tracker_type}")
    
    def process_frame_detections(self, frame_id: int, frame_object) -> None:
        """
        Process detections for a single frame.
        
        Args:
            frame_id: Frame number
            frame_object: FrameObject containing detection boxes
        """
        if not self.enabled:
            return
        
        # Extract anchor role detections from frame
        anchor_detections = []
        for box in frame_object.boxes:
            if not box.is_excluded and self.role_mapper.is_primary_anchor(box.class_name):
                detection = {
                    'bbox': [box.x1, box.y1, box.x2, box.y2],
                    'confidence': box.confidence,
                    'class_name': box.class_name
                }
                anchor_detections.append(detection)
        
        # Apply object tracking if detections found
        if anchor_detections:
            tracked_detections = self.object_tracker.update(anchor_detections, frame_id)
            # Use the first tracked detection (highest confidence typically)
            if tracked_detections:
                self.signal_builder.add_detection(frame_id, tracked_detections[0])
            else:
                self.signal_builder.add_detection(frame_id, None)
        else:
            # No anchor detections in this frame
            self.signal_builder.add_detection(frame_id, None)
    
    def get_anchor_position(self, frame_id: int) -> Optional[float]:
        """
        Get normalized anchor position for a frame.
        
        Args:
            frame_id: Frame number
            
        Returns:
            Normalized position (0-100) or None if not available
        """
        if not self.enabled:
            return None
        
        return self.signal_builder.get_normalized_position(frame_id)
    
    def finalize(self, total_frames: int) -> Optional[AnchorMetrics]:
        """
        Finalize anchor tracking and compute metrics.
        
        Args:
            total_frames: Total number of frames processed
            
        Returns:
            AnchorMetrics or None if disabled
        """
        if not self.enabled:
            return None
        
        # Update ID switches from tracker
        id_switches = self.object_tracker.get_id_switches()
        self.signal_builder.set_id_switches(id_switches)
        
        # Finalize and get metrics
        self.metrics = self.signal_builder.finalize(total_frames)
        return self.metrics
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get anchor tracking metrics as dictionary."""
        if self.metrics:
            return self.metrics.to_dict()
        return None
    
    def apply_to_frame_objects(self, frame_objects: List, fallback_to_legacy: bool = True) -> bool:
        """
        Apply anchor positions to frame objects.
        
        This method updates the funscript_distance field on frame objects
        with anchor-based positions where available, falling back to
        legacy values if anchor tracking is disabled or positions unavailable.
        
        Args:
            frame_objects: List of FrameObject instances
            fallback_to_legacy: If True, keep legacy values when anchor unavailable
            
        Returns:
            True if anchor positions were applied, False otherwise
        """
        if not self.enabled:
            return False
        
        applied_count = 0
        for frame_obj in frame_objects:
            anchor_pos = self.get_anchor_position(frame_obj.frame_id)
            if anchor_pos is not None:
                # Store anchor position (optionally preserve legacy for comparison)
                if hasattr(frame_obj, 'anchor_position'):
                    frame_obj.anchor_position = anchor_pos
                # Update funscript_distance with anchor position
                if not fallback_to_legacy or frame_obj.funscript_distance == 50:
                    # Only override if fallback disabled or legacy is default/invalid
                    frame_obj.funscript_distance = int(round(anchor_pos))
                    applied_count += 1
        
        if applied_count > 0:
            self.logger.info(f"Applied anchor positions to {applied_count}/{len(frame_objects)} frames")
            return True
        
        return False


def create_anchor_integration(app, logger: Optional[logging.Logger] = None,
                              enable_anchor_tracking: bool = True) -> AnchorTrackingIntegration:
    """
    Factory function to create anchor tracking integration.
    
    Args:
        app: Application instance with settings
        logger: Optional logger
        enable_anchor_tracking: Whether to enable anchor tracking
        
    Returns:
        AnchorTrackingIntegration instance
    """
    app_settings = app.app_settings if hasattr(app, 'app_settings') else app
    return AnchorTrackingIntegration(
        app_settings=app_settings,
        logger=logger,
        enable_anchor_tracking=enable_anchor_tracking
    )


# Backward compatibility check
def is_anchor_tracking_available() -> bool:
    """Check if anchor tracking modules are available."""
    return _ANCHOR_TRACKING_AVAILABLE
