"""
Relative Motion & Axis Projection
Author: FunGen AI System
Version: 1.0.0

Provides high-fidelity motion signal extraction:
- Stroker displacement relative to anchor axis
- Primary (stroke) and secondary (roll/lateral) component decomposition
- PCA-based or stabilized shaft vector approach
- Roll inference heuristic
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from sklearn.decomposition import PCA


@dataclass
class AxisProjectionResult:
    """Result of axis projection analysis."""
    primary_position: float  # Main stroke axis position (0-100)
    secondary_position: Optional[float] = None  # Roll/lateral position (0-100)
    primary_velocity: float = 0.0
    secondary_velocity: float = 0.0
    confidence: float = 1.0
    axis_angle: float = 0.0  # Angle of primary axis in degrees


class MotionProjector:
    """
    Compute stroker displacement relative to anchor axis with dual-axis decomposition.
    """
    
    def __init__(self, 
                 window_size: int = 30,
                 enable_secondary: bool = False,
                 roll_threshold: float = 0.15,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize motion projector.
        
        Args:
            window_size: Sliding window size for PCA
            enable_secondary: Enable secondary axis (roll/lateral) computation
            roll_threshold: Normalized variance threshold for roll detection
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.window_size = window_size
        self.enable_secondary = enable_secondary
        self.roll_threshold = roll_threshold
        
        # History buffers
        self.stroker_positions: deque = deque(maxlen=window_size)
        self.anchor_positions: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)
        
        # Axis state
        self.primary_axis: Optional[np.ndarray] = None
        self.secondary_axis: Optional[np.ndarray] = None
        self.axis_origin: Optional[np.ndarray] = None
        
        # Series storage
        self.primary_series: List[Tuple[int, float]] = []  # (time_ms, position)
        self.secondary_series: List[Tuple[int, float]] = []
        
    def add_observation(self, 
                       time_ms: int,
                       stroker_pos: Tuple[float, float],
                       anchor_pos: Tuple[float, float],
                       confidence: float = 1.0) -> Optional[AxisProjectionResult]:
        """
        Add new observation and compute projection.
        
        Args:
            time_ms: Timestamp in milliseconds
            stroker_pos: (x, y) position of stroker
            anchor_pos: (x, y) position of anchor
            confidence: Detection confidence
            
        Returns:
            AxisProjectionResult if projection computed, None otherwise
        """
        # Add to history
        self.stroker_positions.append(np.array(stroker_pos))
        self.anchor_positions.append(np.array(anchor_pos))
        self.timestamps.append(time_ms)
        
        # Need minimum observations for PCA
        if len(self.stroker_positions) < min(10, self.window_size // 3):
            return None
            
        # Update axis estimation
        self._update_axes()
        
        if self.primary_axis is None:
            return None
            
        # Compute relative displacement
        stroker = np.array(stroker_pos)
        anchor = np.array(anchor_pos)
        displacement = stroker - anchor
        
        # Project onto primary axis
        primary_proj = np.dot(displacement, self.primary_axis)
        
        # Normalize to 0-100 range (need to determine range from history)
        primary_range = self._compute_primary_range()
        if primary_range > 0:
            primary_position = (primary_proj - primary_range[0]) / (primary_range[1] - primary_range[0]) * 100
            primary_position = np.clip(primary_position, 0, 100)
        else:
            primary_position = 50.0
            
        result = AxisProjectionResult(
            primary_position=float(primary_position),
            confidence=confidence,
            axis_angle=self._compute_axis_angle()
        )
        
        # Compute secondary if enabled
        if self.enable_secondary and self.secondary_axis is not None:
            secondary_proj = np.dot(displacement, self.secondary_axis)
            secondary_range = self._compute_secondary_range()
            
            if secondary_range > 0:
                secondary_position = (secondary_proj - secondary_range[0]) / (secondary_range[1] - secondary_range[0]) * 100
                secondary_position = np.clip(secondary_position, 0, 100)
                result.secondary_position = float(secondary_position)
                
                # Compute velocities
                if len(self.primary_series) > 0:
                    dt = (time_ms - self.primary_series[-1][0]) / 1000.0
                    if dt > 0:
                        result.primary_velocity = (primary_position - self.primary_series[-1][1]) / dt
                        if result.secondary_position is not None and len(self.secondary_series) > 0:
                            result.secondary_velocity = (secondary_position - self.secondary_series[-1][1]) / dt
                            
        # Store in series
        self.primary_series.append((time_ms, result.primary_position))
        if result.secondary_position is not None:
            self.secondary_series.append((time_ms, result.secondary_position))
            
        return result
        
    def _update_axes(self):
        """Update primary and secondary axes using PCA."""
        if len(self.stroker_positions) < min(10, self.window_size // 3):
            return
            
        # Compute displacement vectors
        displacements = []
        for stroker, anchor in zip(self.stroker_positions, self.anchor_positions):
            displacement = stroker - anchor
            displacements.append(displacement)
            
        displacements = np.array(displacements)
        
        # Apply PCA
        try:
            pca = PCA(n_components=2)
            pca.fit(displacements)
            
            # Primary axis is first principal component
            self.primary_axis = pca.components_[0]
            
            # Ensure consistent orientation (pointing in direction of motion)
            if np.dot(self.primary_axis, np.array([0, 1])) < 0:
                self.primary_axis = -self.primary_axis
                
            # Secondary axis is second principal component (perpendicular)
            if self.enable_secondary:
                self.secondary_axis = pca.components_[1]
                
                # Check if secondary axis has significant variance
                variance_ratio = pca.explained_variance_ratio_[1]
                if variance_ratio < self.roll_threshold:
                    # Not enough variance for meaningful secondary axis
                    self.secondary_axis = None
                    
            # Update origin as mean anchor position
            self.axis_origin = np.mean(list(self.anchor_positions), axis=0)
            
        except Exception as e:
            self.logger.warning(f"Failed to update axes via PCA: {e}")
            
    def _compute_primary_range(self) -> Tuple[float, float]:
        """Compute min/max range of primary projections."""
        if len(self.stroker_positions) < 2:
            return (0.0, 100.0)
            
        projections = []
        for stroker, anchor in zip(self.stroker_positions, self.anchor_positions):
            displacement = stroker - anchor
            proj = np.dot(displacement, self.primary_axis)
            projections.append(proj)
            
        return (min(projections), max(projections))
        
    def _compute_secondary_range(self) -> Tuple[float, float]:
        """Compute min/max range of secondary projections."""
        if len(self.stroker_positions) < 2 or self.secondary_axis is None:
            return (0.0, 100.0)
            
        projections = []
        for stroker, anchor in zip(self.stroker_positions, self.anchor_positions):
            displacement = stroker - anchor
            proj = np.dot(displacement, self.secondary_axis)
            projections.append(proj)
            
        return (min(projections), max(projections))
        
    def _compute_axis_angle(self) -> float:
        """Compute angle of primary axis in degrees."""
        if self.primary_axis is None:
            return 0.0
            
        angle = np.arctan2(self.primary_axis[1], self.primary_axis[0])
        return float(np.degrees(angle))
        
    def get_primary_series(self) -> List[Dict[str, Any]]:
        """
        Get primary axis motion series in funscript action format.
        
        Returns:
            List of actions [{"at": time_ms, "pos": position}]
        """
        return [{"at": time_ms, "pos": int(pos)} for time_ms, pos in self.primary_series]
        
    def get_secondary_series(self) -> List[Dict[str, Any]]:
        """
        Get secondary axis motion series (roll) in funscript action format.
        
        Returns:
            List of actions [{"at": time_ms, "pos": position}]
        """
        if not self.enable_secondary:
            return []
        return [{"at": time_ms, "pos": int(pos)} for time_ms, pos in self.secondary_series]
        
    def has_secondary_axis(self) -> bool:
        """Check if secondary axis is active."""
        return self.enable_secondary and self.secondary_axis is not None
        
    def get_projection_metrics(self) -> Dict[str, Any]:
        """Get metrics about projection quality."""
        metrics = {
            "observations": len(self.stroker_positions),
            "primary_range": list(self._compute_primary_range()) if len(self.stroker_positions) >= 2 else [0, 0],
            "has_secondary": self.has_secondary_axis(),
        }
        
        if self.has_secondary_axis():
            metrics["secondary_range"] = list(self._compute_secondary_range())
            
        if self.primary_axis is not None:
            metrics["axis_angle"] = self._compute_axis_angle()
            
        return metrics
        
    def reset(self):
        """Reset projector state."""
        self.stroker_positions.clear()
        self.anchor_positions.clear()
        self.timestamps.clear()
        self.primary_axis = None
        self.secondary_axis = None
        self.axis_origin = None
        self.primary_series.clear()
        self.secondary_series.clear()
