"""
Anchor Signal Builder for Phase A tracking.

Collects per-frame detections for anchor role, applies temporal smoothing,
outlier filtering, and dynamic normalization to generate high-quality
0-100 motion signal for funscript generation.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass, asdict


@dataclass
class AnchorMetrics:
    """Quality metrics for anchor tracking."""
    coverage_ratio: float = 0.0  # valid anchor frames / total
    jitter_score: float = 0.0    # std of first diff after smoothing / amplitude
    id_switches: int = 0          # if tracker active
    outliers_removed: int = 0     # count of outlier detections
    normalization_range: Tuple[float, float] = (0.0, 100.0)  # [low, high]
    warmup_frames: int = 0        # frames used for warmup
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class AnchorSignalBuilder:
    """
    Builds normalized anchor motion signal from per-frame detections.
    
    Pipeline:
    1. Collect anchor role detections per frame
    2. Apply object tracking for ID continuity (optional)
    3. Temporal smoothing (EMA with adaptive alpha)
    4. Outlier filtering (MAD-based)
    5. Gap interpolation
    6. Dynamic percentile normalization
    7. Generate quality metrics
    """
    
    def __init__(self,
                 base_alpha: float = 0.25,
                 adaptive_smoothing: bool = False,
                 gap_tolerance: int = 10,
                 outlier_mult: float = 4.0,
                 percentile_window: int = 120,
                 min_warmup: int = 80,
                 normalization_percentiles: Tuple[float, float] = (5.0, 95.0),
                 logger: Optional[logging.Logger] = None):
        """
        Initialize anchor signal builder.
        
        Args:
            base_alpha: Base EMA smoothing factor (0=max smoothing, 1=no smoothing)
            adaptive_smoothing: Enable adaptive smoothing based on velocity
            gap_tolerance: Max frames to interpolate missing detections
            outlier_mult: Multiplier for MAD-based outlier detection
            percentile_window: Window size for rolling percentile normalization
            min_warmup: Minimum frames to accumulate before normalization
            normalization_percentiles: (low, high) percentiles for normalization
            logger: Optional logger instance
        """
        self.base_alpha = base_alpha
        self.adaptive_smoothing = adaptive_smoothing
        self.gap_tolerance = gap_tolerance
        self.outlier_mult = outlier_mult
        self.percentile_window = percentile_window
        self.min_warmup = min_warmup
        self.norm_percentiles = normalization_percentiles
        self.logger = logger or logging.getLogger(__name__)
        
        # State
        self.frame_positions: Dict[int, float] = {}  # frame_id -> raw position
        self.smoothed_positions: Dict[int, float] = {}  # frame_id -> smoothed position
        self.normalized_positions: Dict[int, float] = {}  # frame_id -> 0-100 normalized
        
        # Tracking state
        self.last_smoothed_pos: Optional[float] = None
        self.position_history = deque(maxlen=percentile_window)
        
        # Gap tracking
        self.last_detected_frame: Optional[int] = None
        self.gap_start_frame: Optional[int] = None
        
        # Metrics
        self.total_frames = 0
        self.valid_frames = 0
        self.outliers_removed = 0
        self.id_switches = 0
        
        # Normalization range (updated during warmup)
        self.norm_low = 0.0
        self.norm_high = 100.0
        self.warmup_complete = False
    
    def add_detection(self, frame_id: int, detection: Optional[Dict[str, Any]]) -> None:
        """
        Add a detection for the current frame.
        
        Args:
            frame_id: Frame number
            detection: Detection dict with 'bbox' key, or None if no detection
        """
        self.total_frames = max(self.total_frames, frame_id + 1)
        
        if detection is None:
            # No detection - handle gap
            self._handle_gap(frame_id)
            return
        
        # Extract position from detection (using vertical center)
        bbox = detection['bbox']
        cy = (bbox[1] + bbox[3]) / 2.0
        
        # Check for outliers (if we have enough history)
        if self._is_outlier(cy):
            self.outliers_removed += 1
            self.logger.debug(f"Frame {frame_id}: Outlier detected at cy={cy:.1f}, skipping")
            self._handle_gap(frame_id)
            return
        
        # Valid detection
        self.frame_positions[frame_id] = cy
        self.valid_frames += 1
        
        # Interpolate gap if we had one
        if self.gap_start_frame is not None:
            self._interpolate_gap(self.gap_start_frame, frame_id)
            self.gap_start_frame = None
        
        # Apply temporal smoothing
        smoothed = self._apply_smoothing(cy, frame_id)
        self.smoothed_positions[frame_id] = smoothed
        self.position_history.append(smoothed)
        self.last_smoothed_pos = smoothed
        self.last_detected_frame = frame_id
        
        # Update normalization range if in warmup
        if not self.warmup_complete:
            self._update_normalization_range()
    
    def _is_outlier(self, position: float) -> bool:
        """Check if position is an outlier using MAD-based method."""
        if len(self.position_history) < 10:
            return False
        
        # Calculate median and MAD
        history_array = np.array(list(self.position_history))
        median = np.median(history_array)
        mad = np.median(np.abs(history_array - median))
        
        # Avoid division by zero
        if mad < 1e-6:
            return False
        
        # Check if position is beyond threshold
        z_score = np.abs(position - median) / (1.4826 * mad)  # 1.4826 converts MAD to std
        return z_score > self.outlier_mult
    
    def _apply_smoothing(self, raw_position: float, frame_id: int) -> float:
        """Apply EMA smoothing with optional adaptive alpha."""
        if self.last_smoothed_pos is None:
            return raw_position
        
        alpha = self.base_alpha
        
        if self.adaptive_smoothing and len(self.position_history) >= 2:
            # Calculate velocity z-score
            recent_positions = list(self.position_history)[-5:]
            if len(recent_positions) >= 2:
                velocities = np.diff(recent_positions)
                if len(velocities) > 0:
                    velocity = raw_position - self.last_smoothed_pos
                    mean_velocity = np.mean(velocities)
                    std_velocity = np.std(velocities)
                    
                    if std_velocity > 1e-6:
                        velocity_z = abs(velocity - mean_velocity) / std_velocity
                        
                        # Increase alpha (reduce smoothing) for rapid changes
                        if velocity_z > 2.0:
                            alpha = min(0.6, self.base_alpha * 2.0)
                        # Decrease alpha (increase smoothing) for low activity
                        elif velocity_z < 0.5:
                            alpha = max(0.1, self.base_alpha * 0.5)
        
        # Apply EMA
        smoothed = alpha * raw_position + (1 - alpha) * self.last_smoothed_pos
        return smoothed
    
    def _handle_gap(self, frame_id: int) -> None:
        """Handle missing detection by tracking gap start."""
        if self.gap_start_frame is None and self.last_detected_frame is not None:
            # Start of gap
            gap_length = frame_id - self.last_detected_frame
            if gap_length <= self.gap_tolerance:
                self.gap_start_frame = self.last_detected_frame + 1
            else:
                # Gap too long, reset tracking
                self.last_detected_frame = None
                self.last_smoothed_pos = None
    
    def _interpolate_gap(self, start_frame: int, end_frame: int) -> None:
        """Linear interpolation for short gaps."""
        if start_frame not in self.smoothed_positions and (start_frame - 1) in self.smoothed_positions:
            start_pos = self.smoothed_positions[start_frame - 1]
        elif start_frame in self.smoothed_positions:
            start_pos = self.smoothed_positions[start_frame]
        else:
            return
        
        end_pos = self.smoothed_positions.get(end_frame)
        if end_pos is None:
            return
        
        gap_frames = end_frame - start_frame
        for i in range(1, gap_frames):
            frame = start_frame + i
            alpha = i / float(gap_frames)
            interp_pos = start_pos + alpha * (end_pos - start_pos)
            self.smoothed_positions[frame] = interp_pos
            self.frame_positions[frame] = interp_pos
    
    def _update_normalization_range(self) -> None:
        """Update normalization range during warmup using percentiles."""
        if len(self.position_history) < self.min_warmup:
            return
        
        history_array = np.array(list(self.position_history))
        low_percentile, high_percentile = self.norm_percentiles
        
        self.norm_low = np.percentile(history_array, low_percentile)
        self.norm_high = np.percentile(history_array, high_percentile)
        
        # Mark warmup complete
        if not self.warmup_complete:
            self.warmup_complete = True
            self.logger.info(f"Anchor warmup complete: range [{self.norm_low:.1f}, {self.norm_high:.1f}]")
            
            # Retro-adjust early frames
            self._retroactive_normalize()
    
    def _retroactive_normalize(self) -> None:
        """Normalize all accumulated smoothed positions."""
        for frame_id, smoothed_pos in self.smoothed_positions.items():
            normalized = self._normalize_position(smoothed_pos)
            self.normalized_positions[frame_id] = normalized
    
    def _normalize_position(self, position: float) -> float:
        """Normalize position to 0-100 range."""
        if self.norm_high - self.norm_low < 1e-6:
            return 50.0
        
        normalized = 100.0 * (position - self.norm_low) / (self.norm_high - self.norm_low)
        return np.clip(normalized, 0.0, 100.0)
    
    def get_normalized_position(self, frame_id: int) -> Optional[float]:
        """
        Get normalized position for a frame.
        
        Args:
            frame_id: Frame number
            
        Returns:
            Normalized position (0-100) or None if not available
        """
        if frame_id in self.normalized_positions:
            return self.normalized_positions[frame_id]
        
        # If warmup complete but frame not normalized, try to normalize now
        if self.warmup_complete and frame_id in self.smoothed_positions:
            normalized = self._normalize_position(self.smoothed_positions[frame_id])
            self.normalized_positions[frame_id] = normalized
            return normalized
        
        return None
    
    def finalize(self, total_frames: Optional[int] = None) -> AnchorMetrics:
        """
        Finalize signal building and compute metrics.
        
        Args:
            total_frames: Total number of frames in video (if known)
            
        Returns:
            AnchorMetrics with quality statistics
        """
        if total_frames is not None:
            self.total_frames = max(self.total_frames, total_frames)
        
        # Ensure all smoothed positions are normalized
        if self.warmup_complete:
            for frame_id, smoothed_pos in self.smoothed_positions.items():
                if frame_id not in self.normalized_positions:
                    normalized = self._normalize_position(smoothed_pos)
                    self.normalized_positions[frame_id] = normalized
        
        # Calculate coverage ratio
        coverage_ratio = self.valid_frames / self.total_frames if self.total_frames > 0 else 0.0
        
        # Calculate jitter score (normalized std of first differences)
        jitter_score = 0.0
        if len(self.normalized_positions) >= 2:
            sorted_frames = sorted(self.normalized_positions.keys())
            positions = [self.normalized_positions[f] for f in sorted_frames]
            first_diffs = np.diff(positions)
            if len(first_diffs) > 0:
                jitter_std = np.std(first_diffs)
                amplitude = self.norm_high - self.norm_low if self.norm_high > self.norm_low else 1.0
                jitter_score = jitter_std / max(amplitude, 1.0)
        
        metrics = AnchorMetrics(
            coverage_ratio=coverage_ratio,
            jitter_score=jitter_score,
            id_switches=self.id_switches,
            outliers_removed=self.outliers_removed,
            normalization_range=(self.norm_low, self.norm_high),
            warmup_frames=self.min_warmup
        )
        
        # Log summary
        self.logger.info(
            f"ANCHOR: coverage={coverage_ratio:.2f} jitter={jitter_score:.3f} "
            f"idsw={self.id_switches} outliers={self.outliers_removed} "
            f"range={self.norm_low:.1f}→{self.norm_high:.1f}"
        )
        
        return metrics
    
    def reset(self) -> None:
        """Reset builder state."""
        self.frame_positions.clear()
        self.smoothed_positions.clear()
        self.normalized_positions.clear()
        self.last_smoothed_pos = None
        self.position_history.clear()
        self.last_detected_frame = None
        self.gap_start_frame = None
        self.total_frames = 0
        self.valid_frames = 0
        self.outliers_removed = 0
        self.id_switches = 0
        self.norm_low = 0.0
        self.norm_high = 100.0
        self.warmup_complete = False
    
    def set_id_switches(self, count: int) -> None:
        """Set ID switch count from external tracker."""
        self.id_switches = count
