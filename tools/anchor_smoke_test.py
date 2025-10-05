#!/usr/bin/env python
"""
Anchor Tracking Smoke Test

Quick validation tool for anchor tracking system.
Tests basic functionality with synthetic data.
"""

import sys
import json
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from application.detection.roles import RoleMapper, ROLE_PRIMARY_ANCHOR
from application.signals.anchor_signal_builder import AnchorSignalBuilder
from application.signals.anchor_object_tracker import create_anchor_tracker

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_role_mapper():
    """Test role mapping functionality."""
    logger.info("Testing RoleMapper...")
    
    mapper = RoleMapper(logger=logger)
    
    # Test basic mappings
    assert mapper.is_primary_anchor("penis"), "penis should be primary anchor"
    assert mapper.is_primary_anchor("Penis"), "case-insensitive matching should work"
    assert mapper.is_stroker("hand"), "hand should be stroker"
    assert mapper.is_target("pussy"), "pussy should be target"
    
    # Test unknown class
    assert mapper.get_role("unknown_class") == "ignore", "unknown class should be ignored"
    
    logger.info("✓ RoleMapper tests passed")
    return True


def test_anchor_signal_builder():
    """Test anchor signal builder with synthetic data."""
    logger.info("Testing AnchorSignalBuilder...")
    
    builder = AnchorSignalBuilder(
        base_alpha=0.25,
        gap_tolerance=10,
        min_warmup=50,
        logger=logger
    )
    
    # Simulate detections with vertical motion
    for frame_id in range(150):
        # Sinusoidal motion pattern
        import math
        cy = 300 + 100 * math.sin(frame_id * 0.1)
        
        detection = {
            'bbox': [100, cy - 10, 200, cy + 10],
            'confidence': 0.9,
            'class_name': 'penis'
        }
        
        builder.add_detection(frame_id, detection)
    
    # Add a gap
    for frame_id in range(150, 155):
        builder.add_detection(frame_id, None)
    
    # Resume detection
    for frame_id in range(155, 200):
        import math
        cy = 300 + 100 * math.sin(frame_id * 0.1)
        detection = {
            'bbox': [100, cy - 10, 200, cy + 10],
            'confidence': 0.9,
            'class_name': 'penis'
        }
        builder.add_detection(frame_id, detection)
    
    # Finalize and get metrics
    metrics = builder.finalize(total_frames=200)
    
    # Validate metrics
    assert metrics.coverage_ratio > 0.9, f"Coverage too low: {metrics.coverage_ratio}"
    assert 0 <= metrics.jitter_score < 1, f"Invalid jitter score: {metrics.jitter_score}"
    assert metrics.outliers_removed >= 0, f"Invalid outlier count: {metrics.outliers_removed}"
    assert len(metrics.normalization_range) == 2, "Invalid normalization range"
    
    # Check normalized positions are in valid range
    for frame_id in range(200):
        norm_pos = builder.get_normalized_position(frame_id)
        if norm_pos is not None:
            assert 0 <= norm_pos <= 100, f"Frame {frame_id}: position {norm_pos} out of range"
    
    logger.info("✓ AnchorSignalBuilder tests passed")
    logger.info(f"  Coverage: {metrics.coverage_ratio:.2%}")
    logger.info(f"  Jitter: {metrics.jitter_score:.3f}")
    logger.info(f"  Range: {metrics.normalization_range[0]:.1f} → {metrics.normalization_range[1]:.1f}")
    
    return True


def test_anchor_tracker():
    """Test anchor object tracker."""
    logger.info("Testing AnchorObjectTracker...")
    
    tracker = create_anchor_tracker('simple', logger=logger)
    
    # Create synthetic detections
    detections_frame1 = [
        {'bbox': [100, 200, 150, 250], 'confidence': 0.9, 'class_name': 'penis'},
        {'bbox': [300, 400, 350, 450], 'confidence': 0.8, 'class_name': 'hand'}
    ]
    
    detections_frame2 = [
        {'bbox': [105, 205, 155, 255], 'confidence': 0.9, 'class_name': 'penis'},
        {'bbox': [305, 405, 355, 455], 'confidence': 0.8, 'class_name': 'hand'}
    ]
    
    # Track across frames
    tracked1 = tracker.update(detections_frame1, frame_id=0)
    tracked2 = tracker.update(detections_frame2, frame_id=1)
    
    # Validate tracking
    assert len(tracked1) == 2, "Should track 2 objects in frame 1"
    assert len(tracked2) == 2, "Should track 2 objects in frame 2"
    assert all('track_id' in det for det in tracked1), "All detections should have track_id"
    assert all('track_id' in det for det in tracked2), "All detections should have track_id"
    
    # Check ID consistency (same objects should keep same IDs)
    ids_frame1 = sorted([det['track_id'] for det in tracked1])
    ids_frame2 = sorted([det['track_id'] for det in tracked2])
    assert ids_frame1 == ids_frame2, "Track IDs should be consistent across frames"
    
    logger.info("✓ AnchorObjectTracker tests passed")
    return True


def test_integration():
    """Test integration of components."""
    logger.info("Testing component integration...")
    
    # Create components
    mapper = RoleMapper(logger=logger)
    tracker = create_anchor_tracker('simple', logger=logger)
    builder = AnchorSignalBuilder(min_warmup=30, logger=logger)
    
    # Simulate pipeline
    for frame_id in range(100):
        # Simulate YOLO detections
        import math
        cy = 300 + 80 * math.sin(frame_id * 0.15)
        
        raw_detections = [
            {'bbox': [100, cy - 15, 180, cy + 15], 'confidence': 0.95, 'class_name': 'penis'},
            {'bbox': [300, 400, 350, 450], 'confidence': 0.85, 'class_name': 'hand'}
        ]
        
        # Apply tracking
        tracked = tracker.update(raw_detections, frame_id)
        
        # Filter to primary anchor using role mapper
        anchor_detections = [det for det in tracked if mapper.is_primary_anchor(det['class_name'])]
        
        # Add to signal builder (take first anchor if multiple)
        anchor_det = anchor_detections[0] if anchor_detections else None
        builder.add_detection(frame_id, anchor_det)
    
    # Finalize
    metrics = builder.finalize(total_frames=100)
    
    # Validate end-to-end
    assert metrics.coverage_ratio == 1.0, "Should have 100% coverage in integration test"
    assert metrics.normalization_range[0] < metrics.normalization_range[1], "Range should be valid"
    
    logger.info("✓ Integration tests passed")
    logger.info(f"  Metrics: {json.dumps(metrics.to_dict(), indent=2)}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Anchor Tracking Smoke Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Role Mapper", test_role_mapper),
        ("Anchor Signal Builder", test_anchor_signal_builder),
        ("Anchor Object Tracker", test_anchor_tracker),
        ("Component Integration", test_integration)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                logger.error(f"✗ {test_name} failed")
        except Exception as e:
            failed += 1
            logger.error(f"✗ {test_name} failed with exception: {e}", exc_info=True)
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
