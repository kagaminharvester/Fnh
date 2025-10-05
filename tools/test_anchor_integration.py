#!/usr/bin/env python
"""
Test anchor tracking integration with Stage 2-like structure.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

# Mock app settings
class MockSettings:
    def __init__(self):
        self._settings = {
            "anchor_alpha": 0.25,
            "anchor_adaptive": False,
            "anchor_gap_tolerance": 10,
            "anchor_outlier_mult": 4.0,
            "anchor_percentile_window": 120,
            "anchor_tracker": "simple",
            "anchor_min_warmup": 50,
        }
    
    def get(self, key, default=None):
        return self._settings.get(key, default)

# Mock frame object
class MockFrameObject:
    def __init__(self, frame_id):
        self.frame_id = frame_id
        self.boxes = []
        self.funscript_distance = 50  # Default legacy value

class MockBox:
    def __init__(self, x1, y1, x2, y2, class_name, confidence=0.9):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.class_name = class_name
        self.confidence = confidence
        self.is_excluded = False
        self.cx = (x1 + x2) / 2
        self.cy = (y1 + y2) / 2

def test_integration():
    """Test the integration module."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("Anchor Integration Test")
    print("=" * 60)
    
    from application.signals.anchor_integration import create_anchor_integration
    
    # Create mock app with settings
    class MockApp:
        def __init__(self):
            self.app_settings = MockSettings()
    
    app = MockApp()
    
    # Create integration
    integration = create_anchor_integration(app, logger=logger, enable_anchor_tracking=True)
    
    if not integration.enabled:
        print("✗ Integration not enabled")
        return False
    
    print("✓ Integration created and enabled")
    
    # Simulate processing frames
    frame_objects = []
    import math
    
    for frame_id in range(150):
        frame_obj = MockFrameObject(frame_id)
        
        # Add anchor detection (penis)
        cy = 300 + 80 * math.sin(frame_id * 0.1)
        box = MockBox(100, cy - 15, 180, cy + 15, "penis", 0.95)
        frame_obj.boxes.append(box)
        
        # Add other detection
        other_box = MockBox(300, 400, 350, 450, "hand", 0.85)
        frame_obj.boxes.append(other_box)
        
        frame_objects.append(frame_obj)
        
        # Process frame
        integration.process_frame_detections(frame_id, frame_obj)
    
    print(f"✓ Processed {len(frame_objects)} frames")
    
    # Finalize
    metrics = integration.finalize(total_frames=150)
    
    if metrics:
        print(f"✓ Metrics generated:")
        print(f"  Coverage: {metrics.coverage_ratio:.2%}")
        print(f"  Jitter: {metrics.jitter_score:.3f}")
        print(f"  Range: {metrics.normalization_range[0]:.1f} → {metrics.normalization_range[1]:.1f}")
    else:
        print("✗ No metrics generated")
        return False
    
    # Test getting positions
    valid_positions = 0
    for frame_id in range(150):
        pos = integration.get_anchor_position(frame_id)
        if pos is not None:
            valid_positions += 1
            assert 0 <= pos <= 100, f"Position {pos} out of range"
    
    print(f"✓ Retrieved {valid_positions}/150 valid positions")
    
    # Test applying to frame objects
    applied = integration.apply_to_frame_objects(frame_objects, fallback_to_legacy=False)
    
    if applied:
        # Check that positions were updated
        updated_count = sum(1 for fo in frame_objects if fo.funscript_distance != 50)
        print(f"✓ Applied anchor positions: {updated_count}/150 frames updated")
    else:
        print("✗ Failed to apply anchor positions")
        return False
    
    print("=" * 60)
    print("✓ All integration tests passed")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_integration()
    sys.exit(0 if success else 1)
