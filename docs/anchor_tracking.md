# Anchor Tracking System

## Overview

The Anchor Tracking system (Phase A) provides enhanced primary anatomical tracking for generating cleaner, more natural funscripts. It implements a sophisticated pipeline for detecting, tracking, smoothing, and normalizing motion signals from video frames.

## Conceptual Pipeline

The anchor tracking pipeline consists of the following stages:

```
Raw Detections → Role Mapping → Continuity Tracking → Smoothing → 
Outlier Filtering → Gap Interpolation → Normalization → Funscript Emission
```

### 1. Detection & Role Mapping

- YOLO model produces per-frame object detections
- `RoleMapper` maps raw class names to functional roles:
  - `ROLE_PRIMARY_ANCHOR`: Main tracking target (e.g., "penis")
  - `ROLE_SECONDARY_ANCHOR`: Backup tracking target (e.g., "glans")
  - `ROLE_STROKER`: Interactive objects (e.g., "hand", "pussy")
  - `ROLE_TARGET`: Contact targets
  - `ROLE_IGNORE`: Classes to skip

### 2. Object Tracking (Optional)

The `AnchorObjectTracker` interface provides continuity across frames:

- **SimpleContinuityTracker** (default): Center-based proximity matching
- **BYTETrack** (optional): Advanced ML-based tracking
- **OC-SORT** (optional): Advanced ML-based tracking

If advanced trackers are not installed, the system gracefully falls back to `SimpleContinuityTracker`.

### 3. Temporal Smoothing

Exponential Moving Average (EMA) smoothing reduces jitter:

- **Base Alpha**: Controls smoothing strength (default: 0.25)
  - Lower values = more smoothing (lag)
  - Higher values = less smoothing (more responsive)

- **Adaptive Smoothing** (optional): Dynamically adjusts alpha based on velocity
  - Rapid motion → increase alpha (reduce lag)
  - Low activity → decrease alpha (suppress jitter)

### 4. Outlier Filtering

Median Absolute Deviation (MAD) based outlier detection:

- Calculates median and MAD from recent history
- Flags positions beyond `outlier_mult * MAD` threshold
- Outliers are skipped, gaps are interpolated if short enough

### 5. Gap Handling

When detection is lost:

- **Short gaps** (≤ `gap_tolerance` frames): Linear interpolation
- **Long gaps** (> `gap_tolerance` frames): Reset tracking segment

### 6. Dynamic Normalization

Percentile-based normalization ensures stable 0-100 output:

- **Warmup Phase**: Accumulate `min_warmup` frames
- **Percentile Calculation**: Use 5th and 95th percentiles (configurable)
- **Range Update**: Recompute when range shifts significantly
- **Retro-adjustment**: Normalize early frames after warmup

### 7. Quality Metrics

The system generates comprehensive quality metrics:

```python
{
  "coverage_ratio": 0.94,      # valid frames / total frames
  "jitter_score": 0.07,         # normalized std of first differences
  "id_switches": 0,             # tracker ID changes
  "outliers_removed": 5,        # outlier count
  "normalization_range": [12.0, 87.0],  # [low, high] percentiles
  "warmup_frames": 80           # frames used for warmup
}
```

## CLI Parameters

### Basic Usage

```bash
python main.py video.mp4 --mode 3-stage
```

### Anchor Tracking Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--anchor-alpha` | float | 0.25 | Base EMA smoothing factor |
| `--anchor-adaptive` | flag | false | Enable adaptive smoothing |
| `--anchor-gap-tolerance` | int | 10 | Max frames to interpolate |
| `--anchor-outlier-mult` | float | 4.0 | MAD outlier threshold multiplier |
| `--anchor-percentile-window` | int | 120 | Rolling window for percentiles |
| `--anchor-tracker` | choice | none | Tracker type (none/simple/bytetrack/ocs) |
| `--anchor-min-warmup` | int | 80 | Min frames before normalization |

### Examples

**Default anchor tracking:**
```bash
python main.py video.mp4 --mode 3-stage
```

**High responsiveness, low smoothing:**
```bash
python main.py video.mp4 --mode 3-stage --anchor-alpha 0.5
```

**Adaptive smoothing for mixed content:**
```bash
python main.py video.mp4 --mode 3-stage --anchor-adaptive
```

**Advanced tracker with BYTETrack:**
```bash
python main.py video.mp4 --mode 3-stage --anchor-tracker bytetrack
```

**Custom normalization for longer scenes:**
```bash
python main.py video.mp4 --mode 3-stage --anchor-percentile-window 240 --anchor-min-warmup 120
```

## Parameter Tuning Guide

### Smoothing (`--anchor-alpha`)

- **Problem**: Output too jittery
  - **Solution**: Decrease alpha (e.g., 0.15-0.20)
  
- **Problem**: Output lags behind motion
  - **Solution**: Increase alpha (e.g., 0.35-0.45)

### Adaptive Smoothing (`--anchor-adaptive`)

- **Best for**: Videos with varying motion speeds (slow then fast)
- **Avoid**: Consistent-speed content where base smoothing is sufficient

### Gap Tolerance (`--anchor-gap-tolerance`)

- **Problem**: Too many tracking resets
  - **Solution**: Increase gap tolerance (e.g., 15-20)
  
- **Problem**: Interpolation creates unrealistic motion
  - **Solution**: Decrease gap tolerance (e.g., 5-8)

### Outlier Multiplier (`--anchor-outlier-mult`)

- **Problem**: False positives (good detections marked as outliers)
  - **Solution**: Increase multiplier (e.g., 5.0-6.0)
  
- **Problem**: Outliers not being filtered
  - **Solution**: Decrease multiplier (e.g., 3.0-3.5)

### Percentile Window (`--anchor-percentile-window`)

- **Short scenes**: Use smaller window (60-90 frames)
- **Long scenes**: Use larger window (180-300 frames)
- **Mixed content**: Default (120) is usually good

### Min Warmup (`--anchor-min-warmup`)

- Should be ≥ 50% of percentile window
- Larger values = more stable normalization
- Smaller values = faster adaptation to scene changes

## Troubleshooting

### Low Coverage Ratio

**Symptom**: `coverage_ratio < 0.7`

**Possible causes**:
- Anchor object frequently occluded or out of frame
- Model confidence threshold too high
- Wrong video preprocessing (VR format mismatch)

**Solutions**:
- Check YOLO confidence threshold
- Verify video format detection
- Review discarded class list

### High Jitter Score

**Symptom**: `jitter_score > 0.15`

**Possible causes**:
- Insufficient smoothing
- Unstable detections

**Solutions**:
- Decrease `--anchor-alpha`
- Enable `--anchor-adaptive`
- Increase `--anchor-outlier-mult`

### Excessive ID Switches

**Symptom**: `id_switches > 10`

**Possible causes**:
- Multiple similar objects in frame
- Simple tracker limitations

**Solutions**:
- Use `--anchor-tracker bytetrack` or `--anchor-tracker ocs`
- Review video for multi-person scenes
- Check role mapping configuration

### Normalization Range Issues

**Symptom**: `normalization_range` too narrow (< 30 units)

**Possible causes**:
- Limited motion in scene
- Camera follows subject too closely

**Solutions**:
- Increase `--anchor-percentile-window`
- Adjust percentile values (in code)
- Consider scene suitability for tracking

## Known Limitations

### Current Phase A

1. **Single anchor tracking only**: Multi-anchor blending not yet implemented
2. **No pose keypoint fusion**: Pose data not integrated (placeholder only)
3. **No multi-person disambiguation**: Assumes single subject
4. **Limited tracker options**: BYTETrack/OC-SORT stubs (not fully integrated)

### Future Enhancements (Phase B/C)

1. **Pose fusion**: Integrate hip/torso keypoints for robust tracking
2. **Multi-anchor blending**: Combine primary + secondary anchor signals
3. **Relative motion mapping**: Track stroker-target interactions
4. **Performance optimization**: Micro-batching, TensorRT integration
5. **Advanced ID stability**: Better multi-person handling

## Integration Notes

### For Developers

The anchor tracking system integrates into the existing pipeline:

1. **Stage 1**: Raw YOLO detections → `RoleMapper`
2. **Between Stages**: `AnchorSignalBuilder` processes anchor detections
3. **Stage 2/3**: Normalized anchor signal replaces legacy displacement
4. **Output**: Funscript generation uses enhanced signal

**Key classes**:
- `application.detection.roles.RoleMapper`: Class name → role mapping
- `application.signals.anchor_signal_builder.AnchorSignalBuilder`: Main pipeline
- `application.signals.anchor_object_tracker.AnchorObjectTracker`: Tracking interface

**Configuration**:
- Role mappings: JSON file or defaults
- CLI args propagate through `ApplicationLogic`
- Backward compatible: legacy path used if anchor disabled

## Performance Considerations

### Memory Usage

- Position history: ~1KB per frame (negligible)
- Full video tracking: ~1MB per 1000 frames

### CPU Impact

- Simple tracker: ~0.01ms per frame
- Advanced trackers: ~0.5-2ms per frame (if available)
- Overall: < 5% pipeline overhead

### Recommended Settings

**Fast processing (realtime)**:
```bash
--anchor-alpha 0.3 --anchor-gap-tolerance 5 --anchor-min-warmup 60
```

**High quality (offline)**:
```bash
--anchor-alpha 0.2 --anchor-adaptive --anchor-tracker bytetrack --anchor-percentile-window 180
```

**Balanced (default)**:
```bash
# Use defaults, optionally add --anchor-adaptive
```

## References

- YOLO Object Detection: https://github.com/ultralytics/ultralytics
- BYTETrack: https://github.com/ifzhang/ByteTrack
- OC-SORT: https://github.com/noahcao/OC_SORT
- MAD Outlier Detection: https://en.wikipedia.org/wiki/Median_absolute_deviation
