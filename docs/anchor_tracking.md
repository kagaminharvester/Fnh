# Anchor Tracking and Relative Motion

## Overview

Anchor tracking establishes a reference frame for motion analysis, enabling relative motion computation and dual-axis signal extraction. This document describes anchor detection, relative motion projection, and roll/secondary axis inference.

## Anchor Detection

### What is an Anchor?

The **anchor** is a stable reference point in the scene, typically:
- The camera viewpoint (for POV videos)
- A static object (bed, chair, etc.)
- The "stationary" participant in the action

### Anchor Selection Strategies

#### Automatic Detection

FunGen automatically identifies anchors using:

1. **Motion Analysis**: Low-motion regions likely to be anchor
2. **Spatial Consistency**: Stable position over time
3. **Size Heuristics**: Appropriate scale for reference
4. **Confidence Scoring**: Best candidate selected

#### Manual Override

Force specific anchor selection:

```bash
python main.py video.mp4 --anchor-class person --anchor-relative
```

### Anchor Classes

Supported anchor classes:
- `person`: Human figure (default)
- `bed`: Furniture anchor
- `chair`: Furniture anchor
- `auto`: Automatic selection (default)

## Relative Motion Projection

### Concept

Instead of absolute screen coordinates, compute motion **relative to anchor**:

```
displacement = stroker_position - anchor_position
```

This provides:
- Scene-independent measurements
- Invariance to camera motion
- Better generalization across videos

### Axis Decomposition

The displacement vector is projected onto two axes:

#### Primary Axis (Stroke)

**Definition**: Main direction of stroking motion

**Computation**:
1. Collect displacement vectors over sliding window (default: 30 frames)
2. Apply PCA to find principal component
3. Project displacement onto this axis

**Properties**:
- Represents up/down or in/out motion
- Normalized to 0-100 range
- Used for primary funscript output

#### Secondary Axis (Roll/Lateral)

**Definition**: Perpendicular to primary axis

**Computation**:
1. Secondary axis = perpendicular to primary (from PCA)
2. Project displacement onto this axis
3. Check if variance exceeds threshold (default: 0.15)

**Properties**:
- Represents side-to-side or rotational motion
- Only active if sufficient variance detected
- Used for optional `.roll.funscript` output

### Coordinate System

```
Screen Space:          Relative Space:
  (0,0)                    anchor
    ┌─────┐                  │
    │  •  │ anchor           │ primary axis
    │     │                  │    ↓
    │  •  │ stroker      stroker •──→ secondary axis
    └─────┘                       
```

## Enabling Relative Motion

### Basic Usage

```bash
# Enable anchor-relative mode
python main.py video.mp4 --anchor-relative
```

### With Roll Detection

```bash
# Enable secondary axis for roll
python main.py video.mp4 --anchor-relative --roll-axis
```

This generates two funscripts:
- `video.funscript` (primary axis)
- `video.roll.funscript` (secondary axis)

### Configuration

```bash
python main.py video.mp4 \
  --anchor-relative \
  --roll-axis \
  --axis-window 40 \
  --roll-threshold 0.2
```

**Parameters:**
- `--axis-window`: PCA window size (default: 30)
- `--roll-threshold`: Minimum variance for roll detection (default: 0.15)

## Axis Projection Algorithm

### Step-by-Step

1. **Observation Collection**
   ```python
   stroker_pos = detect_stroker(frame)
   anchor_pos = detect_anchor(frame)
   displacement = stroker_pos - anchor_pos
   ```

2. **Axis Estimation** (every N frames)
   ```python
   # Collect recent displacements
   displacements = [d1, d2, ..., dN]
   
   # Apply PCA
   pca = PCA(n_components=2)
   pca.fit(displacements)
   
   primary_axis = pca.components_[0]
   secondary_axis = pca.components_[1]
   ```

3. **Projection**
   ```python
   # Project onto axes
   primary_proj = dot(displacement, primary_axis)
   secondary_proj = dot(displacement, secondary_axis)
   
   # Normalize to 0-100
   primary_pos = normalize(primary_proj, min_seen, max_seen)
   ```

4. **Roll Detection**
   ```python
   # Check secondary axis variance
   variance_ratio = pca.explained_variance_ratio_[1]
   
   if variance_ratio > roll_threshold:
       enable_secondary_output()
   ```

## Roll Inference Heuristic

### When to Enable Roll

Roll detection is useful when:
- Side-to-side motion is significant
- Rotational/twisting action present
- Multi-axis toys used

### Roll Metrics

When profiling enabled:

```json
{
  "projection_metrics": {
    "primary_range": [0, 85],
    "secondary_range": [15, 65],
    "has_secondary": true,
    "axis_angle": 87.3,
    "variance_ratio": 0.23
  }
}
```

**Interpretation:**
- `variance_ratio > 0.15`: Roll active
- `variance_ratio < 0.15`: Roll negligible
- `axis_angle`: Primary axis orientation (degrees)

## Integration with Trackers

### Stage 2/3 Support

Relative motion integrates with stage-based analysis:

```bash
# Stage 3 with relative motion
python main.py video.mp4 \
  --mode 3-stage \
  --anchor-relative \
  --roll-axis
```

### Multi-Tracker Compatibility

Works with all multi-trackers:

```bash
python main.py video.mp4 \
  --multi-tracker bytetrack \
  --anchor-relative
```

## Quality Considerations

### When Relative Motion Helps

✅ **Good for:**
- POV videos with camera motion
- Multi-person scenes
- Handheld camera footage
- Zoom/pan shots

❌ **Not needed for:**
- Fixed camera angles
- Single subject, no camera motion
- Simple up/down motion

### Fallback Behavior

If anchor detection fails:
1. Log warning: "Anchor not detected, using absolute mode"
2. Fall back to absolute screen coordinates
3. Continue processing without relative projection

## Advanced Usage

### Custom Axis Orientation

For scenes with unusual motion directions:

```bash
# Invert primary axis
python main.py video.mp4 --anchor-relative --invert-axis

# Force horizontal primary axis
python main.py video.mp4 --anchor-relative --axis-hint horizontal
```

### Axis Smoothing

Reduce axis jitter in shaky footage:

```bash
python main.py video.mp4 \
  --anchor-relative \
  --axis-window 60 \
  --axis-smooth 0.3
```

### Visualization

Debug axis projection:

```bash
python main.py video.mp4 \
  --anchor-relative \
  --debug-overlay axis
```

Overlay shows:
- Anchor point (red)
- Stroker point (blue)
- Primary axis (green arrow)
- Secondary axis (yellow arrow, if active)

## Troubleshooting

### Axis Flickering

**Symptoms**: Rapid changes in axis orientation

**Solutions:**
1. Increase window size: `--axis-window 60`
2. Enable smoothing: `--axis-smooth 0.4`
3. Use more stable anchor class

### No Secondary Axis Detected

**Symptoms**: No `.roll.funscript` generated

**Solutions:**
1. Lower threshold: `--roll-threshold 0.1`
2. Check if motion truly has secondary component
3. Verify enough observations collected

### Poor Anchor Tracking

**Symptoms**: Anchor jumps between frames

**Solutions:**
1. Use multi-tracker: `--multi-tracker bytetrack`
2. Increase gap tolerance: `--gap-tolerance 10`
3. Improve lighting/contrast
4. Manual anchor class selection

### Inverted Motion

**Symptoms**: Funscript inverted from expected

**Solutions:**
1. Add `--invert-axis` flag
2. Check axis angle in metrics
3. Verify anchor vs stroker detection

## Best Practices

### For POV Videos

```bash
python main.py pov_video.mp4 \
  --anchor-relative \
  --anchor-class person \
  --multi-tracker bytetrack
```

### For Roll-Heavy Content

```bash
python main.py video.mp4 \
  --anchor-relative \
  --roll-axis \
  --roll-threshold 0.12 \
  --axis-window 40
```

### For Stable Cameras

```bash
python main.py video.mp4
# Relative motion not needed
```

## Performance Impact

**Overhead**: ~5-10% additional processing time

**Breakdown:**
- Axis estimation (PCA): 2-3%
- Projection computation: 1-2%
- Secondary axis (if enabled): +2-3%

**Mitigation:**
- Use larger `--axis-window` (less frequent PCA)
- Disable `--roll-axis` if not needed
- Enable `--tensorrt` for overall speedup

## Future Enhancements

**Post Phase C:**
- 3D motion estimation
- Depth-aware projection
- Multi-anchor support
- Automatic axis hint detection
- Pose keypoint integration
