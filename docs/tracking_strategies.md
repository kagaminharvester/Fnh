# Tracking Strategies Guide

## Overview

FunGen supports multiple tracking strategies for different use cases. This guide explains the available multi-object tracking options, their trade-offs, and when to use each.

## Multi-Object Tracking

### What is Multi-Object Tracking?

Multi-object tracking (MOT) assigns persistent IDs to detected objects across frames, enabling:

- Consistent identity assignment for multiple people/objects
- Trajectory analysis over time
- Better handling of occlusions and temporary disappearances
- Smoother motion signals

### Available Trackers

#### 1. BYTETrack (Recommended)

**Description**: State-of-the-art MOT algorithm combining high and low confidence detections.

**Pros:**
- Excellent accuracy for crowded scenes
- Robust to occlusions
- Fast and efficient
- Well-maintained library

**Cons:**
- Requires `yolox` package (optional dependency)
- Slightly higher memory usage

**Use cases:**
- Videos with multiple people
- Scenes with occlusions
- When accuracy is priority

**Installation:**
```bash
pip install yolox
```

**Usage:**
```bash
python main.py video.mp4 --multi-tracker bytetrack
```

#### 2. OC-SORT

**Description**: Improved SORT with observation-centric approach and motion smoothing.

**Pros:**
- Good balance of speed and accuracy
- Better motion smoothing than BYTETrack
- Lower memory footprint

**Cons:**
- Requires `ocsort` package (optional dependency)
- Less robust to crowding than BYTETrack

**Use cases:**
- Single or few subjects
- When motion smoothness is important
- Resource-constrained systems

**Installation:**
```bash
pip install ocsort
```

**Usage:**
```bash
python main.py video.mp4 --multi-tracker ocsort
```

#### 3. Simple Fallback (Default)

**Description**: Basic IOU-based tracker with no external dependencies.

**Pros:**
- No dependencies required
- Lightweight and fast
- Always available

**Cons:**
- Less accurate than specialized trackers
- Struggles with occlusions
- More ID switches

**Use cases:**
- Single subject videos
- When dependencies unavailable
- Testing and development

**Usage:**
```bash
python main.py video.mp4 --multi-tracker none
# or let auto-selection choose fallback
```

#### 4. Auto-Selection (Default)

**Description**: Automatically selects best available tracker.

**Selection priority:**
1. BYTETrack (if installed)
2. OC-SORT (if installed)
3. Simple Fallback (always available)

**Usage:**
```bash
python main.py video.mp4 --multi-tracker auto
# or omit flag (auto is default)
```

## Tracking Parameters

### Gap Tolerance

Maximum frames to interpolate for disappeared objects:

```bash
python main.py video.mp4 --gap-tolerance 10
```

- **Lower (5)**: Faster ID reassignment, less interpolation
- **Higher (30)**: More robust to brief occlusions
- **Default: 5**

### Track Threshold

Minimum confidence for track initialization (BYTETrack):

```bash
python main.py video.mp4 --track-thresh 0.6
```

- **Lower (0.3)**: More tracks, more false positives
- **Higher (0.7)**: Fewer tracks, higher quality
- **Default: 0.5**

## Tracking Quality Metrics

When `--profile-run` is enabled, tracking metrics are reported:

```json
{
  "tracking": {
    "total_tracks": 2,
    "id_switches": 1,
    "avg_track_length": 450,
    "disappearance_events": 3
  }
}
```

### ID Switches

**Definition**: Number of times a tracked object's ID changed.

**Target**: <2% of total frames

**If high:**
- Lower track threshold
- Increase gap tolerance
- Try different tracker

### Track Length

**Definition**: Average number of frames per tracked object.

**Target**: >50% of video length for main subjects

**If low:**
- Increase gap tolerance
- Improve detection quality
- Check for scene cuts

## Integration with Existing Trackers

### Stage 2/3 Compatibility

Multi-object tracking integrates with existing stage-based trackers:

```bash
# 3-stage with multi-tracking
python main.py video.mp4 --mode 3-stage --multi-tracker bytetrack

# Stage 2 with multi-tracking
python main.py video.mp4 --mode stage2 --multi-tracker ocsort
```

### Live vs Offline

**Offline modes** (Stage 2, Stage 3):
- Full multi-tracking support
- Can use detection cache
- Best quality

**Live modes** (Oscillation, Hybrid):
- Simple tracking only (latency requirements)
- Real-time constraints
- Immediate response needed

## Best Practices

### For Single-Subject Videos

```bash
python main.py video.mp4 --multi-tracker none
```
- Simple fallback sufficient
- Saves memory
- Faster processing

### For Multi-Person Videos

```bash
python main.py video.mp4 --multi-tracker bytetrack --gap-tolerance 10
```
- Use BYTETrack for accuracy
- Higher gap tolerance for occlusions
- Monitor ID switches

### For Fast Processing

```bash
python main.py video.mp4 --multi-tracker ocsort --gap-tolerance 5
```
- OC-SORT is lighter than BYTETrack
- Lower gap tolerance = less interpolation work

### For Maximum Quality

```bash
python main.py video.mp4 \
  --multi-tracker bytetrack \
  --track-thresh 0.6 \
  --gap-tolerance 15 \
  --tensorrt
```
- Best tracker
- Balanced threshold
- Moderate gap tolerance
- Hardware acceleration

## Troubleshooting

### Too Many ID Switches

**Symptoms**: Jerky motion, position jumps

**Solutions:**
1. Increase gap tolerance: `--gap-tolerance 15`
2. Lower track threshold: `--track-thresh 0.4`
3. Switch to BYTETrack if using Simple
4. Check detection quality (model, lighting)

### Missing Tracks

**Symptoms**: Subject not tracked consistently

**Solutions:**
1. Lower track threshold: `--track-thresh 0.3`
2. Increase gap tolerance: `--gap-tolerance 20`
3. Improve lighting/contrast in video
4. Use better detection model

### High Memory Usage

**Symptoms**: Out of memory errors

**Solutions:**
1. Switch to Simple fallback: `--multi-tracker none`
2. Reduce batch size: `--batch-size 2`
3. Use OC-SORT instead of BYTETrack

### Slow Processing

**Symptoms**: Low FPS during tracking stage

**Solutions:**
1. Use Simple fallback for single-subject
2. Reduce gap tolerance: `--gap-tolerance 3`
3. Enable TensorRT for inference speedup
4. Use detection cache for repeated processing

## Future Enhancements

**Post Phase C:**
- INT8 quantization for trackers
- Multi-person disambiguation heuristics
- Pose keypoint fusion
- Track quality scoring
- Automatic parameter tuning
