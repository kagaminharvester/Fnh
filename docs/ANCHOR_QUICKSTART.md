# Phase A: Enhanced Primary Anchor Tracking - Quick Start

## Overview

Phase A introduces an enhanced anchor tracking system that provides more stable, accurate funscript generation by tracking the primary anatomical anchor (penis) with:

- Advanced temporal smoothing (EMA with adaptive alpha)
- Outlier filtering (MAD-based)
- Dynamic percentile normalization
- Quality metrics for analysis

## Enabling Anchor Tracking

Anchor tracking is **disabled by default** for backward compatibility. To enable:

### Method 1: Via Settings File

Add to your `settings.json`:

```json
{
  "enable_anchor_tracking": true,
  "anchor_alpha": 0.25,
  "anchor_adaptive": false,
  "anchor_gap_tolerance": 10,
  "anchor_outlier_mult": 4.0,
  "anchor_percentile_window": 120,
  "anchor_tracker": "simple",
  "anchor_min_warmup": 80
}
```

### Method 2: Via CLI Arguments

When processing videos via CLI, use the anchor tracking flags:

```bash
python main.py video.mp4 --mode 3-stage \
  --anchor-alpha 0.25 \
  --anchor-adaptive \
  --anchor-tracker simple \
  --anchor-gap-tolerance 10
```

**Note**: CLI arguments require `enable_anchor_tracking: true` in settings.

### Method 3: Via GUI (Future)

Settings will be available in the GUI control panel in a future update.

## Quick Start Examples

### Basic Usage (Recommended Defaults)

```bash
# Enable in settings.json first
python main.py video.mp4 --mode 3-stage
```

### High Responsiveness

```bash
python main.py video.mp4 --mode 3-stage \
  --anchor-alpha 0.4 \
  --anchor-adaptive
```

### Maximum Smoothness

```bash
python main.py video.mp4 --mode 3-stage \
  --anchor-alpha 0.15 \
  --anchor-percentile-window 180
```

### Advanced Tracking (Requires BYTETrack)

```bash
pip install bytetrack
python main.py video.mp4 --mode 3-stage \
  --anchor-tracker bytetrack \
  --anchor-adaptive
```

## CLI Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--anchor-alpha` | float | 0.25 | Base smoothing (0=max smooth, 1=no smooth) |
| `--anchor-adaptive` | flag | false | Enable adaptive smoothing |
| `--anchor-gap-tolerance` | int | 10 | Max frames to interpolate |
| `--anchor-outlier-mult` | float | 4.0 | Outlier threshold (MAD multiplier) |
| `--anchor-percentile-window` | int | 120 | Rolling normalization window |
| `--anchor-tracker` | choice | none | Tracker type (none/simple/bytetrack/ocs) |
| `--anchor-min-warmup` | int | 80 | Min warmup frames |

## Interpreting Metrics

After processing, you'll see a log line like:

```
ANCHOR: coverage=0.94 jitter=0.07 idsw=0 outliers=3 range=12.5→87.3
```

- **coverage**: Percentage of frames with valid anchor detection (aim for >0.85)
- **jitter**: Normalized motion jitter (lower is better, <0.10 is good)
- **idsw**: ID switches from tracker (0 is best)
- **outliers**: Count of rejected outlier detections
- **range**: Normalization percentile range in pixels

## Tuning Guide

### If output is too jittery:
- Decrease `--anchor-alpha` (e.g., 0.15-0.20)
- Enable `--anchor-adaptive`
- Increase `--anchor-percentile-window` (e.g., 180)

### If output lags behind motion:
- Increase `--anchor-alpha` (e.g., 0.35-0.45)
- Disable `--anchor-adaptive`

### If coverage is low (<0.7):
- Check YOLO confidence threshold
- Verify video format detection
- Review discarded class list

### If many outliers are detected:
- Increase `--anchor-outlier-mult` (e.g., 5.0-6.0)
- Check for tracking instabilities

## Troubleshooting

### Anchor tracking not working?

1. **Check settings**: Verify `"enable_anchor_tracking": true` in settings.json
2. **Check logs**: Look for "Anchor tracking enhancement enabled" message
3. **Check dependencies**: Ensure numpy and scipy are installed
4. **Test modules**: Run `python tools/anchor_smoke_test.py`

### No improvement in output?

- Anchor tracking works best when the primary anchor is consistently visible
- May not help for videos with frequent occlusions or multi-person scenes
- Try different alpha and adaptive settings
- Compare with and without anchor tracking enabled

### Performance issues?

- Simple tracker adds <1% overhead
- Advanced trackers (BYTETrack/OC-SORT) add ~2-5% if installed
- Disable anchor tracking if not beneficial: `"enable_anchor_tracking": false`

## Testing

Run the included test suites to verify installation:

```bash
# Smoke tests (module functionality)
python tools/anchor_smoke_test.py

# Integration tests (Stage 2 integration)
python tools/test_anchor_integration.py
```

Both should report **all tests passing**.

## Documentation

For detailed information, see:
- `docs/anchor_tracking.md` - Complete system documentation
- Parameter tuning guide
- Known limitations
- Future enhancements (Phase B/C)

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Run the test suites to verify installation
3. Review logs for error messages
4. Check GitHub issues for similar problems

## Future Enhancements

Phase A is complete. Planned for future phases:

- **Phase B**: Pose keypoint fusion, multi-anchor blending
- **Phase C**: Performance optimization (TensorRT, micro-batching)
- Multi-person disambiguation
- GUI settings panel
- Metrics persistence to project files

---

**Backward Compatibility**: When anchor tracking is disabled (default), the system uses the legacy displacement calculation. Your existing workflows are unaffected.
