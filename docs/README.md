# FunGen Phase B+C Documentation

This directory contains documentation for the Phase B+C performance and tracking enhancements.

## Available Documentation

### [Performance Pipeline](performance_pipeline.md)

Complete guide to the async pipeline architecture, batching strategies, TensorRT integration, and cache management.

**Topics covered:**
- Async Decode → Inference pipeline
- Micro-batching strategy
- TensorRT engine building and caching
- Detection cache usage
- Performance tuning for different GPUs
- Profiling and metrics

### [Tracking Strategies](tracking_strategies.md)

Guide to multi-object tracking options and their trade-offs.

**Topics covered:**
- BYTETrack integration
- OC-SORT integration
- Simple fallback tracker
- Auto-selection logic
- Tracking parameters and tuning
- Quality metrics and troubleshooting

### [Anchor Tracking](anchor_tracking.md)

Detailed documentation on anchor detection and relative motion projection.

**Topics covered:**
- Anchor detection strategies
- Relative motion computation
- Dual-axis decomposition (primary + secondary)
- Roll inference heuristic
- Integration with existing trackers
- Best practices and troubleshooting

## Quick Start

### Enable Performance Features

```bash
# Use TensorRT for faster inference
python main.py video.mp4 --tensorrt --tensorrt-build

# Use micro-batching
python main.py video.mp4 --batch-size 8

# Reuse cached detections
python main.py video.mp4 --reuse-detections
```

### Enable Tracking Features

```bash
# Use BYTETrack for multi-object tracking
python main.py video.mp4 --multi-tracker bytetrack

# Enable anchor-relative motion
python main.py video.mp4 --anchor-relative

# Generate roll axis output
python main.py video.mp4 --anchor-relative --roll-axis
```

### Enable Profiling

```bash
# Get detailed performance metrics
python main.py video.mp4 --profile-run
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Video Input                          │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│            Async Decode Thread                          │
│  - Reads frames                                         │
│  - Pushes to decode queue                               │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│         Inference Worker (Batching)                     │
│  - Aggregates frames                                    │
│  - Runs YOLO/TensorRT                                   │
│  - Pushes detections                                    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│          Multi-Object Tracking                          │
│  - Assigns persistent IDs                               │
│  - Handles occlusions                                   │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│         Motion Projection                               │
│  - Computes relative motion                             │
│  - Primary/secondary axis                               │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│         Filter Chain Optimizer                          │
│  - Fuses operations                                     │
│  - Vectorized processing                                │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              Funscript Output                           │
└─────────────────────────────────────────────────────────┘
```

## Performance Targets

**Phase B+C Goals:**
- ≥25% FPS improvement over baseline
- Stable memory usage
- Minimal latency increase (<100ms)

**Typical Results (RTX 3080):**
- Baseline: ~45 FPS
- With TensorRT + batching: ~60-70 FPS
- With cache reuse: Instant (skip Stage 1)

## Contributing

When adding new features:

1. Update relevant documentation
2. Add CLI flags if needed
3. Include usage examples
4. Document performance impact
5. Add troubleshooting section

## Related Files

- `main.py` - CLI argument parsing
- `application/tracking/` - Multi-object tracking
- `application/signals/` - Motion projection
- `application/engines/` - TensorRT management
- `application/utils/async_pipeline.py` - Pipeline implementation
- `application/utils/detection_cache.py` - Cache management
- `application/utils/profiler.py` - Profiling system
- `funscript/filters/chain_optimizer.py` - Filter optimization
