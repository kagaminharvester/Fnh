# Performance Pipeline Guide

## Overview

FunGen's Phase B+C performance pipeline implements advanced optimizations for video processing and funscript generation on modern GPUs. This document describes the async pipeline architecture, batching strategies, TensorRT integration, and cache management.

## Architecture

### Async Decode → Inference Pipeline

The pipeline consists of three main stages operating asynchronously:

```
Decode Thread → Inference Worker → Tracking/Signal Builder
     ↓               ↓                    ↓
 Frame Queue    Detection Queue    Funscript Actions
```

#### 1. Decode Thread
- Reads video frames using OpenCV/FFmpeg
- Pushes frames to `decode_queue` with frame indices
- Uses pinned memory where available for faster GPU transfer
- Configurable queue size (default: 16 frames)

#### 2. Inference Worker
- Consumes frames from decode queue
- Aggregates up to `--batch-size` frames (default: 4)
- Flushes batch on timeout (default: 12ms) to maintain low latency
- Runs YOLO/TensorRT inference
- Pushes per-frame detections to tracking queue

#### 3. Tracking/Signal Builder
- Consumes detections asynchronously
- Applies multi-object tracking
- Computes motion projections
- Builds funscript actions

### Micro-Batching Strategy

Micro-batching improves GPU utilization while maintaining low latency:

- **Batch Size**: Default 4 on CUDA, 1 on CPU
  - Configurable via `--batch-size N`
  - Automatically adjusts if pipeline lag exceeds threshold
  
- **Timeout Flush**: Default 12ms
  - Prevents stale batches during scene transitions
  - Configurable via `--inference-timeout-ms`
  
- **Pipeline Lag Safety**: Default 24 frames max
  - If exceeded, reduces batch size or flushes decode queue
  - Configurable via `--max-pipeline-lag`

## TensorRT Integration

### Engine Building

TensorRT engines are compiled versions of YOLO models optimized for specific hardware:

```bash
# Build engine during analysis
python main.py video.mp4 --tensorrt --tensorrt-build

# Use existing engine
python main.py video.mp4 --tensorrt
```

### Engine Cache

Engines are cached based on model hash:

```
~/.fungen_cache/engines/
  └── <model_hash_precision_batch>/
      └── model.engine
```

**Hash computation**: SHA256 of (first 4MB + file size + modification time)

**Cache invalidation**: Automatic when model changes

### Precision Modes

- **FP16** (default): ~2x speedup, minimal accuracy loss
- **FP32**: Full precision, reference performance
- **INT8**: (Future) ~4x speedup, requires calibration

### Fallback Strategy

1. Check if TensorRT available
2. Check if engine exists in cache
3. If `--tensorrt-build`: Build engine
4. If build fails > 3 times: Fall back to PyTorch
5. Log fallback reason

## Detection Cache

### Purpose

Reuse Stage 1 detection results across multiple analyses:

- Skip expensive inference when video unchanged
- Enables rapid filter experimentation
- Reduces processing time for iterative workflows

### Cache Structure

```
output_dir/
  └── video_name.detections.msgpack  # Compressed detection cache
```

### Usage

```bash
# Generate and cache detections
python main.py video.mp4 --mode 3-stage

# Reuse cached detections
python main.py video.mp4 --mode 3-stage --reuse-detections
```

### Invalidation

Cache invalidated when:
- Model hash changes
- Video file modified
- Funscript metadata version changes

## Performance Tuning

### GPU Optimization

**For RTX 30xx series:**
```bash
python main.py video.mp4 \
  --tensorrt --tensorrt-build \
  --batch-size 8 \
  --decode-queue-size 24
```

**For RTX 40xx series:**
```bash
python main.py video.mp4 \
  --tensorrt --tensorrt-build \
  --batch-size 16 \
  --decode-queue-size 32
```

**For GTX/RTX 20xx:**
```bash
python main.py video.mp4 \
  --batch-size 4 \
  --decode-queue-size 16
```

### Memory Considerations

- **Batch size**: Higher = more VRAM, better throughput
- **Queue size**: Higher = more system RAM, smoother pipeline
- **TensorRT**: +500-800MB VRAM for engine

### CPU Mode

```bash
python main.py video.mp4 \
  --batch-size 1 \
  --decode-queue-size 4
```

## Profiling

Enable detailed performance profiling:

```bash
python main.py video.mp4 --profile-run
```

Output includes:
- Per-stage timing (decode, inference, tracking, filters)
- GPU utilization and peak VRAM
- Throughput (FPS)
- Filter optimization metrics

## Troubleshooting

### Low FPS

1. Check GPU utilization: Should be >80%
2. Increase batch size if VRAM available
3. Enable TensorRT if not already
4. Check for CPU bottlenecks in tracking/signal processing

### High Memory Usage

1. Reduce batch size
2. Reduce decode queue size
3. Disable detection cache if not needed

### Pipeline Lag

If `max_pipeline_lag` warnings appear:
1. Reduce batch size
2. Increase inference timeout
3. Check for slow disk I/O

### TensorRT Build Failures

1. Check CUDA version compatibility
2. Ensure sufficient disk space (~2GB)
3. Check logs for specific error
4. System will auto-fallback to PyTorch

## Best Practices

1. **First Run**: Use `--tensorrt-build` to generate engine
2. **Subsequent Runs**: Just use `--tensorrt` to reuse engine
3. **Experimentation**: Use `--reuse-detections` to skip inference
4. **Batch Processing**: Use larger batch sizes for offline processing
5. **Real-time**: Use smaller batch sizes and queues for lower latency

## Performance Targets

**Phase B+C Goals:**
- ≥25% FPS improvement vs baseline (PyTorch FP16 single-frame)
- Stable memory usage (no unbounded growth)
- Minimal latency increase (<100ms)

**Typical Results (RTX 3080):**
- Baseline: ~45 FPS
- With TensorRT + batching: ~60-70 FPS
- With cache reuse: Instant (skip Stage 1)
