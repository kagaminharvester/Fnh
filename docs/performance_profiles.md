# Performance Profiles and Optimization

This document describes FunGen's performance optimization features and configuration options.

## CLI Performance Flags

### `--precision {auto,fp32,fp16}`
Controls the numerical precision used for model inference.

- **auto** (default): Automatically selects FP16 on GPUs with compute capability >= 7.0 (Ampere and newer), otherwise uses FP32
- **fp32**: Forces full 32-bit floating point precision (slower but maximum compatibility)
- **fp16**: Forces half-precision (16-bit) floating point (faster on modern GPUs, requires CUDA)

**Examples:**
```bash
python main.py video.mp4 --precision fp16  # Force FP16
python main.py video.mp4 --precision auto  # Auto-detect best precision
```

### `--batch-size N`
Sets the batch size for processing operations (experimental).

Currently stored for validation only. Future updates will support micro-batching for YOLO inference.

**Default:** 1

### `--reuse-detections`
Attempts to load cached Stage 1 detection results if available.

When enabled, FunGen will check for previously computed detection results based on video and model hashes. If found, Stage 1 processing is skipped, significantly reducing processing time for re-runs.

**Cache location:** `.fungen_cache/detections/<video_hash>/<model_hash>/stage1.msgpack`

### `--profile-run`
Generates a JSON performance profile after processing completes.

Output is saved to `<video>.profile.json` and includes timing information for:
- Stage 1 (Object Detection)
- Stage 2 (Tracking)
- Stage 3 (Oscillation Detection)
- Post-processing operations
- Video decoding

**Example output:**
```json
{
  "stage1": {
    "count": 1,
    "total_seconds": 45.2,
    "avg_seconds": 45.2
  },
  "stage2": {
    "count": 1,
    "total_seconds": 12.8,
    "avg_seconds": 12.8
  }
}
```

### `--profile <name>`
Sets a profile name for logging purposes (stored but not yet used for configuration).

Future versions will support loading preset configurations from JSON profile files.

## Environment Caching

FunGen maintains an environment cache (`.fungen_env_cache.json`) to speed up startup times.

The cache stores:
- Python version
- PyTorch version
- GPU name and driver version
- Hash of requirements files
- Timestamp of last check

**Cache validity:** 24 hours

### Force Full Dependency Check
Set the environment variable to bypass caching:
```bash
export FUNGEN_FORCE_DEP_CHECK=1
python main.py
```

## Logging Configuration

### JSON Log Format
Set the `FUNGEN_LOG_FORMAT` environment variable to output structured JSON logs:

```bash
export FUNGEN_LOG_FORMAT=json
python main.py video.mp4
```

JSON log format includes:
- `timestamp`: ISO 8601 timestamp
- `level`: Log level (INFO, WARNING, ERROR, etc.)
- `message`: Log message
- `module`: Source module name

## Future Roadmap

The following optimization features are planned for future releases:

### INT8 Quantization
- Reduced model size and memory usage
- Faster inference on compatible hardware
- Automatic calibration from sample videos

### TensorRT Integration
- Native TensorRT engine compilation
- Optimized inference on NVIDIA GPUs
- Automatic fallback to PyTorch if TensorRT unavailable

### DALI Pipeline
- GPU-accelerated video decoding
- Preprocessing pipeline optimization
- Reduced CPU-GPU data transfer overhead

### Advanced Batch Processing
- Dynamic batch sizing based on GPU memory
- Multi-stream processing for high-end GPUs
- Automatic workload balancing

## Performance Tips

1. **Use FP16 on modern GPUs**: RTX 30xx and 40xx series benefit significantly from FP16 precision
2. **Enable detection caching**: Use `--reuse-detections` when re-processing the same video
3. **Profile your runs**: Use `--profile-run` to identify bottlenecks
4. **Update drivers**: Ensure GPU drivers are up to date for best performance

## Troubleshooting

### Cache Issues
If you encounter issues with cached data:
```bash
# Force full dependency check
export FUNGEN_FORCE_DEP_CHECK=1
python main.py

# Clear detection cache
rm -rf .fungen_cache/
```

### Precision Errors
If FP16 mode causes errors:
- The system will automatically fall back to FP32
- Check logs for downgrade messages
- Verify GPU supports FP16 (compute capability >= 7.0)

### Performance Regression
If performance decreases after updates:
1. Clear environment cache: `rm .fungen_env_cache.json`
2. Verify GPU drivers are current
3. Check for competing GPU processes
4. Run with `--profile-run` to identify slow stages
