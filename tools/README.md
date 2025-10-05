# FunGen Tools

This directory contains standalone tools for FunGen development and benchmarking.

## Available Tools

### build_trt_engine.py

Build TensorRT engines from YOLO models for optimized inference.

**Usage:**
```bash
# Build FP16 engine (recommended)
python tools/build_trt_engine.py models/yolo_model.pt --precision fp16

# Build with specific batch size
python tools/build_trt_engine.py models/yolo_model.pt --batch-size 8

# Force rebuild existing engine
python tools/build_trt_engine.py models/yolo_model.pt --force

# Custom cache directory
python tools/build_trt_engine.py models/yolo_model.pt --cache-dir ./my_engines
```

**Requirements:**
- TensorRT installed
- CUDA-compatible GPU
- Ultralytics YOLO

### run_benchmark.py

Run performance benchmarks on test video clips.

**Usage:**
```bash
# Benchmark all videos with default modes
python tools/run_benchmark.py videos/

# Benchmark with specific modes
python tools/run_benchmark.py videos/ --modes 3-stage stage2

# Save results to CSV
python tools/run_benchmark.py videos/ --output results.csv --format csv

# Add extra arguments to benchmark runs
python tools/run_benchmark.py videos/ --extra-args "--tensorrt --batch-size 8"
```

**Output:**
- JSON or CSV file with detailed metrics
- FPS, processing time, errors
- Comparison across modes

## Installation

Tools use the main FunGen environment. No additional setup required.

Make sure you're in the FunGen root directory when running these tools.

## Development

To add new tools:

1. Create Python script in `tools/` directory
2. Add shebang: `#!/usr/bin/env python3`
3. Make executable: `chmod +x tools/your_tool.py`
4. Document usage in this README
