#!/usr/bin/env python3
"""
TensorRT Engine Builder Tool
Author: FunGen AI System
Version: 1.0.0

Standalone tool for building TensorRT engines from YOLO models.
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine from YOLO model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build FP16 engine
  python build_trt_engine.py model.pt --precision fp16
  
  # Build with specific batch size
  python build_trt_engine.py model.pt --batch-size 8
  
  # Force rebuild existing engine
  python build_trt_engine.py model.pt --force
  
  # Custom cache directory
  python build_trt_engine.py model.pt --cache-dir ./engines
        """
    )
    
    parser.add_argument('model_path', type=str,
                       help='Path to YOLO model (.pt file)')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'int8'],
                       default='fp16', help='Precision mode (default: fp16)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (default: 1)')
    parser.add_argument('--dynamic-shapes', action='store_true',
                       help='Enable dynamic shape support')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Engine cache directory (default: ~/.fungen_cache/engines)')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild even if engine exists')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return 1
        
    # Import engine manager
    try:
        # Add parent directory to path to import from application
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        
        from application.engines.tensorrt_manager import TensorRTManager, EngineConfig
    except ImportError as e:
        logger.error(f"Failed to import TensorRT manager: {e}")
        logger.error("Make sure you're running this from the FunGen directory")
        return 1
    
    # Create configuration
    config = EngineConfig(
        model_path=str(model_path),
        precision=args.precision,
        batch_size=args.batch_size,
        dynamic_shapes=args.dynamic_shapes
    )
    
    # Create manager
    manager = TensorRTManager(cache_dir=args.cache_dir, logger=logger)
    
    # Check if engine already exists
    if manager.engine_exists(config) and not args.force:
        engine_path = manager.get_engine_path(config)
        logger.info(f"Engine already exists: {engine_path}")
        logger.info("Use --force to rebuild")
        return 0
    
    # Build engine
    logger.info("=" * 60)
    logger.info("Building TensorRT Engine")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Precision: {args.precision}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Dynamic shapes: {args.dynamic_shapes}")
    logger.info("=" * 60)
    
    success, engine_path, error = manager.build_engine(config, force=args.force)
    
    if success:
        logger.info("=" * 60)
        logger.info("✓ Engine built successfully!")
        logger.info(f"✓ Location: {engine_path}")
        logger.info("=" * 60)
        
        # Show cache info
        cache_info = manager.get_cache_info()
        logger.info(f"\nCache directory: {cache_info['cache_dir']}")
        logger.info(f"Cached engines: {len(cache_info['cached_engines'])}")
        
        return 0
    else:
        logger.error("=" * 60)
        logger.error("✗ Engine build failed!")
        logger.error(f"✗ Error: {error}")
        logger.error("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
