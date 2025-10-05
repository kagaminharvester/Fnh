"""
TensorRT Engine Manager
Author: FunGen AI System
Version: 1.0.0

Manages TensorRT engine building, caching, and loading:
- Engine cache with hash-based invalidation
- Dynamic profile support
- Precision control (FP16/FP32)
- Graceful fallback on errors
"""

import os
import sys
import hashlib
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class EngineConfig:
    """Configuration for TensorRT engine."""
    model_path: str
    precision: str = "fp16"  # fp16, fp32, or int8
    batch_size: int = 1
    dynamic_shapes: bool = False
    workspace_size: int = 4  # GB


def compute_model_hash(model_path: str, sample_size: int = 4 * 1024 * 1024) -> str:
    """
    Compute hash of model file (first 4MB + file size + modification time).
    
    Args:
        model_path: Path to model file
        sample_size: Number of bytes to hash from start of file
        
    Returns:
        SHA256 hash string
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    hasher = hashlib.sha256()
    
    # Add file size
    file_size = os.path.getsize(model_path)
    hasher.update(str(file_size).encode())
    
    # Add first N MB of file
    with open(model_path, 'rb') as f:
        chunk = f.read(sample_size)
        hasher.update(chunk)
        
    # Add modification time
    mtime = os.path.getmtime(model_path)
    hasher.update(str(int(mtime)).encode())
    
    return hasher.hexdigest()


class TensorRTManager:
    """
    Manages TensorRT engine lifecycle with caching.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize TensorRT manager.
        
        Args:
            cache_dir: Directory for engine cache (default: .fungen_cache/engines)
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".fungen_cache", "engines")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"TensorRT cache directory: {self.cache_dir}")
        
        # Check TensorRT availability
        self.tensorrt_available = self._check_tensorrt()
        
    def _check_tensorrt(self) -> bool:
        """Check if TensorRT is available."""
        try:
            import tensorrt as trt
            self.logger.info(f"TensorRT version: {trt.__version__}")
            return True
        except ImportError:
            self.logger.warning("TensorRT not available")
            return False
            
    def get_engine_path(self, config: EngineConfig) -> Path:
        """
        Get path to cached engine file.
        
        Args:
            config: Engine configuration
            
        Returns:
            Path to engine file
        """
        # Compute model hash
        model_hash = compute_model_hash(config.model_path)
        
        # Include config in hash
        config_str = f"{model_hash}_{config.precision}_b{config.batch_size}_dyn{config.dynamic_shapes}"
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]
        
        # Create subdirectory for this model
        engine_dir = self.cache_dir / config_hash
        engine_dir.mkdir(parents=True, exist_ok=True)
        
        return engine_dir / "model.engine"
        
    def engine_exists(self, config: EngineConfig) -> bool:
        """
        Check if engine exists in cache.
        
        Args:
            config: Engine configuration
            
        Returns:
            True if engine file exists
        """
        engine_path = self.get_engine_path(config)
        return engine_path.exists()
        
    def build_engine(self, config: EngineConfig, force: bool = False) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Build TensorRT engine from model.
        
        Args:
            config: Engine configuration
            force: Force rebuild even if engine exists
            
        Returns:
            Tuple of (success, engine_path, error_message)
        """
        if not self.tensorrt_available:
            return False, None, "TensorRT not available"
            
        engine_path = self.get_engine_path(config)
        
        # Check if already exists
        if engine_path.exists() and not force:
            self.logger.info(f"Engine already exists: {engine_path}")
            return True, str(engine_path), None
            
        self.logger.info(f"Building TensorRT engine for {config.model_path}")
        self.logger.info(f"  Precision: {config.precision}")
        self.logger.info(f"  Batch size: {config.batch_size}")
        self.logger.info(f"  Dynamic shapes: {config.dynamic_shapes}")
        
        try:
            # Use subprocess to isolate engine building
            # This allows us to handle crashes and timeouts gracefully
            import subprocess
            
            # Get the export script from utils
            export_script = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "utils",
                "tensorrt_export_engine_model.py"
            )
            
            if not os.path.exists(export_script):
                return False, None, f"Export script not found: {export_script}"
                
            # Prepare command
            cmd = [
                sys.executable,
                export_script,
                config.model_path,
                str(engine_path.parent)
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            # Parse result
            try:
                output = json.loads(result.stdout)
                if output.get('success'):
                    # Move engine to cache location if needed
                    generated_engine = output.get('engine_file')
                    if generated_engine and os.path.exists(generated_engine):
                        if generated_engine != str(engine_path):
                            import shutil
                            shutil.move(generated_engine, str(engine_path))
                        self.logger.info(f"Engine built successfully: {engine_path}")
                        return True, str(engine_path), None
                    else:
                        return False, None, "Engine file not created"
                else:
                    error = output.get('error', 'Unknown error')
                    self.logger.error(f"Engine build failed: {error}")
                    return False, None, error
            except json.JSONDecodeError:
                # Fallback to checking stderr
                error = result.stderr if result.stderr else "Build failed with no output"
                self.logger.error(f"Engine build failed: {error}")
                return False, None, error
                
        except subprocess.TimeoutExpired:
            error = "Engine build timed out after 10 minutes"
            self.logger.error(error)
            return False, None, error
        except Exception as e:
            error = f"Engine build error: {str(e)}"
            self.logger.error(error)
            return False, None, error
            
    def load_engine(self, config: EngineConfig) -> Tuple[bool, Optional[Any], Optional[str]]:
        """
        Load TensorRT engine from cache.
        
        Args:
            config: Engine configuration
            
        Returns:
            Tuple of (success, engine_object, error_message)
        """
        if not self.tensorrt_available:
            return False, None, "TensorRT not available"
            
        engine_path = self.get_engine_path(config)
        
        if not engine_path.exists():
            return False, None, f"Engine not found: {engine_path}"
            
        try:
            import tensorrt as trt
            
            # Load engine
            self.logger.info(f"Loading TensorRT engine: {engine_path}")
            
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            with open(engine_path, 'rb') as f:
                engine = runtime.deserialize_cuda_engine(f.read())
                
            if engine is None:
                return False, None, "Failed to deserialize engine"
                
            self.logger.info("Engine loaded successfully")
            return True, engine, None
            
        except Exception as e:
            error = f"Failed to load engine: {str(e)}"
            self.logger.error(error)
            return False, None, error
            
    def invalidate_cache(self, config: EngineConfig):
        """
        Invalidate cached engine for given configuration.
        
        Args:
            config: Engine configuration
        """
        engine_path = self.get_engine_path(config)
        if engine_path.exists():
            try:
                engine_path.unlink()
                self.logger.info(f"Invalidated cache: {engine_path}")
            except Exception as e:
                self.logger.error(f"Failed to invalidate cache: {e}")
                
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached engines."""
        info = {
            "cache_dir": str(self.cache_dir),
            "tensorrt_available": self.tensorrt_available,
            "cached_engines": []
        }
        
        if self.cache_dir.exists():
            for engine_dir in self.cache_dir.iterdir():
                if engine_dir.is_dir():
                    engine_file = engine_dir / "model.engine"
                    if engine_file.exists():
                        info["cached_engines"].append({
                            "hash": engine_dir.name,
                            "path": str(engine_file),
                            "size_mb": engine_file.stat().st_size / (1024 * 1024)
                        })
                        
        return info


# Convenience functions

def build_engine(model_path: str, 
                precision: str = "fp16",
                batch_size: int = 1,
                dynamic_shapes: bool = False,
                cache_dir: Optional[str] = None,
                force: bool = False) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Build TensorRT engine (convenience function).
    
    Args:
        model_path: Path to model file
        precision: Precision mode (fp16, fp32, int8)
        batch_size: Batch size
        dynamic_shapes: Enable dynamic shapes
        cache_dir: Cache directory
        force: Force rebuild
        
    Returns:
        Tuple of (success, engine_path, error_message)
    """
    config = EngineConfig(
        model_path=model_path,
        precision=precision,
        batch_size=batch_size,
        dynamic_shapes=dynamic_shapes
    )
    
    manager = TensorRTManager(cache_dir=cache_dir)
    return manager.build_engine(config, force=force)


def load_engine(model_path: str,
               precision: str = "fp16", 
               batch_size: int = 1,
               cache_dir: Optional[str] = None) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Load TensorRT engine (convenience function).
    
    Args:
        model_path: Path to original model file
        precision: Precision mode
        batch_size: Batch size
        cache_dir: Cache directory
        
    Returns:
        Tuple of (success, engine_object, error_message)
    """
    config = EngineConfig(
        model_path=model_path,
        precision=precision,
        batch_size=batch_size
    )
    
    manager = TensorRTManager(cache_dir=cache_dir)
    return manager.load_engine(config)
