"""
Detection Cache Manager
Author: FunGen AI System
Version: 1.0.0

Manages caching of detection results for reuse:
- Hash-based invalidation
- Compressed storage
- Metadata versioning
"""

import os
import hashlib
import logging
import msgpack
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class CacheMetadata:
    """Metadata for detection cache."""
    model_hash: str
    video_hash: str
    version: str = "1.0"
    total_frames: int = 0
    created_timestamp: float = 0.0


class DetectionCache:
    """
    Manages detection cache with hash-based invalidation.
    """
    
    CACHE_VERSION = "1.0"
    
    def __init__(self, 
                 cache_dir: Optional[Path] = None,
                 enabled: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize detection cache.
        
        Args:
            cache_dir: Cache directory (default: alongside video)
            enabled: Enable caching
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.cache_dir = cache_dir
        self.enabled = enabled
        
    def compute_video_hash(self, video_path: Path) -> str:
        """
        Compute hash of video file.
        
        Args:
            video_path: Path to video
            
        Returns:
            SHA256 hash string
        """
        hasher = hashlib.sha256()
        
        # Add file size
        file_size = os.path.getsize(video_path)
        hasher.update(str(file_size).encode())
        
        # Add modification time
        mtime = os.path.getmtime(video_path)
        hasher.update(str(int(mtime)).encode())
        
        # Add first 1MB of file
        with open(video_path, 'rb') as f:
            chunk = f.read(1024 * 1024)
            hasher.update(chunk)
            
        return hasher.hexdigest()
        
    def compute_model_hash(self, model_path: Path) -> str:
        """
        Compute hash of model file.
        
        Args:
            model_path: Path to model
            
        Returns:
            SHA256 hash string
        """
        if not model_path.exists():
            return "unknown"
            
        hasher = hashlib.sha256()
        
        # Add file size
        file_size = os.path.getsize(model_path)
        hasher.update(str(file_size).encode())
        
        # Add modification time
        mtime = os.path.getmtime(model_path)
        hasher.update(str(int(mtime)).encode())
        
        return hasher.hexdigest()
        
    def get_cache_path(self, video_path: Path) -> Path:
        """
        Get cache file path for video.
        
        Args:
            video_path: Path to video
            
        Returns:
            Path to cache file
        """
        if self.cache_dir:
            cache_dir = self.cache_dir
        else:
            cache_dir = video_path.parent
            
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{video_path.stem}.detections.msgpack"
        
    def check_cache_valid(self, 
                         video_path: Path,
                         model_path: Path,
                         cache_path: Optional[Path] = None) -> bool:
        """
        Check if cache is valid for given video and model.
        
        Args:
            video_path: Path to video
            model_path: Path to model
            cache_path: Optional cache path (default: auto-compute)
            
        Returns:
            True if cache is valid
        """
        if not self.enabled:
            return False
            
        if cache_path is None:
            cache_path = self.get_cache_path(video_path)
            
        if not cache_path.exists():
            return False
            
        try:
            # Load metadata
            with open(cache_path, 'rb') as f:
                data = msgpack.unpack(f)
                
            metadata = data.get('metadata', {})
            
            # Check version
            if metadata.get('version') != self.CACHE_VERSION:
                self.logger.info("Cache version mismatch")
                return False
                
            # Check video hash
            video_hash = self.compute_video_hash(video_path)
            if metadata.get('video_hash') != video_hash:
                self.logger.info("Video hash mismatch")
                return False
                
            # Check model hash
            model_hash = self.compute_model_hash(model_path)
            if metadata.get('model_hash') != model_hash:
                self.logger.info("Model hash mismatch")
                return False
                
            self.logger.info("Detection cache is valid")
            return True
            
        except Exception as e:
            self.logger.warning(f"Cache validation error: {e}")
            return False
            
    def load_cache(self, cache_path: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Load detections from cache.
        
        Args:
            cache_path: Path to cache file
            
        Returns:
            List of frame detections or None
        """
        if not self.enabled or not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'rb') as f:
                data = msgpack.unpack(f)
                
            detections = data.get('detections', [])
            self.logger.info(f"Loaded {len(detections)} frames from cache")
            return detections
            
        except Exception as e:
            self.logger.error(f"Failed to load cache: {e}")
            return None
            
    def save_cache(self,
                  video_path: Path,
                  model_path: Path,
                  detections: List[Dict[str, Any]],
                  cache_path: Optional[Path] = None):
        """
        Save detections to cache.
        
        Args:
            video_path: Path to video
            model_path: Path to model
            detections: List of frame detections
            cache_path: Optional cache path (default: auto-compute)
        """
        if not self.enabled:
            return
            
        if cache_path is None:
            cache_path = self.get_cache_path(video_path)
            
        try:
            import time
            
            # Create metadata
            metadata = CacheMetadata(
                model_hash=self.compute_model_hash(model_path),
                video_hash=self.compute_video_hash(video_path),
                version=self.CACHE_VERSION,
                total_frames=len(detections),
                created_timestamp=time.time()
            )
            
            # Pack data
            data = {
                'metadata': asdict(metadata),
                'detections': detections
            }
            
            # Save to file
            with open(cache_path, 'wb') as f:
                msgpack.pack(data, f)
                
            self.logger.info(f"Saved {len(detections)} frames to cache: {cache_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")
            
    def invalidate_cache(self, video_path: Path):
        """
        Invalidate cache for video.
        
        Args:
            video_path: Path to video
        """
        cache_path = self.get_cache_path(video_path)
        if cache_path.exists():
            try:
                cache_path.unlink()
                self.logger.info(f"Invalidated cache: {cache_path}")
            except Exception as e:
                self.logger.error(f"Failed to invalidate cache: {e}")


class CacheManager:
    """High-level cache management."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config
        
        # Determine if caching is enabled
        self.enabled = not config.get('no_cache', False)
        
        # Create cache instance
        self.cache = DetectionCache(
            enabled=self.enabled,
            logger=self.logger
        )
        
    def should_use_cache(self, video_path: Path, model_path: Path) -> bool:
        """
        Check if cache should be used.
        
        Args:
            video_path: Path to video
            model_path: Path to model
            
        Returns:
            True if cache should be used
        """
        if not self.enabled:
            return False
            
        if not self.config.get('reuse_detections', False):
            return False
            
        return self.cache.check_cache_valid(video_path, model_path)
        
    def load_or_compute(self,
                       video_path: Path,
                       model_path: Path,
                       compute_func: callable) -> List[Dict[str, Any]]:
        """
        Load from cache or compute detections.
        
        Args:
            video_path: Path to video
            model_path: Path to model
            compute_func: Function to compute detections if cache miss
            
        Returns:
            List of frame detections
        """
        # Check if we should use cache
        if self.should_use_cache(video_path, model_path):
            cache_path = self.cache.get_cache_path(video_path)
            detections = self.cache.load_cache(cache_path)
            
            if detections is not None:
                self.logger.info("Using cached detections")
                return detections
                
        # Cache miss - compute
        self.logger.info("Computing detections (cache miss)")
        detections = compute_func()
        
        # Save to cache if enabled
        if self.enabled:
            self.cache.save_cache(video_path, model_path, detections)
            
        return detections
