"""
Detection result caching system for Stage 1 outputs.

Provides functions to cache and retrieve detection results based on
video and model hashes to avoid redundant computation.
"""
import os
import hashlib
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_ROOT = ".fungen_cache"
DETECTIONS_CACHE_SUBDIR = "detections"


def get_cache_dir() -> Path:
    """
    Get the root cache directory.
    
    Returns:
        Path to cache directory
    """
    cache_path = Path(CACHE_ROOT)
    cache_path.mkdir(exist_ok=True)
    return cache_path


def compute_video_hash(video_path: str) -> str:
    """
    Compute a hash for a video file based on size, mtime, and partial content.
    
    Uses file size, modification time, and SHA256 of first and last 2MB chunks
    to create a unique identifier for the video.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Hexadecimal hash string representing the video
    """
    try:
        stat = os.stat(video_path)
        size = stat.st_size
        mtime = stat.st_mtime
        
        # Create hash from metadata
        hasher = hashlib.sha256()
        hasher.update(f"size:{size}".encode())
        hasher.update(f"mtime:{mtime}".encode())
        
        # Add content hash from first and last 2MB
        chunk_size = 2 * 1024 * 1024  # 2MB
        
        with open(video_path, 'rb') as f:
            # First chunk
            first_chunk = f.read(chunk_size)
            hasher.update(first_chunk)
            
            # Last chunk (if file is large enough)
            if size > chunk_size:
                f.seek(-min(chunk_size, size - chunk_size), 2)
                last_chunk = f.read(chunk_size)
                hasher.update(last_chunk)
        
        video_hash = hasher.hexdigest()[:16]  # Use first 16 chars for brevity
        return video_hash
        
    except Exception as e:
        logger.warning(f"Failed to compute video hash for {video_path}: {e}")
        # Fallback to filename-based hash
        return hashlib.sha256(os.path.basename(video_path).encode()).hexdigest()[:16]


def model_hash(model_path: str) -> str:
    """
    Compute a hash for a model file.
    
    For large models, only reads the first 4MB to avoid excessive I/O.
    Falls back to filename if file doesn't exist or can't be read.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Hexadecimal hash string representing the model
    """
    try:
        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}, using filename hash")
            return hashlib.sha256(os.path.basename(model_path).encode()).hexdigest()[:16]
        
        hasher = hashlib.sha256()
        max_read = 4 * 1024 * 1024  # 4MB limit
        
        with open(model_path, 'rb') as f:
            chunk = f.read(max_read)
            hasher.update(chunk)
        
        model_hash_val = hasher.hexdigest()[:16]  # Use first 16 chars
        return model_hash_val
        
    except Exception as e:
        logger.warning(f"Failed to compute model hash for {model_path}: {e}")
        return hashlib.sha256(os.path.basename(model_path).encode()).hexdigest()[:16]


def load_cached_detections(video_hash: str, model_hash_val: str) -> Optional[str]:
    """
    Load cached detection results if they exist.
    
    Args:
        video_hash: Hash of the video file
        model_hash_val: Hash of the model file
        
    Returns:
        Path to cached msgpack file if it exists, None otherwise
    """
    cache_dir = get_cache_dir() / DETECTIONS_CACHE_SUBDIR / video_hash / model_hash_val
    cache_file = cache_dir / "stage1.msgpack"
    
    if cache_file.exists():
        logger.info(f"Cache HIT: Found cached detections at {cache_file}")
        return str(cache_file)
    else:
        logger.debug(f"Cache MISS: No cached detections for video={video_hash}, model={model_hash_val}")
        return None


def store_cached_detections(video_hash: str, model_hash_val: str, detections_msgpack_path: str) -> bool:
    """
    Store detection results in the cache.
    
    Creates the necessary directory structure and copies the detection file.
    
    Args:
        video_hash: Hash of the video file
        model_hash_val: Hash of the model file
        detections_msgpack_path: Path to the msgpack file to cache
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cache_dir = get_cache_dir() / DETECTIONS_CACHE_SUBDIR / video_hash / model_hash_val
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / "stage1.msgpack"
        
        # Copy the detection file to cache
        import shutil
        shutil.copy2(detections_msgpack_path, cache_file)
        
        logger.info(f"Cache STORE: Saved detections to {cache_file}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to store cached detections: {e}")
        return False


# TODO: Future enhancements
# - Add cache expiration/cleanup mechanism
# - Add cache size limits
# - Add metadata file with creation timestamp and version info
# - Implement cache validation (verify file integrity)
