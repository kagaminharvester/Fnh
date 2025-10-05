"""Detection cache framework for FunGen."""
from .detections_cache import (
    compute_video_hash,
    model_hash,
    load_cached_detections,
    store_cached_detections,
    get_cache_dir
)

__all__ = [
    'compute_video_hash',
    'model_hash',
    'load_cached_detections',
    'store_cached_detections',
    'get_cache_dir'
]
