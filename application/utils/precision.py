"""
Precision handling abstraction for model inference.

Provides utilities for managing FP16/FP32 precision in PyTorch models.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - precision control disabled")


class PrecisionPolicy:
    """
    Manages precision settings for model inference.
    
    Supports auto-detection based on GPU capabilities, or explicit FP16/FP32 modes.
    """
    
    def __init__(self, mode: str = "auto"):
        """
        Initialize precision policy.
        
        Args:
            mode: One of 'auto', 'fp32', 'fp16'
                - auto: Uses FP16 on Ampere+ GPUs (compute capability >= 7.0), FP32 otherwise
                - fp32: Forces full precision
                - fp16: Forces half precision (requires CUDA)
        """
        if mode not in {"auto", "fp32", "fp16"}:
            logger.warning(f"Unknown precision mode '{mode}', defaulting to 'auto'")
            mode = "auto"
        
        self.mode = mode
        self.use_fp16 = False
        self.use_autocast = False
        
        if not TORCH_AVAILABLE:
            logger.info("PyTorch not available, using default precision")
            return
        
        # Determine actual precision to use
        if mode == "fp32":
            self.use_fp16 = False
            logger.info("Precision: FP32 (forced)")
        elif mode == "fp16":
            if torch.cuda.is_available():
                self.use_fp16 = True
                self.use_autocast = True
                logger.info("Precision: FP16 (forced)")
            else:
                logger.warning("FP16 requested but CUDA not available, falling back to FP32")
                self.use_fp16 = False
        elif mode == "auto":
            if torch.cuda.is_available():
                # Check compute capability for Ampere+ (7.0+)
                try:
                    device = torch.cuda.current_device()
                    capability = torch.cuda.get_device_capability(device)
                    compute_capability = capability[0] + capability[1] / 10.0
                    
                    if compute_capability >= 7.0:
                        self.use_fp16 = True
                        self.use_autocast = True
                        logger.info(f"Precision: FP16 (auto, compute capability {compute_capability})")
                    else:
                        self.use_fp16 = False
                        logger.info(f"Precision: FP32 (auto, compute capability {compute_capability} < 7.0)")
                except Exception as e:
                    logger.warning(f"Failed to detect GPU compute capability: {e}")
                    self.use_fp16 = False
                    logger.info("Precision: FP32 (auto, capability check failed)")
            else:
                self.use_fp16 = False
                logger.info("Precision: FP32 (auto, no CUDA)")
    
    def apply_model(self, model):
        """
        Apply precision settings to a PyTorch model.
        
        Args:
            model: PyTorch model to configure
            
        Returns:
            Configured model
        """
        if not TORCH_AVAILABLE:
            return model
        
        try:
            if self.use_fp16 and torch.cuda.is_available():
                # Convert model to half precision
                model = model.half()
                logger.debug("Model converted to FP16")
        except Exception as e:
            logger.warning(f"Failed to apply FP16 to model, reverting to FP32: {e}")
            self.use_fp16 = False
            self.use_autocast = False
            if hasattr(model, 'float'):
                model = model.float()
        
        return model
    
    def get_autocast_context(self):
        """
        Get autocast context manager for inference.
        
        Returns:
            Context manager for automatic mixed precision (or dummy context if not needed)
        """
        if not TORCH_AVAILABLE:
            return _DummyContext()
        
        if self.use_autocast and torch.cuda.is_available():
            try:
                return torch.cuda.amp.autocast(dtype=torch.float16)
            except Exception as e:
                logger.warning(f"Failed to create autocast context: {e}")
                return _DummyContext()
        else:
            return _DummyContext()


class _DummyContext:
    """Dummy context manager that does nothing."""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
