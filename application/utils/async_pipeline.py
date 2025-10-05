"""
Async Video Processing Pipeline
Author: FunGen AI System
Version: 1.0.0

Implements micro-batching and async decode→inference→tracking pipeline:
- Decode thread with frame queue
- Inference worker with batching
- Tracking/signal builder
"""

import threading
import queue
import logging
import time
import numpy as np
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from collections import deque


@dataclass
class FramePacket:
    """Frame packet for pipeline."""
    frame_idx: int
    frame: np.ndarray
    timestamp_ms: int


@dataclass
class DetectionPacket:
    """Detection packet for pipeline."""
    frame_idx: int
    detections: List[Dict[str, Any]]
    timestamp_ms: int


class AsyncPipeline:
    """
    Async video processing pipeline with micro-batching.
    """
    
    def __init__(self,
                 batch_size: int = 4,
                 decode_queue_size: int = 16,
                 inference_timeout_ms: int = 12,
                 max_pipeline_lag: int = 24,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize async pipeline.
        
        Args:
            batch_size: Maximum batch size for inference
            decode_queue_size: Size of decode queue
            inference_timeout_ms: Timeout for batch flush
            max_pipeline_lag: Maximum allowed pipeline lag in frames
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.batch_size = batch_size
        self.decode_queue_size = decode_queue_size
        self.inference_timeout_ms = inference_timeout_ms / 1000.0  # Convert to seconds
        self.max_pipeline_lag = max_pipeline_lag
        
        # Queues
        self.decode_queue: queue.Queue = queue.Queue(maxsize=decode_queue_size)
        self.tracking_queue: queue.Queue = queue.Queue(maxsize=decode_queue_size * 2)
        
        # Control
        self.stop_event = threading.Event()
        self.inference_thread: Optional[threading.Thread] = None
        
        # Stats
        self.frames_decoded = 0
        self.frames_inferred = 0
        self.batches_processed = 0
        self.current_lag = 0
        
        # Callbacks
        self.inference_callback: Optional[Callable] = None
        
    def start(self, inference_callback: Callable):
        """
        Start pipeline threads.
        
        Args:
            inference_callback: Function to call for inference (batch of frames) -> detections
        """
        self.inference_callback = inference_callback
        self.stop_event.clear()
        
        # Start inference worker thread
        self.inference_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True,
            name="InferenceWorker"
        )
        self.inference_thread.start()
        
        self.logger.info(f"Pipeline started (batch_size={self.batch_size}, queue_size={self.decode_queue_size})")
        
    def stop(self):
        """Stop pipeline threads."""
        self.stop_event.set()
        
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=5.0)
            
        self.logger.info(f"Pipeline stopped (decoded={self.frames_decoded}, inferred={self.frames_inferred}, batches={self.batches_processed})")
        
    def push_frame(self, frame_idx: int, frame: np.ndarray, timestamp_ms: int):
        """
        Push frame to decode queue.
        
        Args:
            frame_idx: Frame index
            frame: Frame data
            timestamp_ms: Timestamp
        """
        packet = FramePacket(frame_idx=frame_idx, frame=frame, timestamp_ms=timestamp_ms)
        
        try:
            self.decode_queue.put(packet, timeout=1.0)
            self.frames_decoded += 1
            
            # Check pipeline lag
            self.current_lag = self.decode_queue.qsize()
            if self.current_lag > self.max_pipeline_lag:
                self.logger.warning(f"Pipeline lag exceeded: {self.current_lag} frames")
                # Could reduce batch size or flush queue here
                
        except queue.Full:
            self.logger.warning("Decode queue full, dropping frame")
            
    def pop_detections(self, timeout: float = 0.1) -> Optional[DetectionPacket]:
        """
        Pop detections from tracking queue.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Detection packet or None
        """
        try:
            return self.tracking_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def _inference_worker(self):
        """Inference worker thread (batching + inference)."""
        batch: List[FramePacket] = []
        last_batch_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                # Try to get frame with short timeout
                packet = self.decode_queue.get(timeout=0.01)
                batch.append(packet)
                
                # Check if batch is ready or timeout exceeded
                should_flush = (
                    len(batch) >= self.batch_size or
                    (time.time() - last_batch_time) > self.inference_timeout_ms
                )
                
                if should_flush and batch:
                    self._process_batch(batch)
                    batch = []
                    last_batch_time = time.time()
                    
            except queue.Empty:
                # Flush partial batch on timeout
                if batch and (time.time() - last_batch_time) > self.inference_timeout_ms:
                    self._process_batch(batch)
                    batch = []
                    last_batch_time = time.time()
                continue
                
        # Process remaining batch
        if batch:
            self._process_batch(batch)
            
    def _process_batch(self, batch: List[FramePacket]):
        """
        Process a batch of frames.
        
        Args:
            batch: List of frame packets
        """
        if not self.inference_callback:
            return
            
        try:
            # Extract frames
            frames = [p.frame for p in batch]
            
            # Run inference callback
            batch_detections = self.inference_callback(frames)
            
            # Push detections to tracking queue
            for i, packet in enumerate(batch):
                detections = batch_detections[i] if i < len(batch_detections) else []
                
                detection_packet = DetectionPacket(
                    frame_idx=packet.frame_idx,
                    detections=detections,
                    timestamp_ms=packet.timestamp_ms
                )
                
                try:
                    self.tracking_queue.put(detection_packet, timeout=1.0)
                    self.frames_inferred += 1
                except queue.Full:
                    self.logger.warning("Tracking queue full, dropping detections")
                    
            self.batches_processed += 1
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'frames_decoded': self.frames_decoded,
            'frames_inferred': self.frames_inferred,
            'batches_processed': self.batches_processed,
            'current_lag': self.current_lag,
            'decode_queue_size': self.decode_queue.qsize(),
            'tracking_queue_size': self.tracking_queue.qsize(),
            'avg_batch_size': self.frames_inferred / self.batches_processed if self.batches_processed > 0 else 0
        }


class PipelineManager:
    """Manages pipeline lifecycle."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize pipeline manager.
        
        Args:
            config: Pipeline configuration
            logger: Optional logger
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config
        self.pipeline: Optional[AsyncPipeline] = None
        
    def create_pipeline(self, inference_callback: Callable) -> AsyncPipeline:
        """
        Create and start pipeline.
        
        Args:
            inference_callback: Inference function
            
        Returns:
            AsyncPipeline instance
        """
        # Determine batch size based on device
        batch_size = self.config.get('batch_size')
        if batch_size is None:
            # Auto-determine based on CUDA availability
            try:
                import torch
                batch_size = 4 if torch.cuda.is_available() else 1
            except ImportError:
                batch_size = 1
                
        self.pipeline = AsyncPipeline(
            batch_size=batch_size,
            decode_queue_size=self.config.get('decode_queue_size', 16),
            inference_timeout_ms=self.config.get('inference_timeout_ms', 12),
            max_pipeline_lag=self.config.get('max_pipeline_lag', 24),
            logger=self.logger
        )
        
        self.pipeline.start(inference_callback)
        return self.pipeline
        
    def shutdown(self):
        """Shutdown pipeline."""
        if self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
