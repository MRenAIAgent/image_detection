"""
Batch processor for handling multiple image inference requests efficiently
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import uuid

from .config import settings

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Represents a single request in the batch."""
    id: str
    image_bytes: bytes
    filename: str
    timestamp: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=asyncio.Future)

@dataclass
class BatchResponse:
    """Represents a response for a batch request."""
    request_id: str
    filename: str
    detections: List[Dict[str, Any]]
    inference_time: float
    error: Optional[str] = None

class BatchProcessor:
    """
    Processes image inference requests in batches for improved throughput.
    """
    
    def __init__(self, triton_client):
        self.triton_client = triton_client
        self.max_batch_size = settings.max_batch_size
        self.batch_timeout = settings.batch_timeout
        self.max_queue_size = settings.max_queue_size
        
        # Request queue and processing state
        self.request_queue: deque = deque()
        self.processing_lock = asyncio.Lock()
        self.is_processing = False
        self.processor_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.total_requests = 0
        self.total_batches = 0
        self.total_inference_time = 0.0
        
    async def start(self):
        """Start the batch processor."""
        if self.processor_task is None or self.processor_task.done():
            self.processor_task = asyncio.create_task(self._process_batches())
            logger.info("Batch processor started")
    
    async def stop(self):
        """Stop the batch processor."""
        if self.processor_task and not self.processor_task.done():
            self.processor_task.cancel()
            try:
                await self.processor_task
            except asyncio.CancelledError:
                pass
            logger.info("Batch processor stopped")
    
    async def add_request(self, image_bytes: bytes, filename: str) -> BatchResponse:
        """
        Add a new inference request to the batch queue.
        
        Args:
            image_bytes: Raw image bytes
            filename: Original filename
            
        Returns:
            BatchResponse with inference results
        """
        if len(self.request_queue) >= self.max_queue_size:
            raise RuntimeError("Request queue is full")
        
        # Create request
        request_id = str(uuid.uuid4())
        request = BatchRequest(
            id=request_id,
            image_bytes=image_bytes,
            filename=filename
        )
        
        # Add to queue
        async with self.processing_lock:
            self.request_queue.append(request)
            self.total_requests += 1
        
        # Start processor if not running
        await self.start()
        
        # Wait for result
        try:
            result = await request.future
            return result
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            return BatchResponse(
                request_id=request_id,
                filename=filename,
                detections=[],
                inference_time=0.0,
                error=str(e)
            )
    
    async def _process_batches(self):
        """Main batch processing loop."""
        logger.info("Starting batch processing loop")
        
        while True:
            try:
                # Wait for requests or timeout
                await asyncio.sleep(0.01)  # Small delay to prevent busy waiting
                
                # Check if we have requests to process
                if not self.request_queue:
                    continue
                
                # Collect batch
                batch = await self._collect_batch()
                
                if not batch:
                    continue
                
                # Process batch
                await self._process_batch(batch)
                
            except asyncio.CancelledError:
                logger.info("Batch processor cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(1)  # Wait before retrying
    
    async def _collect_batch(self) -> List[BatchRequest]:
        """Collect requests for a batch."""
        batch = []
        batch_start_time = time.time()
        
        async with self.processing_lock:
            # Collect requests up to max batch size or timeout
            while (len(batch) < self.max_batch_size and 
                   self.request_queue and 
                   (time.time() - batch_start_time) < self.batch_timeout):
                
                batch.append(self.request_queue.popleft())
                
                # If we have enough requests, process immediately
                if len(batch) >= self.max_batch_size:
                    break
        
        # Wait for timeout if batch is not full
        if batch and len(batch) < self.max_batch_size:
            remaining_time = self.batch_timeout - (time.time() - batch_start_time)
            if remaining_time > 0:
                await asyncio.sleep(remaining_time)
                
                # Collect any additional requests that arrived
                async with self.processing_lock:
                    while (len(batch) < self.max_batch_size and 
                           self.request_queue):
                        batch.append(self.request_queue.popleft())
        
        return batch
    
    async def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of requests."""
        if not batch:
            return
        
        batch_start_time = time.time()
        logger.debug(f"Processing batch of {len(batch)} requests")
        
        try:
            # Extract image data
            image_batch = [req.image_bytes for req in batch]
            
            # Run inference
            results, inference_time = await self.triton_client.infer_batch(image_batch)
            
            # Create responses
            for i, request in enumerate(batch):
                detections = results[i] if i < len(results) else []
                
                response = BatchResponse(
                    request_id=request.id,
                    filename=request.filename,
                    detections=detections,
                    inference_time=inference_time / len(batch),  # Average time per image
                    error=None
                )
                
                # Set result
                if not request.future.done():
                    request.future.set_result(response)
            
            # Update statistics
            self.total_batches += 1
            self.total_inference_time += inference_time
            
            batch_total_time = time.time() - batch_start_time
            logger.debug(f"Batch processed in {batch_total_time:.3f}s (inference: {inference_time:.3f}s)")
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Set error for all requests in batch
            for request in batch:
                if not request.future.done():
                    error_response = BatchResponse(
                        request_id=request.id,
                        filename=request.filename,
                        detections=[],
                        inference_time=0.0,
                        error=str(e)
                    )
                    request.future.set_result(error_response)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processor statistics."""
        avg_inference_time = (
            self.total_inference_time / self.total_batches 
            if self.total_batches > 0 else 0.0
        )
        
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "queue_size": len(self.request_queue),
            "max_queue_size": self.max_queue_size,
            "max_batch_size": self.max_batch_size,
            "batch_timeout": self.batch_timeout,
            "avg_inference_time": avg_inference_time,
            "is_processing": self.is_processing
        }
    
    def clear_stats(self):
        """Clear statistics."""
        self.total_requests = 0
        self.total_batches = 0
        self.total_inference_time = 0.0 