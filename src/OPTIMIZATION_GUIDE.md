# 🚀 Performance Optimization Guide

This guide provides recommendations for optimizing your Image Recognition API deployment for production workloads.

## 📊 Performance Metrics

### Current Performance Baseline
- **Model**: YOLOv8n (6.2MB)
- **Input Size**: 640x640 pixels
- **Inference Time**: ~20-50ms (CPU), ~5-15ms (GPU)
- **Throughput**: ~20-50 requests/second (single instance)
- **Memory Usage**: ~2-4GB per instance

## 🎯 Optimization Strategies

### 1. Model Optimization

#### Use TensorRT Optimization (GPU)
```yaml
# In deploy/cloudbuild.yaml, update Triton config:
optimization {
  execution_accelerators {
    gpu_execution_accelerator : [ {
      name : "tensorrt"
      parameters { key: "precision_mode" value: "FP16" }
      parameters { key: "max_workspace_size_bytes" value: "1073741824" }
    } ]
  }
}
```

#### Model Quantization
```python
# In setup_model.py, add quantization:
model.export(
    format="onnx",
    imgsz=640,
    dynamic=False,
    simplify=True,
    opset=11,
    half=True  # Enable FP16 quantization
)
```

### 2. Batch Processing Optimization

#### Optimal Batch Sizes
```bash
# Environment variables for different workloads:

# High throughput (batch processing)
export MAX_BATCH_SIZE=32
export BATCH_TIMEOUT=0.2

# Low latency (real-time)
export MAX_BATCH_SIZE=4
export BATCH_TIMEOUT=0.05

# Balanced (recommended)
export MAX_BATCH_SIZE=16
export BATCH_TIMEOUT=0.1
```

#### Dynamic Batching Configuration
```protobuf
# In model config (models/yolov8n/config.pbtxt):
dynamic_batching {
  max_queue_delay_microseconds: 50000  # Reduce for lower latency
  preferred_batch_size: [ 4, 8, 16 ]   # Optimize for your workload
  max_queue_size: 64                   # Increase for higher throughput
}
```

### 3. Infrastructure Optimization

#### Cloud Run Configuration
```yaml
# Optimized Cloud Run settings:
resources:
  limits:
    cpu: "4"          # Increase CPU for better performance
    memory: "8Gi"     # More memory for larger batches
  
concurrency: 200      # Increase for higher throughput
min-instances: 2      # Reduce cold starts
max-instances: 50     # Scale for peak load
```

#### Instance Groups
```protobuf
# Multiple instance configuration:
instance_group [
  {
    count: 2          # Multiple instances for redundancy
    kind: KIND_CPU    # Use KIND_GPU if available
    gpus: [ 0 ]      # Specify GPU if using GPU instances
  }
]
```

### 4. Application-Level Optimizations

#### Connection Pooling
```python
# In app/triton_client.py, implement connection pooling:
import asyncio
from contextlib import asynccontextmanager

class TritonClientPool:
    def __init__(self, pool_size=5):
        self.pool = asyncio.Queue(maxsize=pool_size)
        self.pool_size = pool_size
    
    async def initialize(self):
        for _ in range(self.pool_size):
            client = grpcclient.InferenceServerClient(url=self.url)
            await self.pool.put(client)
    
    @asynccontextmanager
    async def get_client(self):
        client = await self.pool.get()
        try:
            yield client
        finally:
            await self.pool.put(client)
```

#### Image Preprocessing Optimization
```python
# Optimized preprocessing pipeline:
def preprocess_batch_optimized(self, images: List[bytes]) -> np.ndarray:
    """Optimized batch preprocessing."""
    processed_images = []
    
    # Use ThreadPoolExecutor for I/O bound operations
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(self._preprocess_single, img) 
                  for img in images]
        
        for future in futures:
            processed_images.append(future.result())
    
    return np.concatenate(processed_images, axis=0)
```

#### Memory Management
```python
# Add memory cleanup in batch processor:
import gc

async def _process_batch(self, batch: List[BatchRequest]):
    try:
        # ... processing logic ...
        pass
    finally:
        # Force garbage collection after batch processing
        gc.collect()
```

### 5. Caching Strategies

#### Response Caching
```python
# Add Redis caching for repeated requests:
import redis
import hashlib

class ResponseCache:
    def __init__(self):
        self.redis_client = redis.Redis(host='redis-host')
    
    def get_cache_key(self, image_bytes: bytes) -> str:
        return hashlib.md5(image_bytes).hexdigest()
    
    async def get_cached_result(self, image_bytes: bytes):
        key = self.get_cache_key(image_bytes)
        cached = self.redis_client.get(key)
        return json.loads(cached) if cached else None
    
    async def cache_result(self, image_bytes: bytes, result: dict):
        key = self.get_cache_key(image_bytes)
        self.redis_client.setex(key, 300, json.dumps(result))  # 5min TTL
```

### 6. Load Balancing

#### Multiple Regions
```bash
# Deploy to multiple regions for global performance:
REGION=us-central1 ./deploy_to_gcp.sh
REGION=europe-west1 ./deploy_to_gcp.sh
REGION=asia-southeast1 ./deploy_to_gcp.sh
```

#### Global Load Balancer
```yaml
# Cloud Load Balancer configuration:
apiVersion: networking.gke.io/v1
kind: ManagedCertificate
metadata:
  name: image-recognition-ssl-cert
spec:
  domains:
    - api.yourdomain.com
---
apiVersion: v1
kind: Service
metadata:
  name: image-recognition-lb
  annotations:
    cloud.google.com/neg: '{"ingress": true}'
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8080
```

## 📈 Monitoring & Profiling

### Key Metrics to Monitor
```python
# Custom metrics to track:
METRICS = {
    "request_duration_seconds": "Request processing time",
    "batch_size_histogram": "Distribution of batch sizes",
    "model_inference_duration": "Model inference time only",
    "queue_size_gauge": "Current queue size",
    "memory_usage_bytes": "Memory consumption",
    "cpu_utilization_percent": "CPU usage",
    "error_rate_percent": "Error rate",
    "cache_hit_ratio": "Cache effectiveness"
}
```

### Performance Testing
```bash
# Load testing with different patterns:

# Sustained load test
python scripts/test_api.py --performance 1000 --concurrent 10

# Spike test  
python scripts/test_api.py --performance 100 --concurrent 50

# Batch processing test
python scripts/test_batch_performance.py --batch-sizes 1,4,8,16,32
```

### Profiling
```python
# Add profiling to identify bottlenecks:
import cProfile
import pstats

def profile_inference():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run inference
    result = await triton_client.infer_batch(images)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)
```

## 🎯 Environment-Specific Optimizations

### Development
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
export MAX_BATCH_SIZE=4
export BATCH_TIMEOUT=0.1
```

### Staging
```bash
export DEBUG=false
export LOG_LEVEL=INFO
export MAX_BATCH_SIZE=16
export BATCH_TIMEOUT=0.1
export MIN_INSTANCES=1
export MAX_INSTANCES=10
```

### Production
```bash
export DEBUG=false
export LOG_LEVEL=WARNING
export MAX_BATCH_SIZE=32
export BATCH_TIMEOUT=0.05
export MIN_INSTANCES=3
export MAX_INSTANCES=100
export ENABLE_RATE_LIMITING=true
export RATE_LIMIT_REQUESTS=1000
```

## 🔧 Advanced Configurations

### Custom Model Pipeline
```python
# Multi-model ensemble for better accuracy:
class EnsemblePredictor:
    def __init__(self):
        self.models = {
            'yolov8n': TritonClient('yolov8n'),
            'yolov8s': TritonClient('yolov8s'),  # More accurate
        }
    
    async def predict_ensemble(self, image_bytes: bytes):
        # Run inference on multiple models
        results = await asyncio.gather(*[
            model.infer_single(image_bytes) 
            for model in self.models.values()
        ])
        
        # Combine results using weighted voting
        return self.combine_predictions(results)
```

### A/B Testing
```python
# Model version A/B testing:
class ModelRouter:
    def __init__(self):
        self.model_weights = {
            'yolov8n_v1': 0.5,  # 50% traffic
            'yolov8n_v2': 0.5,  # 50% traffic
        }
    
    def route_request(self, request_id: str) -> str:
        # Use consistent hashing for user sessions
        hash_value = hash(request_id) % 100
        
        cumulative = 0
        for model, weight in self.model_weights.items():
            cumulative += weight * 100
            if hash_value < cumulative:
                return model
```

## 📊 Performance Benchmarks

### Target Performance Goals

| Metric | Development | Staging | Production |
|--------|------------|---------|------------|
| Latency (p95) | < 200ms | < 100ms | < 50ms |
| Throughput | 10 RPS | 50 RPS | 500 RPS |
| Error Rate | < 5% | < 1% | < 0.1% |
| CPU Usage | < 70% | < 60% | < 50% |
| Memory Usage | < 80% | < 70% | < 60% |

### Optimization Checklist

- [ ] Model optimization (quantization, TensorRT)
- [ ] Batch size tuning
- [ ] Instance scaling configuration
- [ ] Connection pooling
- [ ] Response caching
- [ ] Load balancing setup
- [ ] Monitoring and alerting
- [ ] Performance testing
- [ ] Resource limits optimization
- [ ] Cold start minimization

## 🚀 Next Steps

1. **Implement monitoring** using the metrics above
2. **Run performance tests** to establish baseline
3. **Apply optimizations** incrementally
4. **Monitor impact** of each optimization
5. **Scale resources** based on demand patterns
6. **Consider GPU instances** for higher throughput
7. **Implement caching** for repeated requests
8. **Set up alerts** for performance degradation

---

**Remember**: Optimize based on your specific workload patterns and requirements. Start with monitoring, then apply optimizations incrementally while measuring their impact. 