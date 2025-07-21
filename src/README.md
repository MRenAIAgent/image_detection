# Image Recognition API

A high-performance image recognition API using **YOLOv8** models with support for local development and scalable cloud deployments on Google Cloud Platform.

## 🚀 Quick Start

### Local Development (Recommended for Testing)

```bash
# 1. Clone repository
git clone <your-repo>
cd image_recognization

# 2. Install dependencies
pip3 install -r requirements-local.txt

# 3. Start local API (runs from app directory)
cd app && python3 local_main.py

# 4. Test the API
curl http://localhost:8080/health
```

**🌐 Local Access:**
- **API**: http://localhost:8080
- **Interactive Docs**: http://localhost:8080/docs
- **Health Check**: http://localhost:8080/health

---

## 📦 Deployment Options

### 🏠 Local Development Setup

Perfect for development, testing, and small-scale usage.

#### Prerequisites
- Python 3.8+
- pip3

#### Setup Steps
```bash
# Install local dependencies (simplified, no Triton)
pip3 install -r requirements-local.txt

# Start the local API server
cd app && python3 local_main.py

# The server will automatically download YOLOv8n model if not found
# and start listening on http://localhost:8080
```

#### Local Development Features
- **Direct YOLO inference**: No Triton server needed
- **Automatic model download**: Downloads YOLOv8n.pt if not found
- **Mac/CPU optimized**: Works efficiently on Apple Silicon and Intel Macs
- **Fast startup**: Ready in seconds
- **Hot reload**: Restart quickly during development

### ☁️ CPU Cloud Deployment (GCP)

Production-ready CPU deployment with auto-scaling and monitoring.

#### Prerequisites
- Google Cloud Platform account with billing enabled
- `gcloud` CLI installed and configured
- Docker

#### One-Time Setup
```bash
# 1. Set your project ID
export PROJECT_ID=your-gcp-project-id

# 2. Run setup script (creates resources, enables APIs)
./deploy/setup.sh

# This will:
# - Enable required GCP APIs (Cloud Build, Cloud Run, etc.)
# - Create Artifact Registry repository
# - Configure Docker authentication
# - Create Cloud Storage bucket for models
```

#### Deploy to GCP
```bash
# Option 1: Using Makefile (Recommended)
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1  # optional, defaults to us-central1
make deploy

# Option 2: Using deploy script
export PROJECT_ID=your-gcp-project-id
./deploy/deploy.sh

# Option 3: Using scripts/deploy.sh (legacy)
export PROJECT_ID=your-gcp-project-id
export DEPLOYMENT_TYPE=cpu
scripts/deploy.sh gcp
```

#### Test Your Deployment
```bash
# Get service URL
gcloud run services describe image-recognition-api \
  --region=us-central1 \
  --project=$PROJECT_ID \
  --format="value(status.url)"

# Test health endpoint
curl $(gcloud run services describe image-recognition-api --region=us-central1 --project=$PROJECT_ID --format="value(status.url)")/health
```

---

## 📊 Deployment Comparison

| Deployment Type | Cost | Setup Time | Latency | Throughput | Best For |
|----------------|------|------------|---------|------------|----------|
| **Local Development** | Free | 2 minutes | 100-200ms | Single user | Development, testing |
| **GCP CPU** | ~$15-50/month | 10 minutes | 50-150ms | 20-50 RPS | Production, moderate load |
| **GCP GPU** | ~$150-500/month | 15 minutes | 15-50ms | 100-300 RPS | High-volume production |

### Model Performance Comparison

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **YOLOv8n** | 6MB | Fastest | Good | **Local development** |
| **YOLOv8s** | 22MB | Fast | Better | **CPU production** |
| **YOLOv8m** | 50MB | Medium | High | GPU production |

---

## 🛠 Architecture

### Local Development Architecture
```
┌─────────────────┐    ┌─────────────────────┐
│   FastAPI       │    │   YOLOv8n Model    │
│   Application   │───▶│   (PyTorch)        │
│   localhost:8080│    │   Direct Inference  │
└─────────────────┘    └─────────────────────┘
```

### GCP CPU Architecture
```
┌─────────────────┐    ┌─────────────────────┐
│   FastAPI       │    │   YOLOv8s Model    │
│   Application   │───▶│   (ONNX CPU)        │
│   (Cloud Run)   │    │   Object Detection  │
└─────────────────┘    └─────────────────────┘
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET | API information and status |
| `/health` | GET | Health check and system status |
| `/docs` | GET | Interactive API documentation |
| `/detect` | POST | Single image object detection |
| `/detect/batch` | POST | Batch image processing |
| `/stats` | GET | API performance statistics |
| `/model/info` | GET | Model information and config |

---

## 💡 Usage Examples

### Single Image Detection
```bash
curl -X POST "http://localhost:8080/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-image.jpg"
```

**Response:**
```json
{
  "detections": [
    {
      "class_id": 0,
      "class_name": "person", 
      "confidence": 0.95,
      "bbox": [100, 50, 200, 300]
    }
  ],
  "inference_time": 0.025,
  "filename": "your-image.jpg"
}
```

### Batch Processing
```bash
curl -X POST "http://localhost:8080/detect/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### Python Client
```python
import requests

# Single image detection
with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8080/detect',
        files={'file': f}
    )
    result = response.json()
    print(f"Found {len(result['detections'])} objects")

# Batch processing
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
    ('files', open('image3.jpg', 'rb'))
]
response = requests.post('http://localhost:8080/detect/batch', files=files)
results = response.json()
```

---

## 🧪 Testing & Evaluation

### Quick Local Testing
```bash
# Test with sample images
python3 test_local.py

# Test with specific image
python3 test_with_models.py --single tests/data/person_bicycle.jpg

# API endpoint testing
python3 scripts/test_api.py
```

### Benchmark Testing
```bash
# Download test datasets
python3 download_real_benchmark.py

# Run comprehensive evaluation
python3 tests/real_benchmark/evaluate_real_benchmark.py
```

---

## 🛠 Management Commands

### Local Development
```bash
# Start local API (from project root)
cd app && python3 local_main.py

# Test local API
python3 scripts/test_api.py

# Run with specific model
cd app && MODEL_NAME=yolov8s python3 local_main.py
```

### GCP Deployment Management
```bash
# Check deployment status
gcloud run services describe image-recognition-api \
  --region=us-central1 --project=$PROJECT_ID

# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=image-recognition-api" \
  --limit=50 --project=$PROJECT_ID

# Update deployment
make deploy  # rebuilds and redeploys

# Delete deployment
gcloud run services delete image-recognition-api \
  --region=us-central1 --project=$PROJECT_ID
```

---

## 🚀 Features

- **🎯 Multiple Model Support**: YOLOv8n/s/m/l variants
- **⚡ CPU Optimized**: Efficient inference on CPU hardware
- **📈 Auto Scaling**: Cloud deployments scale from 0 to 50+ instances
- **🌐 Production Ready**: GCP deployment with monitoring and health checks
- **📱 REST API**: FastAPI-based HTTP endpoints for easy integration
- **🧠 Smart Processing**: Intelligent batching for improved throughput
- **☁️ Cloud Native**: Containerized deployment with Docker
- **📊 Real Benchmarking**: Evaluation with official test images

---

## 🆘 Troubleshooting

### Common Local Issues

**"No such file or directory" when running local_main.py:**
```bash
# Make sure you're in the right directory
cd app && python3 local_main.py
# NOT: python3 local_main.py (from root)
```

**Missing dependencies:**
```bash
# Install local requirements
pip3 install -r requirements-local.txt
```

**Port already in use:**
```bash
# Find and kill process using port 8080
lsof -ti:8080 | xargs kill -9
```

**Model download fails:**
```bash
# Check internet connection and try again
# The app will automatically download YOLOv8n.pt on first run
```

### Common GCP Issues

**Setup script fails:**
```bash
# Make sure PROJECT_ID is set
export PROJECT_ID=your-actual-project-id
./deploy/setup.sh
```

**Build fails:**
```bash
# Check if Artifact Registry exists
gcloud artifacts repositories list --project=$PROJECT_ID

# Re-run setup if needed
./deploy/setup.sh
```

**Deployment fails:**
```bash
# Check Cloud Build logs in GCP Console
# Verify billing is enabled on your project
```

**Authentication issues:**
```bash
# Re-authenticate with gcloud
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Support

- **Documentation**: Check the `/docs` endpoint when running locally
- **Issues**: Create an issue on GitHub
- **Local Development**: Use `cd app && python3 local_main.py`
- **GCP Deployment**: Follow the setup script: `./deploy/setup.sh` 