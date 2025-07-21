# Image Recognition API Deployment

This directory contains the deployment configuration for the Image Recognition API, redesigned to follow a clean, simple pattern inspired by the reference implementation.

## Structure

```
deploy/
├── Dockerfile          # Multi-stage Docker build for the API
├── cloudbuild.yaml     # Google Cloud Build configuration
├── Makefile           # Deployment automation
├── deploy.sh          # Simple deployment script
├── setup.sh           # GCP environment setup script
└── README.md          # This file
```

## Quick Start

```bash
# 1. Set your project ID
export PROJECT_ID="your-project-id"

# 2. One-time setup (creates Artifact Registry, enables APIs)
./deploy/setup.sh

# 3. Deploy
make deploy

# 4. Test
curl $(gcloud run services describe image-recognition-api --region=us-central1 --project=$PROJECT_ID --format="value(status.url)")/health
```

## Key Features

### 🏗️ Multi-Stage Docker Build
- **Base Stage**: System dependencies and Google Cloud SDK
- **Dependencies Stage**: Python package installation
- **Runtime Stage**: Application code and model setup

### 🚀 Simplified Deployment
- Single service deployment (no separate Triton server)
- Uses local YOLO model execution
- Automatic model downloading from Google Cloud Storage
- Built-in health checks and monitoring

### 🔧 Clean Configuration
- Environment-based configuration
- Proper layer caching for faster builds
- Non-root user for security
- Auto-scaling based on CPU cores

## Setup (One-time)

Before deploying, run the setup script to prepare your GCP environment:

```bash
# Set your project ID
export PROJECT_ID="your-project-id"

# Run setup (enables APIs, creates Artifact Registry, etc.)
./deploy/setup.sh
```

This will:
- Enable required GCP APIs (Cloud Build, Cloud Run, Artifact Registry, Storage)
- Create Artifact Registry repository for Docker images
- Configure Docker authentication
- Create Cloud Storage bucket for models

## Deployment Options

### Option 1: Using Makefile (Recommended)
```bash
# Set required environment variables
export PROJECT_ID="your-project-id"
export REGION="us-central1"  # optional, defaults to us-central1

# Full deployment (build + deploy)
make deploy

# Or run individual steps
make build      # Just build the Docker image
make deploy_cr  # Just deploy to Cloud Run
```

### Option 2: Using Deploy Script
```bash
# Deploy with environment variables
PROJECT_ID="your-project-id" ./deploy/deploy.sh

# Or export and run
export PROJECT_ID="your-project-id"
export REGION="us-central1"  # optional
./deploy/deploy.sh
```

### Option 3: Manual Commands
```bash
# Build image
gcloud builds submit --region=us-central1 --config=deploy/cloudbuild.yaml \
    --project=your-project-id

# Deploy to Cloud Run
gcloud run deploy image-recognition-api \
    --project your-project-id \
    --image us-central1-docker.pkg.dev/your-project-id/image-recognition/image-recognition-api:latest \
    --memory 8Gi --cpu 4 --min-instances 1 \
    --allow-unauthenticated \
    --port 8080 \
    --region us-central1
```

## Configuration

### Environment Variables
- `PROJECT_ID`: Google Cloud Project ID (**required**)
- `REGION`: Deployment region (optional, default: us-central1)
- `MODEL_BUCKET`: GCS bucket for model storage
- `MODEL_PATH`: Path to model in bucket
- `MODEL_FILENAME`: Model file name

### Build Arguments
- `COMMIT_ID`: Git commit hash for image tagging (auto-generated)
- `MODEL_BUCKET`: Override default model bucket
- `MODEL_PATH`: Override default model path
- `MODEL_FILENAME`: Override default model filename

## Requirements

### Prerequisites
- Google Cloud SDK installed and configured
- Docker (for local building)
- Git (for commit hash)
- Make (for Makefile usage)

### IAM Permissions
- Cloud Build Editor
- Cloud Run Admin
- Storage Object Viewer (for model downloads)
- Artifact Registry Writer

## Differences from Original

### ✅ Improvements
- **Simplified Architecture**: Single service instead of API + Triton
- **Faster Deployment**: Fewer build steps and dependencies
- **Better Caching**: Optimized Docker layer structure
- **Cleaner Code**: Following reference implementation patterns
- **Security**: Non-root user and minimal base image
- **Monitoring**: Built-in health checks
- **Environment Agnostic**: No hardcoded dev/prod environments

### 🔄 Changes
- **No Triton Server**: Uses direct YOLO model execution
- **Local Processing**: CPU-optimized inference in the same container
- **Simplified Storage**: Direct model download instead of complex repository setup
- **Standard Ports**: Uses port 8080 consistently
- **Flexible Configuration**: Environment variable driven

## Monitoring and Health

The deployed service includes:
- Health check endpoint: `/health`
- Model information: `/model/info`
- API documentation: `/docs`
- Metrics and logging via Google Cloud Run

## Troubleshooting

### Common Issues
1. **Build Failures**: Check Cloud Build logs in Google Cloud Console
2. **Model Download Issues**: Verify GCS permissions and bucket access
3. **Memory Issues**: Adjust memory allocation in deployment commands
4. **Missing PROJECT_ID**: Ensure PROJECT_ID environment variable is set
5. **Artifact Registry Issues**: Run `./deploy/setup.sh` first

### Debug Commands
```bash
# Check service status
gcloud run services describe image-recognition-api --region=us-central1 --project=your-project-id

# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=image-recognition-api" --limit=50 --project=your-project-id

# Test endpoints
curl https://your-service-url/health
curl https://your-service-url/model/info
```

### Build Troubleshooting
If you encounter Docker build issues:

1. **Check Artifact Registry exists**:
   ```bash
   gcloud artifacts repositories list --project=your-project-id
   ```

2. **Re-run setup**:
   ```bash
   PROJECT_ID=your-project-id ./deploy/setup.sh
   ```

3. **Manual repository creation**:
   ```bash
   gcloud artifacts repositories create image-recognition \
       --repository-format=docker \
       --location=us-central1 \
       --project=your-project-id
   ``` 