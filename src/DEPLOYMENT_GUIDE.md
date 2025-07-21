# 🚀 Image Recognition API - GCP Deployment Guide

This guide will walk you through deploying your Image Recognition API to Google Cloud Platform in just a few simple steps.

## 📋 Prerequisites

Before you begin, ensure you have:

1. **Google Cloud Platform Account** with billing enabled
2. **gcloud CLI** installed and configured
3. **Git** installed
4. **curl** (usually pre-installed)

### Install gcloud CLI

If you haven't installed the gcloud CLI:

```bash
# macOS (using Homebrew)
brew install google-cloud-sdk

# Or download from: https://cloud.google.com/sdk/docs/install
```

## 🎯 Quick Deployment (Recommended)

### Step 1: Set Your Project ID
```bash
export PROJECT_ID=your-gcp-project-id
```

### Step 2: Run the Deployment Script
```bash
./deploy_to_gcp.sh
```

That's it! The script will:
- ✅ Validate your GCP setup
- ✅ Enable required APIs
- ✅ Set up permissions
- ✅ Deploy the application
- ✅ Provide you with the deployment URLs

## 🔧 Manual Deployment (Advanced)

If you prefer more control, you can use the detailed deployment script:

```bash
# Deploy locally first (optional)
./scripts/deploy.sh local

# Deploy to GCP
PROJECT_ID=your-project-id ./scripts/deploy.sh gcp

# Test deployment
./scripts/deploy.sh test https://your-api-url

# Check status
./scripts/deploy.sh status

# Clean up when done
PROJECT_ID=your-project-id ./scripts/deploy.sh cleanup gcp
```

## 🏗️ What Gets Deployed

The deployment creates the following resources in your GCP project:

### Cloud Run Services
- **Triton Server** (`triton-server-*`): NVIDIA Triton Inference Server running YOLOv8n
- **FastAPI Client** (`image-recognition-api-*`): REST API for image recognition

### Cloud Storage
- **Models Bucket**: Stores the YOLOv8n ONNX model
- **Build Artifacts**: Stores deployment artifacts

### Cloud Scheduler
- **Health Check Job**: Monitors API health every 5 minutes

### Container Registry
- **Docker Images**: Built and stored for the FastAPI client

## 📡 API Endpoints

Once deployed, your API will have these endpoints:

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |
| `/detect` | POST | Single image detection |
| `/detect/batch` | POST | Batch image detection |
| `/stats` | GET | API statistics |
| `/model/info` | GET | Model information |

## 🧪 Testing Your Deployment

### Health Check
```bash
curl https://your-api-url/health
```

### Single Image Detection
```bash
curl -X POST "https://your-api-url/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-image.jpg"
```

### Comprehensive Testing
```bash
python scripts/test_api.py --url https://your-api-url
```

## 📊 Monitoring

### View Logs
```bash
# API logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=image-recognition-api-*"

# Triton logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=triton-server-*"
```

### Service Status
```bash
gcloud run services list --region=us-central1
```

### Build History
```bash
gcloud builds list
```

## 💰 Cost Optimization

The deployment is configured for cost optimization:

- **CPU-only inference** (no GPU costs)
- **Minimum instances**: 1 (can scale to 0 when idle)
- **Right-sized resources**: Balanced CPU/memory allocation
- **Efficient batching**: Reduces per-request costs

### Estimated Monthly Costs
- **Cloud Run**: ~$10-50/month (depending on usage)
- **Cloud Storage**: ~$1-5/month
- **Cloud Build**: ~$0-10/month
- **Total**: ~$11-65/month for moderate usage

## 🔧 Configuration

### Environment Variables

You can customize the deployment by setting these environment variables:

```bash
# Required
export PROJECT_ID=your-gcp-project-id

# Optional
export REGION=us-central1                    # GCP region
export CONFIDENCE_THRESHOLD=0.5              # Detection confidence threshold
export MAX_BATCH_SIZE=16                     # Maximum batch size
export BATCH_TIMEOUT=0.1                     # Batch timeout in seconds
```

### Model Configuration

The deployment uses YOLOv8n by default. To use a different model:

1. Modify `setup_model.py`
2. Update `app/config.py` 
3. Redeploy

## 🛠️ Troubleshooting

### Common Issues

#### 1. Authentication Error
```bash
# Re-authenticate
gcloud auth login
gcloud auth application-default login
```

#### 2. Billing Not Enabled
- Visit [GCP Billing Console](https://console.cloud.google.com/billing)
- Enable billing for your project

#### 3. API Not Enabled
```bash
# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
```

#### 4. Permission Denied
```bash
# Check your permissions
gcloud projects get-iam-policy $PROJECT_ID
```

#### 5. Build Timeout
- Increase timeout in `deploy/cloudbuild.yaml`
- Check build logs for specific errors

### Getting Help

1. **Check build logs**: `gcloud builds log BUILD_ID`
2. **Check service logs**: `gcloud logging read "resource.type=cloud_run_revision"`
3. **Test locally first**: `./scripts/deploy.sh local`
4. **Validate model setup**: `python setup_model.py --validate-only`

## 🧹 Cleanup

To remove all deployed resources:

```bash
PROJECT_ID=your-project-id ./deploy_to_gcp.sh cleanup
```

This will delete:
- Cloud Run services
- Cloud Scheduler jobs  
- Container images

**Note**: Storage buckets are preserved to prevent data loss.

## 📚 Additional Resources

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Triton Inference Server](https://github.com/triton-inference-server/server)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review the deployment logs
3. Test the local deployment first
4. Open an issue with detailed error messages

---

**Happy Deploying! 🎉** 