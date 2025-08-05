#!/bin/bash

set -e

# Configuration - use environment variables or defaults
PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="text-translation-api"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-cpu}"  # cpu or gpu

# Validate required environment variables
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: PROJECT_ID environment variable is required"
    echo "Usage: PROJECT_ID=your-project-id ./deploy/deploy.sh"
    echo "Optional: DEPLOYMENT_TYPE=gpu ./deploy/deploy.sh (for GPU deployment)"
    exit 1
fi

echo "🚀 Starting deployment of Text Translation API"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Deployment Type: $DEPLOYMENT_TYPE"

# Determine build config and service name based on deployment type
if [ "$DEPLOYMENT_TYPE" = "gpu" ]; then
    BUILD_CONFIG="deploy/cloudbuild-gpu.yaml"
    SERVICE_NAME="text-translation-api-gpu"
    IMAGE_NAME="text-translation-api-gpu"
    MEMORY="16Gi"
    CPU="4"
    MIN_INSTANCES="0"
    MAX_INSTANCES="5"
    ENV_VARS="ENVIRONMENT=production,LOG_LEVEL=INFO,DEVICE=cuda,QUANTIZATION_TYPE=fp16"
else
    BUILD_CONFIG="deploy/cloudbuild.yaml"
    SERVICE_NAME="text-translation-api"
    IMAGE_NAME="text-translation-api"
    MEMORY="8Gi"
    CPU="4"
    MIN_INSTANCES="1"
    MAX_INSTANCES="10"
    ENV_VARS="ENVIRONMENT=production,LOG_LEVEL=INFO,DEVICE=cpu,QUANTIZATION_TYPE=int8"
fi

# Step 1: Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    --project=$PROJECT_ID

# Step 2: Create Artifact Registry repository if it doesn't exist
echo "📦 Setting up Artifact Registry..."
gcloud artifacts repositories create text-translation \
    --repository-format=docker \
    --location=$REGION \
    --project=$PROJECT_ID \
    --quiet || echo "Repository already exists"

# Step 3: Build and push Docker image
echo "📦 Building Docker image..."
gcloud builds submit --region=$REGION --config=$BUILD_CONFIG \
    --project=$PROJECT_ID

# Step 4: Deploy to Cloud Run
echo "🌐 Deploying to Cloud Run..."
if [ "$DEPLOYMENT_TYPE" = "gpu" ]; then
    # GPU deployment with special configuration
    gcloud run deploy $SERVICE_NAME \
        --project $PROJECT_ID \
        --image $REGION-docker.pkg.dev/$PROJECT_ID/text-translation/$IMAGE_NAME:latest \
        --memory $MEMORY \
        --cpu $CPU \
        --min-instances $MIN_INSTANCES \
        --max-instances $MAX_INSTANCES \
        --allow-unauthenticated \
        --port 8080 \
        --region $REGION \
        --timeout 900 \
        --concurrency 100 \
        --set-env-vars "$ENV_VARS" \
        --add-cloudsql-instances="" \
        --cpu-boost \
        --execution-environment gen2
else
    # CPU deployment
    gcloud run deploy $SERVICE_NAME \
        --project $PROJECT_ID \
        --image $REGION-docker.pkg.dev/$PROJECT_ID/text-translation/$IMAGE_NAME:latest \
        --memory $MEMORY \
        --cpu $CPU \
        --min-instances $MIN_INSTANCES \
        --max-instances $MAX_INSTANCES \
        --allow-unauthenticated \
        --port 8080 \
        --region $REGION \
        --timeout 600 \
        --concurrency 200 \
        --set-env-vars "$ENV_VARS"
fi

# Step 5: Update traffic to latest
echo "🔄 Updating traffic to latest version..."
gcloud --project $PROJECT_ID run services update-traffic $SERVICE_NAME \
    --to-latest --region $REGION

# Step 6: Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region=$REGION --project=$PROJECT_ID \
    --format="value(status.url)")

echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: $SERVICE_URL"
echo "📚 API Docs: $SERVICE_URL/docs"
echo "❤️  Health Check: $SERVICE_URL/health"
echo "🌍 Supported Languages: $SERVICE_URL/languages"
echo "📊 Model Info: $SERVICE_URL/model/info"
echo ""
echo "🧪 Test the API:"
echo "curl -X POST \"$SERVICE_URL/translate\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\": \"Hello, world!\", \"source_language\": \"english\", \"target_language\": \"spanish\"}'"