#!/bin/bash

set -e

# Configuration - use environment variables or defaults
PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="image-recognition-api"

# Validate required environment variables
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: PROJECT_ID environment variable is required"
    echo "Usage: PROJECT_ID=your-project-id ./deploy/deploy.sh"
    exit 1
fi

echo "🚀 Starting deployment of Image Recognition API"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Step 1: Build and push Docker image
echo "📦 Building Docker image..."
gcloud builds submit --region=$REGION --config=deploy/cloudbuild.yaml \
    --project=$PROJECT_ID

# Step 2: Deploy to Cloud Run (using latest tag)
echo "🌐 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --project $PROJECT_ID \
    --image $REGION-docker.pkg.dev/$PROJECT_ID/image-recognition/image-recognition-api:latest \
    --memory 8Gi \
    --cpu 4 \
    --min-instances 1 \
    --max-instances 10 \
    --allow-unauthenticated \
    --port 8080 \
    --region $REGION \
    --set-env-vars "ENVIRONMENT=production,LOG_LEVEL=INFO"

# Step 3: Update traffic to latest
echo "🔄 Updating traffic to latest version..."
gcloud --project $PROJECT_ID run services update-traffic $SERVICE_NAME \
    --to-latest --region $REGION

# Step 4: Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region=$REGION --project=$PROJECT_ID \
    --format="value(status.url)")

echo "✅ Deployment completed successfully!"
echo "🌐 Service URL: $SERVICE_URL"
echo "📚 API Docs: $SERVICE_URL/docs"
echo "❤️  Health Check: $SERVICE_URL/health"
echo "📊 Model Info: $SERVICE_URL/model/info" 