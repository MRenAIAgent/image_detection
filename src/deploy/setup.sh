#!/bin/bash

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
REPOSITORY_NAME="image-recognition"

# Validate required environment variables
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: PROJECT_ID environment variable is required"
    echo "Usage: PROJECT_ID=your-project-id ./deploy/setup.sh"
    exit 1
fi

echo "🔧 Setting up GCP environment for Image Recognition API"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Step 1: Enable required APIs
echo "🔌 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    storage-api.googleapis.com \
    --project=$PROJECT_ID

# Step 2: Create Artifact Registry repository
echo "📦 Creating Artifact Registry repository..."
gcloud artifacts repositories create $REPOSITORY_NAME \
    --repository-format=docker \
    --location=$REGION \
    --description="Docker repository for Image Recognition API" \
    --project=$PROJECT_ID || echo "Repository may already exist"

# Step 3: Configure Docker authentication
echo "🔐 Configuring Docker authentication..."
gcloud auth configure-docker $REGION-docker.pkg.dev --project=$PROJECT_ID

# Step 4: Create Cloud Storage bucket for models (optional)
echo "💾 Creating Cloud Storage bucket for models..."
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$PROJECT_ID-image-recognition-models || echo "Bucket may already exist"

echo "✅ Setup completed successfully!"
echo "Next steps:"
echo "  1. Run: PROJECT_ID=$PROJECT_ID make deploy"
echo "  2. Or run: PROJECT_ID=$PROJECT_ID ./deploy/deploy.sh" 