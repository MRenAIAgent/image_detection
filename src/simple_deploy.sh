#!/bin/bash

# Simple GCP Deployment Script for Image Recognition API
# This script deploys step by step to avoid complex Cloud Build issues

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
PROJECT_ID="${PROJECT_ID:-image-recognization-466422}"
REGION="${REGION:-us-central1}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-cpu}"
MODEL_NAME="${MODEL_NAME:-yolov8s}"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${BOLD}${BLUE}=== $1 ===${NC}\n"
}

# Check prerequisites
check_prerequisites() {
    log_header "Checking Prerequisites"
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed"
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check if authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 > /dev/null; then
        log_error "Not authenticated with gcloud. Run: gcloud auth login"
        exit 1
    fi
    
    log_success "All prerequisites satisfied"
}

# Setup GCP project
setup_project() {
    log_header "Setting up GCP Project"
    
    log_info "Setting project to: $PROJECT_ID"
    gcloud config set project "$PROJECT_ID"
    
    log_info "Enabling required APIs..."
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        containerregistry.googleapis.com \
        artifactregistry.googleapis.com
    
    log_success "GCP project setup complete"
}

# Build and push Docker image
build_image() {
    log_header "Building Docker Image"
    
    local commit_sha=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
    local image_name="gcr.io/$PROJECT_ID/image-recognition-api"
    
    log_info "Building image: $image_name:$commit_sha"
    
    # Use simple Cloud Build
    gcloud builds submit \
        --config=deploy/cloudbuild-simple.yaml \
        --substitutions=COMMIT_SHA="$commit_sha" \
        .
    
    log_success "Docker image built and pushed successfully"
    echo "$commit_sha" > .build_hash
}

# Setup model
setup_model() {
    log_header "Setting up Model"
    
    log_info "Setting up $MODEL_NAME model for $DEPLOYMENT_TYPE deployment..."
    
    if [ "$DEPLOYMENT_TYPE" = "cpu" ]; then
        python3 setup_model.py --model "$MODEL_NAME" --model-dir models --optimize-cpu
    else
        python3 setup_model.py --model "$MODEL_NAME" --model-dir models --enable-tensorrt
    fi
    
    log_success "Model setup complete"
}

# Deploy to Cloud Run
deploy_service() {
    log_header "Deploying to Cloud Run"
    
    local commit_sha=$(cat .build_hash 2>/dev/null || echo "latest")
    local image_name="gcr.io/$PROJECT_ID/image-recognition-api:$commit_sha"
    local service_name="image-recognition-api-$(echo $commit_sha | cut -c1-7)"
    
    log_info "Deploying service: $service_name"
    log_info "Using image: $image_name"
    
    # Deploy the service
    gcloud run deploy "$service_name" \
        --image="$image_name" \
        --region="$REGION" \
        --platform=managed \
        --allow-unauthenticated \
        --memory=4Gi \
        --cpu=2 \
        --timeout=300 \
        --concurrency=100 \
        --set-env-vars="LOG_LEVEL=INFO,MAX_BATCH_SIZE=16,CONFIDENCE_THRESHOLD=0.5,BATCH_TIMEOUT=0.1,MAX_QUEUE_SIZE=100"
    
    # Get the service URL
    local api_url=$(gcloud run services describe "$service_name" --region="$REGION" --format="value(status.url)")
    
    log_success "Deployment successful!"
    echo
    echo -e "${BOLD}📡 Service URL:${NC}"
    echo "API URL: $api_url"
    echo "API Docs: $api_url/docs"
    echo "Health Check: $api_url/health"
    echo "Model Info: $api_url/model/info"
    
    # Save URLs to file
    cat > "deployment_urls_${DEPLOYMENT_TYPE}.txt" << EOF
# Image Recognition API Deployment URLs ($(echo $DEPLOYMENT_TYPE | tr '[:lower:]' '[:upper:]'))
# Generated on $(date)

API_URL=$api_url
API_DOCS=$api_url/docs
HEALTH_CHECK=$api_url/health
MODEL_INFO=$api_url/model/info
EOF
    
    log_success "Deployment URLs saved to deployment_urls_${DEPLOYMENT_TYPE}.txt"
}

# Test deployment
test_deployment() {
    log_header "Testing Deployment"
    
    local api_url=$(grep "API_URL=" "deployment_urls_${DEPLOYMENT_TYPE}.txt" | cut -d'=' -f2)
    
    if [ -z "$api_url" ]; then
        log_error "Could not find API URL"
        return 1
    fi
    
    log_info "Testing health endpoint..."
    if curl -sf "$api_url/health" > /dev/null; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi
    
    log_info "Testing model info endpoint..."
    if curl -sf "$api_url/model/info" > /dev/null; then
        log_success "Model info endpoint working"
    else
        log_error "Model info endpoint failed"
        return 1
    fi
    
    log_success "All tests passed!"
}

# Main deployment function
main() {
    echo -e "${BOLD}${GREEN}"
    echo "================================================================"
    echo "        Image Recognition API - Simple GCP Deployment"
    echo "                    Deployment Type: $(echo $DEPLOYMENT_TYPE | tr '[:lower:]' '[:upper:]')"
    echo "                    Model: $MODEL_NAME"
    echo "                    Project: $PROJECT_ID"
    echo "================================================================"
    echo -e "${NC}"
    
    check_prerequisites
    setup_project
    setup_model
    build_image
    deploy_service
    test_deployment
    
    log_success "🎉 $(echo $DEPLOYMENT_TYPE | tr '[:lower:]' '[:upper:]') deployment completed successfully!"
}

# Run main function
main "$@" 