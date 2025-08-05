#!/bin/bash

# Gemini Flash Translation API - Simple Deployment Script
# This script handles the complete deployment process for Google Cloud Run

set -e  # Exit on any error

# Configuration
PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="gemini-flash-api"
IMAGE_NAME="gemini-flash-api"
REPOSITORY_NAME="text-translation"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking Prerequisites"
    echo "=================================================="
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if PROJECT_ID is set
    if [ -z "$PROJECT_ID" ]; then
        log_error "PROJECT_ID environment variable is not set."
        log_info "Please run: export PROJECT_ID=your-project-id"
        exit 1
    fi
    
    # Check if user is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "No active gcloud authentication found."
        log_info "Please run: gcloud auth login"
        exit 1
    fi
    
    # Set the project
    gcloud config set project "$PROJECT_ID"
    
    log_success "Prerequisites check passed"
}

# Function to enable required APIs
enable_apis() {
    log_info "Enabling Required APIs"
    echo "=================================================="
    
    local apis=(
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
        "artifactregistry.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log_info "Enabling $api..."
        gcloud services enable "$api" --project="$PROJECT_ID"
    done
    
    log_success "APIs enabled successfully"
}

# Function to setup Artifact Registry
setup_artifact_registry() {
    log_info "Setting up Artifact Registry"
    echo "=================================================="
    
    # Check if repository exists
    if gcloud artifacts repositories describe "$REPOSITORY_NAME" \
        --location="$REGION" --project="$PROJECT_ID" &>/dev/null; then
        log_info "Repository already exists"
    else
        log_info "Creating $REPOSITORY_NAME repository..."
        gcloud artifacts repositories create "$REPOSITORY_NAME" \
            --repository-format=docker \
            --location="$REGION" \
            --description="Text Translation APIs" \
            --project="$PROJECT_ID"
    fi
    
    log_success "Artifact Registry setup complete"
}

# Function to build and deploy
build_and_deploy() {
    log_info "Building and Deploying"
    echo "=================================================="
    
    local build_config="deploy/cloudbuild.yaml"
    local memory="8Gi"
    local cpu="2"
    local min_instances="0"
    local max_instances="10"
    local timeout="300"
    local concurrency="100"
    local env_vars="ENVIRONMENT=production,LOG_LEVEL=INFO"
    
    log_info "Building Docker image..."
    gcloud builds submit --region="$REGION" --config="$build_config" \
        --project="$PROJECT_ID"
    
    log_info "Deploying to Cloud Run..."
    local image_url="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY_NAME/$IMAGE_NAME:latest"
    
    gcloud run deploy "$SERVICE_NAME" \
        --project="$PROJECT_ID" \
        --image="$image_url" \
        --memory="$memory" \
        --cpu="$cpu" \
        --min-instances="$min_instances" \
        --max-instances="$max_instances" \
        --allow-unauthenticated \
        --port=8080 \
        --region="$REGION" \
        --timeout="$timeout" \
        --concurrency="$concurrency" \
        --set-env-vars="$env_vars" \
        --execution-environment=gen2
    
    # Update traffic to latest revision
    gcloud run services update-traffic "$SERVICE_NAME" \
        --to-latest --region="$REGION" --project="$PROJECT_ID"
    
    log_success "Deployment successful!"
}

# Function to get service information
get_service_info() {
    echo ""
    log_info "📡 Service Information:"
    
    local service_url=$(gcloud run services describe "$SERVICE_NAME" \
        --region="$REGION" --project="$PROJECT_ID" \
        --format="value(status.url)")
    
    echo "Service Name: $SERVICE_NAME"
    echo "Service URL: $service_url"
    echo "API Docs: $service_url/docs"
    echo "Health Check: $service_url/health"
    echo "Supported Languages: $service_url/languages"
    echo "Model Info: $service_url/model/info"
    echo ""
    echo "🧪 Test the API:"
    echo "curl -X POST \"$service_url/translate\" \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '{\"text\": \"Hello, world!\", \"source_language\": \"english\", \"target_language\": \"spanish\"}'"
    
    # Save URLs to file
    cat > deployment_urls_gemini.txt << EOF
# Gemini Flash Translation API Deployment URLs
SERVICE_URL=$service_url
API_DOCS=$service_url/docs
HEALTH_CHECK=$service_url/health
LANGUAGES=$service_url/languages
MODEL_INFO=$service_url/model/info
STATS=$service_url/stats
CACHE_INFO=$service_url/cache/info

# Test commands
curl -X POST "$service_url/translate" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "Hello, world!", "source_language": "english", "target_language": "spanish"}'
EOF
    
    log_success "Deployment URLs saved to deployment_urls_gemini.txt"
}

# Function to test deployment
test_deployment() {
    log_info "Waiting for service to initialize..."
    sleep 10
    
    echo ""
    log_info "Testing Deployment"
    echo "=================================================="
    
    local service_url=$(gcloud run services describe "$SERVICE_NAME" \
        --region="$REGION" --project="$PROJECT_ID" \
        --format="value(status.url)")
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    if curl -f -s "$service_url/health" > /dev/null; then
        log_success "Health check passed"
    else
        log_warning "Health check failed - service might still be starting"
    fi
    
    # Note about API key requirement
    log_warning "⚠️  IMPORTANT: This service requires GOOGLE_API_KEY environment variable"
    log_info "To set the API key, run:"
    echo "gcloud run services update $SERVICE_NAME \\"
    echo "  --set-env-vars=\"GOOGLE_API_KEY=your-api-key-here\" \\"
    echo "  --region=$REGION --project=$PROJECT_ID"
    echo ""
    log_info "Get your API key from: https://makersuite.google.com/app/apikey"
}

# Main execution
main() {
    echo "🌍 Gemini Flash Translation API Deployment"
    echo "Project: $PROJECT_ID"
    echo "Region: $REGION"
    echo ""
    
    check_prerequisites
    enable_apis
    setup_artifact_registry
    build_and_deploy
    get_service_info
    test_deployment
    
    echo ""
    log_success "🎉 Deployment completed successfully!"
    echo ""
    log_warning "Remember to set your GOOGLE_API_KEY environment variable!"
}

# Run main function
main "$@"