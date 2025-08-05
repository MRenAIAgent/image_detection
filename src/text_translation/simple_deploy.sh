#!/bin/bash

# Simple deployment script for Text Translation API
# Similar to the image detection simple deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
REGION="${REGION:-us-central1}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-cpu}"  # cpu or gpu

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

log_header() {
    echo -e "\n${BOLD}${BLUE}$1${NC}"
    echo "=================================================="
}

# Check prerequisites
check_prerequisites() {
    log_header "Checking Prerequisites"
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    
    # Check if user is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n 1 > /dev/null; then
        log_error "Please authenticate with gcloud: gcloud auth login"
        exit 1
    fi
    
    # Check if PROJECT_ID is set
    if [ -z "$PROJECT_ID" ]; then
        log_error "PROJECT_ID environment variable is required"
        echo "Usage: PROJECT_ID=your-project-id $0"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Enable required APIs
enable_apis() {
    log_header "Enabling Required APIs"
    
    local apis=(
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
        "artifactregistry.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log_info "Enabling $api..."
        gcloud services enable "$api" --project="$PROJECT_ID" --quiet
    done
    
    log_success "APIs enabled successfully"
}

# Setup Artifact Registry
setup_registry() {
    log_header "Setting up Artifact Registry"
    
    log_info "Creating text-translation repository..."
    gcloud artifacts repositories create text-translation \
        --repository-format=docker \
        --location="$REGION" \
        --project="$PROJECT_ID" \
        --quiet 2>/dev/null || log_info "Repository already exists"
    
    log_success "Artifact Registry setup complete"
}

# Build and deploy
build_and_deploy() {
    log_header "Building and Deploying"
    
    local commit_sha=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
    
    if [ "$DEPLOYMENT_TYPE" = "gpu" ]; then
        local build_config="deploy/cloudbuild-gpu.yaml"
        local service_name="text-translation-api-gpu"
        local image_name="text-translation-api-gpu"
        local memory="16Gi"
        local cpu="4"
        local min_instances="1"
        local max_instances="5"
        local timeout="1800"
        local concurrency="100"
        local env_vars="ENVIRONMENT=production,LOG_LEVEL=INFO,DEVICE=cuda,QUANTIZATION_TYPE=fp16"
    else
        local build_config="deploy/cloudbuild.yaml"
        local service_name="text-translation-api"
        local image_name="text-translation-api"
        local memory="16Gi"
        local cpu="4"
        local min_instances="1"
        local max_instances="5"
        local timeout="1800"
        local concurrency="100"
        local env_vars="ENVIRONMENT=production,LOG_LEVEL=INFO,DEVICE=cpu,QUANTIZATION_TYPE=int8"
    fi
    
    log_info "Building Docker image..."
    gcloud builds submit --region="$REGION" --config="$build_config" \
        --project="$PROJECT_ID"
    
    log_info "Deploying to Cloud Run..."
    local image_url="$REGION-docker.pkg.dev/$PROJECT_ID/text-translation/$image_name:latest"
    
    gcloud run deploy "$service_name" \
        --image="$image_url" \
        --region="$REGION" \
        --platform=managed \
        --allow-unauthenticated \
        --memory="$memory" \
        --cpu="$cpu" \
        --min-instances="$min_instances" \
        --max-instances="$max_instances" \
        --timeout="$timeout" \
        --concurrency="$concurrency" \
        --set-env-vars="$env_vars" \
        --project="$PROJECT_ID"
    
    # Get the service URL
    local service_url=$(gcloud run services describe "$service_name" \
        --region="$REGION" --project="$PROJECT_ID" \
        --format="value(status.url)")
    
    log_success "Deployment successful!"
    echo
    echo -e "${BOLD}📡 Service Information:${NC}"
    echo "Service Name: $service_name"
    echo "Service URL: $service_url"
    echo "API Docs: $service_url/docs"
    echo "Health Check: $service_url/health"
    echo "Supported Languages: $service_url/languages"
    echo "Model Info: $service_url/model/info"
    echo
    echo -e "${BOLD}🧪 Test the API:${NC}"
    echo "curl -X POST \"$service_url/translate\" \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '{\"text\": \"Hello, world!\", \"source_language\": \"english\", \"target_language\": \"spanish\"}'"
    
    # Save URLs to file
    cat > "deployment_urls_${DEPLOYMENT_TYPE}.txt" << EOF
# Text Translation API Deployment URLs ($(echo $DEPLOYMENT_TYPE | tr '[:lower:]' '[:upper:]'))
# Generated on $(date)

SERVICE_URL=$service_url
API_DOCS=$service_url/docs
HEALTH_CHECK=$service_url/health
SUPPORTED_LANGUAGES=$service_url/languages
MODEL_INFO=$service_url/model/info
EOF
    
    log_success "Deployment URLs saved to deployment_urls_${DEPLOYMENT_TYPE}.txt"
}

# Test deployment
test_deployment() {
    log_header "Testing Deployment"
    
    local service_name="text-translation-api"
    if [ "$DEPLOYMENT_TYPE" = "gpu" ]; then
        service_name="text-translation-api-gpu"
    fi
    
    local service_url=$(gcloud run services describe "$service_name" \
        --region="$REGION" --project="$PROJECT_ID" \
        --format="value(status.url)" 2>/dev/null)
    
    if [ -z "$service_url" ]; then
        log_error "Could not get service URL"
        return 1
    fi
    
    log_info "Testing health endpoint..."
    if curl -f -s "$service_url/health" > /dev/null; then
        log_success "Health check passed"
    else
        log_warning "Health check failed - service might still be starting up"
    fi
    
    log_info "Testing translation endpoint..."
    local response=$(curl -s -X POST "$service_url/translate" \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello", "source_language": "english", "target_language": "spanish"}')
    
    if echo "$response" | grep -q "translated_text"; then
        log_success "Translation test passed"
        echo "Response: $response"
    else
        log_warning "Translation test failed or service still initializing"
        echo "Response: $response"
    fi
}

# Main execution
main() {
    echo -e "${BOLD}${BLUE}🌍 Text Translation API Deployment${NC}"
    echo "Project: $PROJECT_ID"
    echo "Region: $REGION"
    echo "Deployment Type: $DEPLOYMENT_TYPE"
    echo
    
    check_prerequisites
    enable_apis
    setup_registry
    build_and_deploy
    
    # Wait a bit for the service to start
    log_info "Waiting for service to initialize..."
    sleep 30
    
    test_deployment
    
    echo
    log_success "🎉 Deployment completed successfully!"
}

# Run main function
main "$@"