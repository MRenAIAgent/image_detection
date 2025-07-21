#!/bin/bash

# Image Recognition API Deployment Script
# Supports local Docker deployment and GCP Cloud Run deployment

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="image-recognition-api"
TRITON_SERVICE_NAME="triton-server"
MODEL_NAME="${MODEL_NAME:-yolov8s}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-cpu}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
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

# Check if required tools are installed
check_requirements() {
    log_info "Checking requirements..."
    
    local missing_tools=()
    
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if ! command -v python3 &> /dev/null; then
        missing_tools+=("python3")
    fi
    
    if [[ "$1" == "gcp" ]]; then
        if ! command -v gcloud &> /dev/null; then
            missing_tools+=("gcloud")
        fi
        
        if [[ -z "$PROJECT_ID" ]]; then
            log_error "PROJECT_ID environment variable is required for GCP deployment"
            log_info "Set it with: export PROJECT_ID=your-gcp-project-id"
            exit 1
        fi
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and try again"
        exit 1
    fi
    
    log_success "All requirements satisfied"
}

# Validate GCP setup
validate_gcp_setup() {
    log_info "Validating GCP setup..."
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_error "Please authenticate with gcloud: gcloud auth login"
        exit 1
    fi
    
    # Check project access
    if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
        log_error "Cannot access project: $PROJECT_ID"
        log_info "Make sure the project exists and you have access"
        exit 1
    fi
    
    # Check billing
    local billing_account
    billing_account=$(gcloud beta billing projects describe "$PROJECT_ID" --format="value(billingAccountName)" 2>/dev/null || echo "")
    
    if [[ -z "$billing_account" ]]; then
        log_warning "No billing account found for project $PROJECT_ID"
        log_warning "Some services may not work without billing enabled"
    fi
    
    log_success "GCP setup validated"
}

# Setup model (download and convert YOLO model)
setup_model() {
    log_info "Setting up $MODEL_NAME model for $DEPLOYMENT_TYPE deployment..."
    
    # Create models directory
    mkdir -p models
    
    # Check if Python packages are installed
    if ! python3 -c "import ultralytics, torch, onnx, cv2, PIL" &> /dev/null; then
        log_info "Installing required Python packages..."
        pip3 install ultralytics torch onnx pillow opencv-python numpy
    fi
    
    # Build setup command based on deployment type
    setup_cmd="python3 setup_model.py --model $MODEL_NAME --model-dir models"
    
    if [[ "$DEPLOYMENT_TYPE" == "cpu" ]]; then
        setup_cmd="$setup_cmd --optimize-cpu"
        log_info "Configuring for CPU deployment with optimizations"
    elif [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
        setup_cmd="$setup_cmd --enable-tensorrt"
        log_info "Configuring for GPU deployment with TensorRT"
    else
        log_error "Invalid deployment type: $DEPLOYMENT_TYPE. Use 'cpu' or 'gpu'"
        exit 1
    fi
    
    # Run model setup
    log_info "Running: $setup_cmd"
    eval $setup_cmd
    
    if [ $? -eq 0 ]; then
        log_success "Model setup completed"
        
        # Validate model files
        if [[ -f "models/model_repository/$MODEL_NAME/1/model.onnx" && -f "models/model_repository/$MODEL_NAME/config.pbtxt" ]]; then
            log_success "Model files validated"
        elif [[ -f "models/model_repository/$MODEL_NAME/1/model.plan" && -f "models/model_repository/$MODEL_NAME/config.pbtxt" ]]; then
            log_success "TensorRT model files validated"
        else
            log_error "Model files missing after setup"
            exit 1
        fi
    else
        log_error "Model setup failed"
        exit 1
    fi
}

# Local deployment using Docker Compose
deploy_local() {
    log_info "Starting local deployment..."
    
    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose not found"
        log_info "Please install Docker Compose"
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        log_info "Please start Docker and try again"
        exit 1
    fi
    
    # Setup model if not exists
    if [ ! -d "models/model_repository" ]; then
        setup_model
    fi
    
    # Create logs directory
    mkdir -p logs
    
    # Clean up any existing containers
    log_info "Cleaning up existing containers..."
    if command -v docker-compose &> /dev/null; then
        docker-compose down 2>/dev/null || true
    else
        docker compose down 2>/dev/null || true
    fi
    
    # Build and start services
    log_info "Building and starting services..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose up --build -d
    else
        docker compose up --build -d
    fi
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    
    # Wait for Triton server
    log_info "Waiting for Triton server..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/v2/health/ready > /dev/null 2>&1; then
            log_success "Triton server is ready"
            break
        fi
        if [ $i -eq 60 ]; then
            log_error "Triton server failed to start within 10 minutes"
            log_info "Check logs with: docker logs triton-server"
            exit 1
        fi
        sleep 10
    done
    
    # Wait for FastAPI client
    log_info "Waiting for FastAPI client..."
    for i in {1..30}; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            log_success "FastAPI client is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "FastAPI client failed to start within 5 minutes"
            log_info "Check logs with: docker logs fastapi-client"
            exit 1
        fi
        sleep 10
    done
    
    log_success "Local deployment completed successfully!"
    echo
    log_info "🚀 Services are now running:"
    log_info "   API: http://localhost:8080"
    log_info "   API Docs: http://localhost:8080/docs"
    log_info "   Health: http://localhost:8080/health"
    log_info "   Triton: http://localhost:8000"
    log_info "   Triton Metrics: http://localhost:8002/metrics"
    echo
    log_info "📊 View logs with:"
    log_info "   docker logs fastapi-client -f"
    log_info "   docker logs triton-server -f"
}

# GCP deployment using Cloud Build
deploy_gcp() {
    log_info "Starting GCP deployment for project: $PROJECT_ID"
    
    # Validate GCP setup
    validate_gcp_setup
    
    # Set project
    log_info "Setting GCP project to: $PROJECT_ID"
    gcloud config set project "$PROJECT_ID"
    
    # Enable required APIs
    log_info "Enabling required APIs..."
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        storage.googleapis.com \
        containerregistry.googleapis.com \
        artifactregistry.googleapis.com
    
    # Create service account for Cloud Build if not exists
    log_info "Setting up service account..."
    
    local sa_email="cloudbuild@$PROJECT_ID.iam.gserviceaccount.com"
    
    if ! gcloud iam service-accounts describe "$sa_email" &> /dev/null; then
        gcloud iam service-accounts create cloudbuild \
            --display-name="Cloud Build Service Account"
    fi
    
    # Grant necessary permissions
    local roles=(
        "roles/run.admin"
        "roles/storage.admin"
        "roles/iam.serviceAccountUser"
        "roles/artifactregistry.admin"
    )
    
    for role in "${roles[@]}"; do
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$sa_email" \
            --role="$role" \
            --quiet
    done
    
    # Create build artifacts bucket
    log_info "Creating build artifacts bucket..."
    gsutil mb -p "$PROJECT_ID" "gs://$PROJECT_ID-build-artifacts" 2>/dev/null || true
    
    # Submit build
    log_info "Submitting Cloud Build (this may take 15-30 minutes)..."
    
    local build_id
    build_id=$(gcloud builds submit \
        --config=deploy/cloudbuild-simple.yaml \
        --substitutions=COMMIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')" \
        --format="value(id)")
    
    if [ $? -eq 0 ]; then
        log_success "GCP deployment completed successfully!"
        
        # Get service URLs
        local commit_sha
        commit_sha=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
        local hash="${commit_sha:0:7}"
        
        local api_url
        local triton_url
        
        api_url=$(gcloud run services describe "image-recognition-api-$hash" --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "")
        triton_url=$(gcloud run services describe "triton-server-$hash" --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "")
        
        echo
        log_success "🚀 Deployment URLs:"
        if [[ -n "$api_url" ]]; then
            log_info "   API: $api_url"
            log_info "   API Docs: $api_url/docs"
            log_info "   Health: $api_url/health"
        fi
        if [[ -n "$triton_url" ]]; then
            log_info "   Triton: $triton_url"
        fi
        echo
        log_info "📊 Monitor with:"
        log_info "   gcloud builds log $build_id"
        log_info "   gcloud run services list --region=$REGION"
        
    else
        log_error "GCP deployment failed"
        log_info "Check build logs with: gcloud builds log $build_id"
        exit 1
    fi
}

# Test deployment
test_deployment() {
    local base_url="$1"
    
    log_info "Testing deployment at: $base_url"
    
    # Test health endpoint
    log_info "Testing health endpoint..."
    if curl -s -f "$base_url/health" | grep -q "healthy\|unhealthy"; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi
    
    # Test root endpoint
    log_info "Testing root endpoint..."
    if curl -s -f "$base_url/" | grep -q "Image Recognition API"; then
        log_success "Root endpoint test passed"
    else
        log_error "Root endpoint test failed"
        return 1
    fi
    
    # Test model info endpoint
    log_info "Testing model info endpoint..."
    if curl -s -f "$base_url/model/info" > /dev/null; then
        log_success "Model info endpoint test passed"
    else
        log_warning "Model info endpoint test failed (may be normal if model not loaded)"
    fi
    
    log_success "Basic tests passed!"
    log_info "Run comprehensive tests with: python scripts/test_api.py --url $base_url"
}

# Cleanup function
cleanup() {
    local deployment_type="$2"
    
    log_info "Cleaning up $deployment_type deployment..."
    
    if [[ "$deployment_type" == "local" ]]; then
        if command -v docker-compose &> /dev/null; then
            docker-compose down -v
        else
            docker compose down -v
        fi
        
        # Clean up Docker images
        docker image prune -f
        
        # Clean up volumes
        docker volume prune -f
        
        log_success "Local cleanup completed"
        
    elif [[ "$deployment_type" == "gcp" ]]; then
        if [[ -z "$PROJECT_ID" ]]; then
            log_error "PROJECT_ID environment variable is required for GCP cleanup"
            exit 1
        fi
        
        # Get hash for service names
        local commit_sha
        commit_sha=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
        local hash="${commit_sha:0:7}"
        
        # Delete Cloud Run services
        log_info "Deleting Cloud Run services..."
        gcloud run services delete "image-recognition-api-$hash" --region="$REGION" --quiet 2>/dev/null || true
        gcloud run services delete "triton-server-$hash" --region="$REGION" --quiet 2>/dev/null || true
        
        # Delete Cloud Scheduler job
        log_info "Deleting Cloud Scheduler job..."
        gcloud scheduler jobs delete "image-recognition-health-check-$hash" --location="$REGION" --quiet 2>/dev/null || true
        
        # Delete Cloud Storage bucket (optional - commented out to preserve models)
        # log_info "Deleting Cloud Storage bucket..."
        # gsutil -m rm -r "gs://$PROJECT_ID-image-recognition-models" 2>/dev/null || true
        
        log_success "GCP cleanup completed"
        log_info "Note: Model storage bucket preserved. Delete manually if needed:"
        log_info "   gsutil -m rm -r gs://$PROJECT_ID-image-recognition-models"
    fi
}

# Show usage
show_usage() {
    echo "Image Recognition API Deployment Script"
    echo
    echo "Usage: $0 {local|gcp|test|cleanup|setup-model|status}"
    echo
    echo "Commands:"
    echo "  local              - Deploy locally using Docker Compose"
    echo "  gcp                - Deploy to GCP using Cloud Build"
    echo "  test <url>         - Test deployment at given URL"
    echo "  cleanup <type>     - Clean up deployment (local|gcp)"
    echo "  setup-model        - Setup YOLO model only"
    echo "  status             - Show deployment status"
    echo
    echo "Environment variables:"
    echo "  PROJECT_ID         - GCP project ID (required for GCP operations)"
    echo "  REGION             - GCP region (default: us-central1)"
    echo "  MODEL_NAME         - YOLO model name (default: yolov8s)"
    echo "  DEPLOYMENT_TYPE    - Deployment type: cpu|gpu (default: cpu)"
    echo
    echo "Examples:"
    echo "  $0 local                                    # Deploy locally with YOLOv8s (CPU)"
    echo "  MODEL_NAME=yolov8n $0 local                 # Deploy with YOLOv8n (CPU)"
    echo "  DEPLOYMENT_TYPE=gpu $0 local                # Deploy with YOLOv8s (GPU)"
    echo "  MODEL_NAME=yolov8m DEPLOYMENT_TYPE=cpu $0 local  # Deploy with YOLOv8m (CPU)"
    echo "  PROJECT_ID=my-project $0 gcp                # Deploy to GCP"
    echo "  $0 test http://localhost:8080               # Test local deployment"
    echo "  PROJECT_ID=my-project $0 cleanup gcp        # Clean up GCP deployment"
}

# Show deployment status
show_status() {
    log_info "Checking deployment status..."
    
    # Check local deployment
    echo
    log_info "Local deployment status:"
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(triton-server|fastapi-client)" > /dev/null 2>&1; then
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(triton-server|fastapi-client)"
        
        # Test local endpoints
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            log_success "Local API is responding"
        else
            log_warning "Local API is not responding"
        fi
    else
        log_info "No local containers running"
    fi
    
    # Check GCP deployment
    if [[ -n "$PROJECT_ID" ]]; then
        echo
        log_info "GCP deployment status for project: $PROJECT_ID"
        
        local services
        services=$(gcloud run services list --region="$REGION" --filter="metadata.name:image-recognition OR metadata.name:triton-server" --format="table(metadata.name,status.url,status.conditions[0].type)" 2>/dev/null || echo "")
        
        if [[ -n "$services" ]]; then
            echo "$services"
        else
            log_info "No GCP services found"
        fi
    else
        echo
        log_info "Set PROJECT_ID environment variable to check GCP deployment status"
    fi
}

# Main function
main() {
    case "$1" in
        "local")
            check_requirements "local"
            deploy_local
            test_deployment "http://localhost:8080"
            ;;
        "gcp")
            check_requirements "gcp"
            deploy_gcp
            ;;
        "test")
            if [[ -z "$2" ]]; then
                log_error "Please provide base URL for testing"
                log_info "Usage: $0 test <url>"
                exit 1
            fi
            test_deployment "$2"
            ;;
        "cleanup")
            if [[ -z "$2" ]]; then
                log_error "Please specify deployment type to cleanup"
                log_info "Usage: $0 cleanup {local|gcp}"
                exit 1
            fi
            cleanup "$@"
            ;;
        "setup-model")
            setup_model
            ;;
        "status")
            show_status
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            show_usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 