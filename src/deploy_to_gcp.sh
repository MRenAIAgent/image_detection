#!/bin/bash

# Complete GCP Deployment Script for Image Recognition API
# This script automates the entire deployment process to Google Cloud Platform
# Supports both GPU and CPU deployment options

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
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

log_header() {
    echo -e "${BOLD}${BLUE}$1${NC}"
}

# Configuration
PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-us-central1}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-cpu}"  # cpu or gpu

# Banner
show_banner() {
    echo -e "${BOLD}${BLUE}"
    echo "================================================================"
    echo "        Image Recognition API - GCP Deployment Script"
    echo "                    Deployment Type: $(echo $DEPLOYMENT_TYPE | tr '[:lower:]' '[:upper:]')"
    echo "================================================================"
    echo -e "${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_header "🔍 Checking Prerequisites"
    
    local missing_tools=()
    
    # Check required tools
    if ! command -v gcloud &> /dev/null; then
        missing_tools+=("gcloud CLI")
    fi
    
    if ! command -v git &> /dev/null; then
        missing_tools+=("git")
    fi
    
    if ! command -v curl &> /dev/null; then
        missing_tools+=("curl")
    fi
    
    # Additional checks for GPU deployment
    if [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
        if ! command -v kubectl &> /dev/null; then
            missing_tools+=("kubectl")
        fi
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        echo
        log_info "Please install the missing tools:"
        log_info "  - gcloud CLI: https://cloud.google.com/sdk/docs/install"
        log_info "  - git: https://git-scm.com/downloads"
        log_info "  - curl: Usually pre-installed on most systems"
        if [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
            log_info "  - kubectl: gcloud components install kubectl"
        fi
        exit 1
    fi
    
    # Check PROJECT_ID
    if [[ -z "$PROJECT_ID" ]]; then
        log_error "PROJECT_ID environment variable is required"
        echo
        log_info "Please set your GCP project ID:"
        log_info "  export PROJECT_ID=your-gcp-project-id"
        log_info "  $0"
        echo
        read -p "Or enter your project ID now: " PROJECT_ID
        
        if [[ -z "$PROJECT_ID" ]]; then
            log_error "Project ID is required. Exiting."
            exit 1
        fi
        
        export PROJECT_ID
    fi
    
    log_success "All prerequisites satisfied"
    log_info "Project ID: $PROJECT_ID"
    log_info "Region: $REGION"
    log_info "Deployment Type: $DEPLOYMENT_TYPE"
}

# Validate GCP setup
validate_gcp() {
    log_header "🔐 Validating GCP Setup"
    
    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_warning "Not authenticated with gcloud"
        log_info "Initiating authentication..."
        gcloud auth login
    fi
    
    local active_account
    active_account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1)
    log_success "Authenticated as: $active_account"
    
    # Set project
    log_info "Setting active project to: $PROJECT_ID"
    if ! gcloud config set project "$PROJECT_ID"; then
        log_error "Failed to set project. Please check if project exists and you have access."
        exit 1
    fi
    
    # Check project access
    if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
        log_error "Cannot access project: $PROJECT_ID"
        log_info "Please ensure:"
        log_info "  1. The project exists"
        log_info "  2. You have the necessary permissions"
        log_info "  3. The project ID is correct"
        exit 1
    fi
    
    # Check billing
    local billing_account
    billing_account=$(gcloud beta billing projects describe "$PROJECT_ID" --format="value(billingAccountName)" 2>/dev/null || echo "")
    
    if [[ -z "$billing_account" ]]; then
        log_warning "No billing account found for project $PROJECT_ID"
        log_warning "Billing is required for Cloud Run and other services"
        echo
        read -p "Continue anyway? (y/N): " continue_without_billing
        if [[ "$continue_without_billing" != "y" && "$continue_without_billing" != "Y" ]]; then
            log_info "Please enable billing for your project and try again"
            log_info "Visit: https://console.cloud.google.com/billing/projects"
            exit 1
        fi
    else
        log_success "Billing is enabled"
    fi
    
    log_success "GCP setup validated"
}

# Enable required APIs
enable_apis() {
    log_header "🔧 Enabling Required APIs"
    
    local apis=(
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
        "storage.googleapis.com"
        "containerregistry.googleapis.com"
        "artifactregistry.googleapis.com"
    )
    
    # Add GPU-specific APIs
    if [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
        apis+=(
            "container.googleapis.com"
            "compute.googleapis.com"
        )
    fi
    
    log_info "Enabling APIs (this may take a few minutes)..."
    
    for api in "${apis[@]}"; do
        log_info "  Enabling $api..."
        if gcloud services enable "$api" --quiet; then
            log_success "  ✓ $api enabled"
        else
            log_error "  ✗ Failed to enable $api"
            exit 1
        fi
    done
    
    log_success "All required APIs enabled"
}

# Setup service account and permissions
setup_permissions() {
    log_header "👤 Setting up Service Account and Permissions"
    
    local sa_email="cloudbuild@$PROJECT_ID.iam.gserviceaccount.com"
    
    # Create service account if it doesn't exist
    if ! gcloud iam service-accounts describe "$sa_email" &> /dev/null; then
        log_info "Creating Cloud Build service account..."
        gcloud iam service-accounts create cloudbuild \
            --display-name="Cloud Build Service Account" \
            --description="Service account for Cloud Build deployments"
    else
        log_info "Cloud Build service account already exists"
    fi
    
    # Grant necessary permissions
    local roles=(
        "roles/run.admin"
        "roles/storage.admin"
        "roles/iam.serviceAccountUser"
        "roles/artifactregistry.admin"
    )
    
    # Add GPU-specific roles
    if [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
        roles+=(
            "roles/container.admin"
            "roles/compute.admin"
        )
    fi
    
    log_info "Granting permissions to service account..."
    for role in "${roles[@]}"; do
        log_info "  Granting $role..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$sa_email" \
            --role="$role" \
            --quiet
    done
    
    log_success "Service account and permissions configured"
}

# Create GKE cluster for GPU deployment
create_gpu_cluster() {
    if [[ "$DEPLOYMENT_TYPE" != "gpu" ]]; then
        return 0
    fi
    
    log_header "🖥️ Setting up GKE Cluster for GPU Deployment"
    
    local cluster_name="image-recognition-gpu-cluster"
    local zone="us-central1-a"
    
    # Check if cluster exists
    if gcloud container clusters describe "$cluster_name" --zone="$zone" &> /dev/null; then
        log_info "GKE cluster already exists"
        return 0
    fi
    
    log_info "Creating GKE cluster with GPU nodes..."
    gcloud container clusters create "$cluster_name" \
        --zone="$zone" \
        --machine-type="n1-standard-4" \
        --accelerator="type=nvidia-tesla-t4,count=1" \
        --num-nodes=1 \
        --min-nodes=0 \
        --max-nodes=3 \
        --enable-autoscaling \
        --enable-autorepair \
        --enable-autoupgrade \
        --disk-size=50GB \
        --scopes="https://www.googleapis.com/auth/cloud-platform"
    
    # Install NVIDIA device plugin
    log_info "Installing NVIDIA device plugin..."
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
    
    log_success "GKE cluster created successfully"
}

# Create storage bucket
create_storage() {
    log_header "🪣 Setting up Cloud Storage"
    
    local bucket_name="$PROJECT_ID-image-recognition-models-$DEPLOYMENT_TYPE"
    local artifacts_bucket="$PROJECT_ID-build-artifacts"
    
    # Create models bucket
    if ! gsutil ls "gs://$bucket_name" &> /dev/null; then
        log_info "Creating models storage bucket: $bucket_name"
        gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$bucket_name"
        log_success "Models bucket created"
    else
        log_info "Models bucket already exists"
    fi
    
    # Create artifacts bucket
    if ! gsutil ls "gs://$artifacts_bucket" &> /dev/null; then
        log_info "Creating build artifacts bucket: $artifacts_bucket"
        gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$artifacts_bucket"
        log_success "Artifacts bucket created"
    else
        log_info "Artifacts bucket already exists"
    fi
}

# Deploy the application
deploy_application() {
    log_header "🚀 Deploying Application ($DEPLOYMENT_TYPE)"
    
    log_info "Starting Cloud Build deployment..."
    log_info "This process typically takes 15-30 minutes"
    echo
    
    # Select appropriate build configuration
    local build_config="deploy/cloudbuild-simple.yaml"
    
    if [[ ! -f "$build_config" ]]; then
        log_error "Build configuration not found: $build_config"
        exit 1
    fi
    
    # Submit build
    local build_id
    local substitutions="COMMIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo 'latest')"
    
    if build_id=$(gcloud builds submit \
        --config="$build_config" \
        --substitutions="$substitutions" \
        --format="value(id)" 2>/dev/null); then
        
        log_info "Build submitted with ID: $build_id"
        log_info "You can monitor the build at:"
        log_info "  https://console.cloud.google.com/cloud-build/builds/$build_id?project=$PROJECT_ID"
        echo
        
        # Stream build logs
        log_info "Streaming build logs (press Ctrl+C to stop streaming, build will continue)..."
        gcloud builds log "$build_id" --stream || true
        
        # Check final build status
        local build_status
        build_status=$(gcloud builds describe "$build_id" --format="value(status)")
        
        if [[ "$build_status" == "SUCCESS" ]]; then
            log_success "Deployment completed successfully!"
            return 0
        else
            log_error "Deployment failed with status: $build_status"
            return 1
        fi
    else
        log_error "Failed to submit build"
        return 1
    fi
}

# Get deployment URLs
get_deployment_urls() {
    log_header "📋 Deployment Information"
    
    # Get hash for service names
    local commit_sha
    commit_sha=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
    local hash="${commit_sha:0:7}"
    
    # Get service URLs
    local api_url
    local triton_url
    
    if [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
        api_url=$(gcloud run services describe "image-recognition-api-gpu-$hash" --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "")
        # For GPU, Triton runs on GKE
        triton_url="Check GKE service: kubectl get service triton-server-gpu-service-$hash"
    else
        api_url=$(gcloud run services describe "image-recognition-api-cpu-$hash" --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "")
        triton_url=$(gcloud run services describe "triton-server-cpu-$hash" --region="$REGION" --format="value(status.url)" 2>/dev/null || echo "")
    fi
    
    if [[ -n "$api_url" ]]; then
        echo -e "${BOLD}${GREEN}🎉 $(echo $DEPLOYMENT_TYPE | tr '[:lower:]' '[:upper:]') Deployment Successful!${NC}"
        echo
        echo -e "${BOLD}📡 Service URLs:${NC}"
        echo -e "  ${BLUE}API Endpoint:${NC} $api_url"
        echo -e "  ${BLUE}API Documentation:${NC} $api_url/docs"
        echo -e "  ${BLUE}Health Check:${NC} $api_url/health"
        echo -e "  ${BLUE}Model Info:${NC} $api_url/model/info"
        
        if [[ -n "$triton_url" && "$triton_url" != "Check"* ]]; then
            echo -e "  ${BLUE}Triton Server:${NC} $triton_url"
        elif [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
            echo -e "  ${BLUE}Triton Server:${NC} $triton_url"
        fi
        
        echo
        echo -e "${BOLD}🧪 Test your deployment:${NC}"
        echo "  curl -X POST \"$api_url/detect\" \\"
        echo "    -H \"Content-Type: multipart/form-data\" \\"
        echo "    -F \"file=@your-image.jpg\""
        
        echo
        echo -e "${BOLD}📊 Monitor your services:${NC}"
        echo "  gcloud run services list --region=$REGION"
        echo "  gcloud logging read \"resource.type=cloud_run_revision\""
        
        if [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
            echo "  kubectl get pods"
            echo "  kubectl logs -l app=triton-server-gpu"
        fi
        
        # Save URLs to file
        cat > "deployment_urls_${DEPLOYMENT_TYPE}.txt" << EOF
# Image Recognition API Deployment URLs ($(echo $DEPLOYMENT_TYPE | tr '[:lower:]' '[:upper:]'))
# Generated on $(date)

API_URL=$api_url
TRITON_URL=$triton_url
PROJECT_ID=$PROJECT_ID
REGION=$REGION
DEPLOYMENT_TYPE=$DEPLOYMENT_TYPE

# Test commands:
# curl -X GET $api_url/health
# curl -X GET $api_url/docs
EOF
        
        log_success "Deployment URLs saved to deployment_urls_${DEPLOYMENT_TYPE}.txt"
        
    else
        log_error "Could not retrieve service URLs"
        log_info "Check the Cloud Console for service status:"
        log_info "  https://console.cloud.google.com/run?project=$PROJECT_ID"
        return 1
    fi
}

# Test deployment
test_deployment() {
    local urls_file="deployment_urls_${DEPLOYMENT_TYPE}.txt"
    
    if [[ -f "$urls_file" ]]; then
        source "$urls_file"
        
        log_header "🧪 Testing Deployment"
        
        if [[ -n "$API_URL" ]]; then
            log_info "Testing health endpoint..."
            if curl -s -f "$API_URL/health" | grep -q "healthy\|unhealthy"; then
                log_success "✓ Health check passed"
            else
                log_warning "✗ Health check failed"
            fi
            
            log_info "Testing root endpoint..."
            if curl -s -f "$API_URL/" | grep -q "Image Recognition API"; then
                log_success "✓ Root endpoint test passed"
            else
                log_warning "✗ Root endpoint test failed"
            fi
            
            echo
            log_info "Run comprehensive tests with:"
            log_info "  python scripts/test_api.py --url $API_URL"
        fi
    fi
}

# Cleanup function
cleanup_deployment() {
    log_header "🧹 Cleanup Options"
    
    echo "This will delete all deployed resources including:"
    echo "  - Cloud Run services"
    echo "  - Cloud Scheduler jobs"
    echo "  - Container images"
    if [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
        echo "  - GKE cluster and nodes"
    fi
    echo
    echo "Note: Storage buckets will be preserved to avoid data loss"
    echo
    read -p "Are you sure you want to proceed? (y/N): " confirm
    
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        log_info "Cleanup cancelled"
        return 0
    fi
    
    # Get hash for service names
    local commit_sha
    commit_sha=$(git rev-parse --short HEAD 2>/dev/null || echo "latest")
    local hash="${commit_sha:0:7}"
    
    log_info "Deleting Cloud Run services..."
    gcloud run services delete "image-recognition-api-${DEPLOYMENT_TYPE}-$hash" --region="$REGION" --quiet 2>/dev/null || true
    
    if [[ "$DEPLOYMENT_TYPE" == "cpu" ]]; then
        gcloud run services delete "triton-server-cpu-$hash" --region="$REGION" --quiet 2>/dev/null || true
    fi
    
    log_info "Deleting Cloud Scheduler jobs..."
    gcloud scheduler jobs delete "image-recognition-health-check-${DEPLOYMENT_TYPE}-$hash" --location="$REGION" --quiet 2>/dev/null || true
    
    if [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
        log_info "Deleting GKE cluster..."
        gcloud container clusters delete "image-recognition-gpu-cluster" --zone="us-central1-a" --quiet 2>/dev/null || true
    fi
    
    log_info "Cleaning up container images..."
    gcloud container images list --repository="gcr.io/$PROJECT_ID" --filter="name:image-recognition-api" --format="value(name)" | while read -r image; do
        gcloud container images delete "$image" --quiet 2>/dev/null || true
    done
    
    log_success "Cleanup completed"
    log_info "Storage buckets preserved:"
    log_info "  gs://$PROJECT_ID-image-recognition-models-$DEPLOYMENT_TYPE"
    log_info "  gs://$PROJECT_ID-build-artifacts"
}

# Show usage
show_usage() {
    echo "Usage: $0 [command]"
    echo
    echo "Commands:"
    echo "  deploy    - Deploy the application (default)"
    echo "  test      - Test existing deployment"
    echo "  cleanup   - Clean up deployment"
    echo "  help      - Show this help message"
    echo
    echo "Environment variables:"
    echo "  PROJECT_ID       - GCP project ID (required)"
    echo "  REGION           - GCP region (default: us-central1)"
    echo "  DEPLOYMENT_TYPE  - Deployment type: cpu or gpu (default: cpu)"
    echo
    echo "Examples:"
    echo "  PROJECT_ID=my-project $0 deploy"
    echo "  PROJECT_ID=my-project DEPLOYMENT_TYPE=gpu $0 deploy"
    echo "  PROJECT_ID=my-project DEPLOYMENT_TYPE=cpu $0 deploy"
    echo "  PROJECT_ID=my-project DEPLOYMENT_TYPE=gpu $0 cleanup"
    echo
    echo "Deployment Types:"
    echo "  cpu - Cost-effective CPU-only deployment using Cloud Run"
    echo "  gpu - High-performance GPU deployment using GKE with NVIDIA T4"
}

# Main deployment function
main_deploy() {
    show_banner
    check_prerequisites
    validate_gcp
    enable_apis
    setup_permissions
    
    if [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
        create_gpu_cluster
    fi
    
    create_storage
    
    if deploy_application; then
        get_deployment_urls
        test_deployment
        
        echo
        log_success "🎉 $(echo $DEPLOYMENT_TYPE | tr '[:lower:]' '[:upper:]') deployment completed successfully!"
        log_info "Your Image Recognition API is now running on Google Cloud Platform"
        
        if [[ "$DEPLOYMENT_TYPE" == "gpu" ]]; then
            echo
            log_info "💡 GPU Deployment Benefits:"
            log_info "  - ~3-5x faster inference"
            log_info "  - Higher throughput (100+ RPS)"
            log_info "  - Better for high-volume workloads"
        else
            echo
            log_info "💡 CPU Deployment Benefits:"
            log_info "  - Lower cost (~$15-30/month)"
            log_info "  - Scales to zero when idle"
            log_info "  - Good for moderate workloads"
        fi
    else
        log_error "Deployment failed"
        log_info "Check the build logs for more details"
        exit 1
    fi
}

# Main function
main() {
    case "${1:-deploy}" in
        "deploy")
            main_deploy
            ;;
        "test")
            test_deployment
            ;;
        "cleanup")
            cleanup_deployment
            ;;
        "help"|"-h"|"--help")
            show_usage
            ;;
        *)
            log_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Trap Ctrl+C
trap 'echo -e "\n${YELLOW}[INFO]${NC} Interrupted by user"; exit 130' INT

# Run main function
main "$@" 