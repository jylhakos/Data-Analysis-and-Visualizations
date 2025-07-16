#!/bin/bash
"""
Complete AWS SageMaker Deployment Script for Arcee Agent

This script provides a complete deployment pipeline from local development
to production SageMaker endpoints.

Usage:
    ./deploy_to_sagemaker.sh [OPTIONS]
    
Options:
    --setup-iam         Setup IAM roles and policies
    --upload-data       Upload dataset to S3
    --build-images      Build and push Docker images
    --train-model       Start SageMaker training job
    --deploy-endpoint   Deploy model to SageMaker endpoint
    --start-api         Start FastAPI service
    --cleanup           Clean up AWS resources
    --help              Show this help message

Environment Variables:
    AWS_REGION          AWS region (default: us-east-1)
    S3_BUCKET          S3 bucket name (required)
    SAGEMAKER_ROLE_ARN  SageMaker execution role ARN
    PROJECT_NAME       Project name (default: arcee-agent)
"""

set -e

# Default configuration
AWS_REGION=${AWS_REGION:-"us-east-1"}
PROJECT_NAME=${PROJECT_NAME:-"arcee-agent"}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="$PROJECT_NAME-training-$TIMESTAMP"
MODEL_NAME="$PROJECT_NAME-model-$TIMESTAMP"
ENDPOINT_NAME="$PROJECT_NAME-endpoint"
CONFIG_NAME="$ENDPOINT_NAME-config-$TIMESTAMP"

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

# Help function
show_help() {
    echo "Complete AWS SageMaker Deployment Script for Arcee Agent"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --setup-iam         Setup IAM roles and policies"
    echo "  --upload-data       Upload dataset to S3"
    echo "  --build-images      Build and push Docker images"
    echo "  --train-model       Start SageMaker training job"
    echo "  --deploy-endpoint   Deploy model to SageMaker endpoint"
    echo "  --start-api         Start FastAPI service"
    echo "  --cleanup           Clean up AWS resources"
    echo "  --all               Run complete deployment pipeline"
    echo "  --help              Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  AWS_REGION          AWS region (default: us-east-1)"
    echo "  S3_BUCKET          S3 bucket name (required)"
    echo "  SAGEMAKER_ROLE_ARN  SageMaker execution role ARN"
    echo "  PROJECT_NAME       Project name (default: arcee-agent)"
    echo ""
    echo "Examples:"
    echo "  export S3_BUCKET=my-arcee-bucket"
    echo "  export SAGEMAKER_ROLE_ARN=arn:aws:iam::123456789012:role/ArceeAgentSageMakerRole"
    echo "  $0 --all"
    echo ""
    echo "  $0 --setup-iam"
    echo "  $0 --upload-data --train-model --deploy-endpoint"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if Python is installed
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install it first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials are not configured. Please run 'aws configure'."
        exit 1
    fi
    
    # Check required environment variables
    if [ -z "$S3_BUCKET" ]; then
        log_error "S3_BUCKET environment variable is required."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Setup IAM roles
setup_iam() {
    log_info "Setting up IAM roles and policies..."
    
    if [ -f "scripts/setup_iam_roles.sh" ]; then
        chmod +x scripts/setup_iam_roles.sh
        bash scripts/setup_iam_roles.sh
        
        # Get the role ARN
        ROLE_ARN=$(aws iam get-role --role-name ArceeAgentSageMakerRole --query 'Role.Arn' --output text)
        export SAGEMAKER_ROLE_ARN=$ROLE_ARN
        
        log_success "IAM setup completed. Role ARN: $ROLE_ARN"
    else
        log_error "IAM setup script not found at scripts/setup_iam_roles.sh"
        exit 1
    fi
}

# Upload dataset to S3
upload_data() {
    log_info "Uploading dataset to S3..."
    
    # Create S3 bucket if it doesn't exist
    if ! aws s3api head-bucket --bucket "$S3_BUCKET" 2>/dev/null; then
        log_info "Creating S3 bucket: $S3_BUCKET"
        if [ "$AWS_REGION" = "us-east-1" ]; then
            aws s3 mb "s3://$S3_BUCKET"
        else
            aws s3 mb "s3://$S3_BUCKET" --region "$AWS_REGION"
        fi
    fi
    
    # Upload dataset
    if [ -d "dataset" ]; then
        log_info "Uploading dataset to s3://$S3_BUCKET/$PROJECT_NAME/datasets/"
        aws s3 sync dataset/ "s3://$S3_BUCKET/$PROJECT_NAME/datasets/" --delete
        log_success "Dataset uploaded successfully"
    else
        log_error "Dataset directory not found"
        exit 1
    fi
}

# Build and push Docker images
build_images() {
    log_info "Building and pushing Docker images..."
    
    # Get AWS account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_REGISTRY="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
    
    # Create ECR repositories if they don't exist
    REPOSITORIES=("$PROJECT_NAME-training" "$PROJECT_NAME-inference" "$PROJECT_NAME-api")
    
    for REPO in "${REPOSITORIES[@]}"; do
        if ! aws ecr describe-repositories --repository-names "$REPO" 2>/dev/null; then
            log_info "Creating ECR repository: $REPO"
            aws ecr create-repository --repository-name "$REPO" --region "$AWS_REGION"
        fi
    done
    
    # Login to ECR
    log_info "Logging into ECR..."
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
    
    # Build and push API image
    log_info "Building API image..."
    docker build -t "$PROJECT_NAME-api:latest" -f Dockerfile .
    docker tag "$PROJECT_NAME-api:latest" "$ECR_REGISTRY/$PROJECT_NAME-api:latest"
    docker push "$ECR_REGISTRY/$PROJECT_NAME-api:latest"
    
    # Build training image (create if doesn't exist)
    if [ -f "docker/training/Dockerfile" ]; then
        log_info "Building training image..."
        docker build -t "$PROJECT_NAME-training:latest" -f docker/training/Dockerfile .
        docker tag "$PROJECT_NAME-training:latest" "$ECR_REGISTRY/$PROJECT_NAME-training:latest"
        docker push "$ECR_REGISTRY/$PROJECT_NAME-training:latest"
    else
        log_warning "Training Dockerfile not found. Using default HuggingFace training image."
    fi
    
    # Build inference image (create if doesn't exist)
    if [ -f "docker/inference/Dockerfile" ]; then
        log_info "Building inference image..."
        docker build -t "$PROJECT_NAME-inference:latest" -f docker/inference/Dockerfile .
        docker tag "$PROJECT_NAME-inference:latest" "$ECR_REGISTRY/$PROJECT_NAME-inference:latest"
        docker push "$ECR_REGISTRY/$PROJECT_NAME-inference:latest"
    else
        log_warning "Inference Dockerfile not found. Using default HuggingFace inference image."
    fi
    
    log_success "Docker images built and pushed successfully"
}

# Train model on SageMaker
train_model() {
    log_info "Starting SageMaker training job..."
    
    if [ -z "$SAGEMAKER_ROLE_ARN" ]; then
        log_error "SAGEMAKER_ROLE_ARN is not set. Please run --setup-iam first."
        exit 1
    fi
    
    # Run training script
    python3 scripts/sagemaker_training.py \
        --job-name "$JOB_NAME" \
        --role-arn "$SAGEMAKER_ROLE_ARN" \
        --s3-bucket "$S3_BUCKET" \
        --dataset-path "./dataset" \
        --region "$AWS_REGION" \
        --instance-type "ml.g4dn.xlarge" \
        --instance-count 1 \
        --max-runtime-hours 24 \
        --monitor
    
    if [ $? -eq 0 ]; then
        log_success "Training job completed successfully"
        
        # Get model artifacts URI
        MODEL_ARTIFACTS_URI=$(aws sagemaker describe-training-job \
            --training-job-name "$JOB_NAME" \
            --query 'ModelArtifacts.S3ModelArtifacts' \
            --output text)
        
        export MODEL_ARTIFACTS_URI
        log_info "Model artifacts available at: $MODEL_ARTIFACTS_URI"
    else
        log_error "Training job failed"
        exit 1
    fi
}

# Deploy model to SageMaker endpoint
deploy_endpoint() {
    log_info "Deploying model to SageMaker endpoint..."
    
    if [ -z "$SAGEMAKER_ROLE_ARN" ]; then
        log_error "SAGEMAKER_ROLE_ARN is not set. Please run --setup-iam first."
        exit 1
    fi
    
    if [ -z "$MODEL_ARTIFACTS_URI" ]; then
        # Try to find the latest model artifacts
        MODEL_ARTIFACTS_URI=$(aws s3 ls "s3://$S3_BUCKET/$PROJECT_NAME/models/" --recursive | sort | tail -1 | awk '{print "s3://'$S3_BUCKET'/" $4}')
        
        if [ -z "$MODEL_ARTIFACTS_URI" ]; then
            log_error "MODEL_ARTIFACTS_URI is not set and no model artifacts found. Please run --train-model first."
            exit 1
        fi
        
        log_info "Using model artifacts: $MODEL_ARTIFACTS_URI"
    fi
    
    # Deploy model
    python3 scripts/sagemaker_deployment.py \
        --model-name "$MODEL_NAME" \
        --endpoint-name "$ENDPOINT_NAME" \
        --role-arn "$SAGEMAKER_ROLE_ARN" \
        --model-artifacts-uri "$MODEL_ARTIFACTS_URI" \
        --region "$AWS_REGION" \
        --instance-type "ml.m5.large" \
        --instance-count 1 \
        --timeout-minutes 30
    
    if [ $? -eq 0 ]; then
        log_success "Model deployed successfully to endpoint: $ENDPOINT_NAME"
        export SAGEMAKER_ENDPOINT_NAME=$ENDPOINT_NAME
    else
        log_error "Model deployment failed"
        exit 1
    fi
}

# Start FastAPI service
start_api() {
    log_info "Starting FastAPI service..."
    
    if [ -z "$SAGEMAKER_ENDPOINT_NAME" ]; then
        SAGEMAKER_ENDPOINT_NAME=$ENDPOINT_NAME
    fi
    
    # Set environment variables
    export SAGEMAKER_ENDPOINT_NAME
    export AWS_REGION
    
    # Install dependencies if needed
    if [ -f "requirements-docker.txt" ]; then
        log_info "Installing Python dependencies..."
        pip3 install -r requirements-docker.txt
    fi
    
    # Start the API server
    log_info "Starting API server on http://localhost:8000"
    log_info "API documentation available at http://localhost:8000/docs"
    
    python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
}

# Clean up AWS resources
cleanup() {
    log_warning "Cleaning up AWS resources..."
    
    # Delete endpoint
    if aws sagemaker describe-endpoint --endpoint-name "$ENDPOINT_NAME" 2>/dev/null; then
        log_info "Deleting endpoint: $ENDPOINT_NAME"
        aws sagemaker delete-endpoint --endpoint-name "$ENDPOINT_NAME"
    fi
    
    # Delete endpoint configurations
    CONFIGS=$(aws sagemaker list-endpoint-configs --name-contains "$ENDPOINT_NAME" --query 'EndpointConfigs[].EndpointConfigName' --output text)
    for CONFIG in $CONFIGS; do
        if [ ! -z "$CONFIG" ]; then
            log_info "Deleting endpoint config: $CONFIG"
            aws sagemaker delete-endpoint-config --endpoint-config-name "$CONFIG"
        fi
    done
    
    # Delete models
    MODELS=$(aws sagemaker list-models --name-contains "$PROJECT_NAME" --query 'Models[].ModelName' --output text)
    for MODEL in $MODELS; do
        if [ ! -z "$MODEL" ]; then
            log_info "Deleting model: $MODEL"
            aws sagemaker delete-model --model-name "$MODEL"
        fi
    done
    
    # Optionally delete S3 data
    read -p "Delete S3 data in s3://$S3_BUCKET/$PROJECT_NAME/? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deleting S3 data..."
        aws s3 rm "s3://$S3_BUCKET/$PROJECT_NAME/" --recursive
    fi
    
    log_success "Cleanup completed"
}

# Main execution
main() {
    case "$1" in
        --setup-iam)
            check_prerequisites
            setup_iam
            ;;
        --upload-data)
            check_prerequisites
            upload_data
            ;;
        --build-images)
            check_prerequisites
            build_images
            ;;
        --train-model)
            check_prerequisites
            train_model
            ;;
        --deploy-endpoint)
            check_prerequisites
            deploy_endpoint
            ;;
        --start-api)
            start_api
            ;;
        --cleanup)
            cleanup
            ;;
        --all)
            check_prerequisites
            setup_iam
            upload_data
            build_images
            train_model
            deploy_endpoint
            log_success "Complete deployment pipeline finished!"
            log_info "Start the API service with: $0 --start-api"
            ;;
        --help)
            show_help
            ;;
        "")
            log_error "No option specified. Use --help for usage information."
            exit 1
            ;;
        *)
            log_error "Unknown option: $1. Use --help for usage information."
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
