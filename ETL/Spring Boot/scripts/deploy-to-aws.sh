#!/bin/bash

# AWS ETL Platform Deployment Script
# This script deploys the microservices to AWS ECS

set -e

# Configuration
REGION=${AWS_REGION:-us-east-1}
PROJECT_NAME="weather-etl"
ENVIRONMENT=${ENVIRONMENT:-dev}
ECR_REPOSITORY_PREFIX="${PROJECT_NAME}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v terraform &> /dev/null; then
        print_warning "Terraform is not installed. Infrastructure deployment will be skipped."
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials are not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    print_status "Prerequisites check completed."
}

# Get AWS account ID
get_account_id() {
    aws sts get-caller-identity --query Account --output text
}

# Create ECR repositories if they don't exist
create_ecr_repositories() {
    print_status "Creating ECR repositories..."
    
    ACCOUNT_ID=$(get_account_id)
    
    SERVICES=(
        "data-ingestion-service"
        "etl-processing-service"
        "stream-processing-service"
        "query-service"
        "dashboard-api-service"
    )
    
    for service in "${SERVICES[@]}"; do
        REPO_NAME="${ECR_REPOSITORY_PREFIX}-${service}"
        
        if aws ecr describe-repositories --repository-names $REPO_NAME --region $REGION &> /dev/null; then
            print_status "ECR repository $REPO_NAME already exists."
        else
            print_status "Creating ECR repository: $REPO_NAME"
            aws ecr create-repository \
                --repository-name $REPO_NAME \
                --region $REGION \
                --image-scanning-configuration scanOnPush=true \
                --encryption-configuration encryptionType=AES256
        fi
    done
}

# Login to ECR
ecr_login() {
    print_status "Logging in to ECR..."
    aws ecr get-login-password --region $REGION | \
        docker login --username AWS --password-stdin $(get_account_id).dkr.ecr.$REGION.amazonaws.com
}

# Build and push Docker images
build_and_push_images() {
    print_status "Building and pushing Docker images..."
    
    ACCOUNT_ID=$(get_account_id)
    ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
    
    # Build parent project first
    print_status "Building parent project..."
    mvn clean package -DskipTests
    
    SERVICES=(
        "data-ingestion-service"
        "etl-processing-service"
        "stream-processing-service"
        "query-service"
        "dashboard-api-service"
    )
    
    for service in "${SERVICES[@]}"; do
        print_status "Building and pushing $service..."
        
        SERVICE_DIR="microservices/$service"
        IMAGE_NAME="${ECR_REPOSITORY_PREFIX}-${service}"
        ECR_URI="${ECR_BASE}/${IMAGE_NAME}"
        
        # Build Docker image
        docker build -t $IMAGE_NAME:latest $SERVICE_DIR
        
        # Tag for ECR
        docker tag $IMAGE_NAME:latest $ECR_URI:latest
        docker tag $IMAGE_NAME:latest $ECR_URI:$(git rev-parse --short HEAD)
        
        # Push to ECR
        docker push $ECR_URI:latest
        docker push $ECR_URI:$(git rev-parse --short HEAD)
        
        print_status "Successfully pushed $ECR_URI"
    done
}

# Deploy infrastructure with Terraform
deploy_infrastructure() {
    if ! command -v terraform &> /dev/null; then
        print_warning "Terraform not found. Skipping infrastructure deployment."
        return
    fi
    
    print_status "Deploying infrastructure with Terraform..."
    
    cd infrastructure/terraform
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan \
        -var="aws_region=$REGION" \
        -var="environment=$ENVIRONMENT" \
        -var="project_name=$PROJECT_NAME"
    
    # Apply if user confirms
    read -p "Do you want to apply the Terraform plan? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        terraform apply \
            -var="aws_region=$REGION" \
            -var="environment=$ENVIRONMENT" \
            -var="project_name=$PROJECT_NAME" \
            -auto-approve
        
        print_status "Infrastructure deployment completed."
    else
        print_status "Infrastructure deployment skipped."
    fi
    
    cd - > /dev/null
}

# Deploy ECS services
deploy_ecs_services() {
    print_status "Deploying ECS services..."
    
    ACCOUNT_ID=$(get_account_id)
    ECR_BASE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
    CLUSTER_NAME="${PROJECT_NAME}-cluster-${ENVIRONMENT}"
    
    SERVICES=(
        "data-ingestion-service"
        "etl-processing-service"
        "stream-processing-service"
        "query-service"
        "dashboard-api-service"
    )
    
    for service in "${SERVICES[@]}"; do
        print_status "Deploying ECS service: $service"
        
        SERVICE_NAME="${PROJECT_NAME}-${service}-${ENVIRONMENT}"
        IMAGE_URI="${ECR_BASE}/${ECR_REPOSITORY_PREFIX}-${service}:latest"
        
        # Create task definition
        TASK_DEF_FILE="ecs/${service}-task-definition.json"
        
        if [ -f "$TASK_DEF_FILE" ]; then
            # Replace placeholders in task definition
            sed -e "s|\${AWS_ACCOUNT_ID}|$ACCOUNT_ID|g" \
                -e "s|\${AWS_REGION}|$REGION|g" \
                -e "s|\${IMAGE_URI}|$IMAGE_URI|g" \
                -e "s|\${ENVIRONMENT}|$ENVIRONMENT|g" \
                -e "s|\${PROJECT_NAME}|$PROJECT_NAME|g" \
                $TASK_DEF_FILE > /tmp/${service}-task-definition.json
            
            # Register task definition
            aws ecs register-task-definition \
                --cli-input-json file:///tmp/${service}-task-definition.json \
                --region $REGION
            
            # Update service or create if it doesn't exist
            if aws ecs describe-services \
                --cluster $CLUSTER_NAME \
                --services $SERVICE_NAME \
                --region $REGION &> /dev/null; then
                
                print_status "Updating existing service: $SERVICE_NAME"
                aws ecs update-service \
                    --cluster $CLUSTER_NAME \
                    --service $SERVICE_NAME \
                    --task-definition $SERVICE_NAME \
                    --region $REGION
            else
                print_status "Creating new service: $SERVICE_NAME"
                # Service creation would require additional parameters
                # This is a simplified version
                print_warning "Service creation requires additional configuration. Please use the AWS console or create service definitions."
            fi
        else
            print_warning "Task definition file not found: $TASK_DEF_FILE"
        fi
    done
}

# Deploy React dashboard to S3
deploy_react_dashboard() {
    print_status "Deploying React dashboard..."
    
    cd frontend/dashboard
    
    # Install dependencies and build
    if [ -f "package.json" ]; then
        npm install
        npm run build
        
        # Deploy to S3 (requires bucket to be created first)
        BUCKET_NAME="${PROJECT_NAME}-dashboard-${ENVIRONMENT}"
        
        if aws s3 ls "s3://$BUCKET_NAME" 2>&1 | grep -q 'NoSuchBucket'; then
            print_status "Creating S3 bucket for dashboard: $BUCKET_NAME"
            aws s3 mb s3://$BUCKET_NAME --region $REGION
            
            # Configure bucket for static website hosting
            aws s3 website s3://$BUCKET_NAME \
                --index-document index.html \
                --error-document error.html
        fi
        
        # Upload build files
        aws s3 sync build/ s3://$BUCKET_NAME --delete
        
        print_status "React dashboard deployed to: http://$BUCKET_NAME.s3-website-$REGION.amazonaws.com"
    else
        print_warning "package.json not found. Skipping React dashboard deployment."
    fi
    
    cd - > /dev/null
}

# Main deployment function
main() {
    print_status "Starting AWS ETL Platform deployment..."
    
    check_prerequisites
    create_ecr_repositories
    ecr_login
    build_and_push_images
    deploy_infrastructure
    deploy_ecs_services
    deploy_react_dashboard
    
    print_status "Deployment completed successfully!"
    print_status "Next steps:"
    echo "1. Configure your weather stations to send data to the MQTT broker"
    echo "2. Monitor the ECS services in the AWS console"
    echo "3. Check CloudWatch logs for any issues"
    echo "4. Access the dashboard at the S3 website URL"
}

# Run main function
main "$@"
