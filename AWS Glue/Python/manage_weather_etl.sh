#!/bin/bash

# AWS Glue Weather ETL Management Script
# =====================================
# This script manages the complete weather ETL pipeline with AWS Glue integration
# including local development, testing, and AWS deployment.

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="weather-etl-glue"
AWS_REGION="us-east-1"
S3_BUCKET="weather-data-lake"
DOCKER_COMPOSE_FILE="docker-compose.yml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $1"; }

show_help() {
    cat << EOF
AWS Glue Weather ETL Management Script

USAGE:
    $0 <command> [options]

COMMANDS:
    Local Development:
        start           Start all services locally with Docker Compose
        stop            Stop all local services
        restart         Restart all local services
        logs <service>  Show logs for a specific service
        status          Show status of all services
        clean           Clean up Docker containers and volumes

    AWS Glue Operations:
        deploy-aws      Deploy AWS Glue infrastructure
        upload-scripts  Upload Glue scripts to S3
        trigger-batch   Trigger batch ETL job
        start-streaming Start streaming ETL job
        job-status      Check Glue job status
        setup-catalog   Setup Data Catalog

    Data Operations:
        simulate        Start weather data simulation
        test-pipeline   Test the complete ETL pipeline
        validate-data   Validate processed data quality

    Development:
        build           Build Docker images
        test            Run test suite
        format          Format Python code
        lint            Run code linting

OPTIONS:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    -e, --env       Environment (development|staging|production)

EXAMPLES:
    $0 start                    # Start local development environment
    $0 deploy-aws              # Deploy to AWS
    $0 trigger-batch           # Trigger batch processing
    $0 logs enhanced-etl       # Show logs for enhanced ETL service
    $0 test-pipeline           # Test complete pipeline

EOF
}

# Environment setup
setup_environment() {
    log_info "Setting up environment..."
    
    # Create directories if they don't exist
    mkdir -p logs data checkpoints aws_credentials
    
    # Create .env file if it doesn't exist
    if [[ ! -f .env ]]; then
        log_info "Creating .env file..."
        cat > .env << EOF
# Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO

# Database
DATABASE_URL=postgresql://weather_user:weather_pass@localhost:5432/weather_db
REDIS_URL=redis://localhost:6379/0

# AWS Configuration
AWS_REGION=$AWS_REGION
S3_BUCKET_NAME=$S3_BUCKET

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# gRPC Configuration
GRPC_HOST=localhost
GRPC_PORT=50051
EOF
    fi
    
    log_info "Environment setup completed."
}

# Docker operations
docker_start() {
    log_info "Starting weather ETL services..."
    setup_environment
    
    # Build images if they don't exist
    if ! docker images | grep -q "${PROJECT_NAME}"; then
        log_info "Building Docker images..."
        docker-compose build
    fi
    
    # Start services
    docker-compose up -d
    
    log_info "Services started. Waiting for health checks..."
    sleep 10
    
    docker_status
}

docker_stop() {
    log_info "Stopping weather ETL services..."
    docker-compose down
    log_info "Services stopped."
}

docker_restart() {
    docker_stop
    sleep 2
    docker_start
}

docker_status() {
    log_info "Service Status:"
    echo "===================="
    docker-compose ps
    echo "===================="
    
    # Check health endpoints
    log_info "Health Checks:"
    services=("api_gateway:8000" "enhanced-etl-processing:50054")
    
    for service in "${services[@]}"; do
        IFS=':' read -r name port <<< "$service"
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓${NC} $name (port $port)"
        else
            echo -e "${RED}✗${NC} $name (port $port)"
        fi
    done
}

docker_logs() {
    local service=${1:-}
    if [[ -z $service ]]; then
        log_error "Please specify a service name"
        docker-compose ps --services
        return 1
    fi
    
    log_info "Showing logs for service: $service"
    docker-compose logs -f "$service"
}

docker_clean() {
    log_warn "This will remove all containers, volumes, and images. Continue? (y/N)"
    read -r response
    if [[ $response =~ ^[Yy]$ ]]; then
        log_info "Cleaning up Docker resources..."
        docker-compose down -v --rmi all
        docker system prune -f
        log_info "Cleanup completed."
    else
        log_info "Cleanup cancelled."
    fi
}

# AWS Glue operations
deploy_aws_infrastructure() {
    log_info "Deploying AWS Glue infrastructure..."
    
    # Check AWS credentials
    if ! aws sts get-caller-identity > /dev/null 2>&1; then
        log_error "AWS credentials not configured. Please run 'aws configure'"
        return 1
    fi
    
    # Run deployment script
    if [[ -f "aws_glue/deploy_glue_infrastructure.sh" ]]; then
        bash aws_glue/deploy_glue_infrastructure.sh
    else
        log_error "Deployment script not found: aws_glue/deploy_glue_infrastructure.sh"
        return 1
    fi
}

upload_glue_scripts() {
    log_info "Uploading Glue scripts to S3..."
    
    # Upload ETL scripts
    aws s3 cp aws_glue/weather_etl_glue_job.py "s3://$S3_BUCKET/scripts/glue_jobs/"
    aws s3 cp aws_glue/weather_streaming_etl_job.py "s3://$S3_BUCKET/scripts/glue_jobs/"
    
    log_info "Scripts uploaded successfully."
}

trigger_batch_job() {
    log_info "Triggering AWS Glue batch ETL job..."
    
    JOB_NAME="weather-batch-etl-job"
    
    JOB_RUN_ID=$(aws glue start-job-run \
        --job-name "$JOB_NAME" \
        --arguments '{"--S3_BUCKET":"'$S3_BUCKET'","--DATABASE_NAME":"weather_analytics_db"}' \
        --query 'JobRunId' \
        --output text)
    
    log_info "Batch job triggered. Job Run ID: $JOB_RUN_ID"
    log_info "Monitor progress: aws glue get-job-run --job-name $JOB_NAME --run-id $JOB_RUN_ID"
}

start_streaming_job() {
    log_info "Starting AWS Glue streaming ETL job..."
    
    JOB_NAME="weather-streaming-etl-job"
    
    JOB_RUN_ID=$(aws glue start-job-run \
        --job-name "$JOB_NAME" \
        --arguments '{"--KINESIS_STREAM_NAME":"weather-data-stream","--S3_BUCKET":"'$S3_BUCKET'"}' \
        --query 'JobRunId' \
        --output text)
    
    log_info "Streaming job started. Job Run ID: $JOB_RUN_ID"
    log_info "Monitor progress: aws glue get-job-run --job-name $JOB_NAME --run-id $JOB_RUN_ID"
}

check_job_status() {
    log_info "Checking AWS Glue job status..."
    
    echo "Available jobs:"
    aws glue list-jobs --query 'JobNames' --output table
    
    echo -e "\nRecent job runs:"
    aws glue get-job-runs --job-name "weather-batch-etl-job" --max-results 5 \
        --query 'JobRuns[].{JobRunId:Id,State:JobRunState,StartedOn:StartedOn,CompletedOn:CompletedOn}' \
        --output table 2>/dev/null || log_warn "No batch job runs found"
}

setup_data_catalog() {
    log_info "Setting up AWS Glue Data Catalog..."
    python3 aws_glue/setup_data_catalog.py
}

# Data operations
simulate_weather_data() {
    log_info "Starting weather data simulation..."
    
    if docker-compose ps | grep -q "running"; then
        # Run simulation inside Docker
        docker-compose exec data-ingestion python3 simulate_weather_stations.py
    else
        # Run simulation locally
        python3 simulate_weather_stations.py
    fi
}

test_pipeline() {
    log_info "Testing complete ETL pipeline..."
    
    # Start services if not running
    if ! docker-compose ps | grep -q "running"; then
        log_info "Starting services for testing..."
        docker_start
        sleep 30  # Wait for services to be ready
    fi
    
    # Generate test data
    log_info "Generating test weather data..."
    python3 -c "
import requests
import json
from datetime import datetime

# Test data
test_data = {
    'station_id': 'TEST_STATION_001',
    'timestamp': datetime.now().isoformat(),
    'temperature': 25.5,
    'humidity': 65.0,
    'pressure': 1013.25,
    'wind_speed': 15.0,
    'wind_direction': 180.0,
    'weather_condition': 'Clear'
}

# Send to API
try:
    response = requests.post('http://localhost:8000/api/v1/weather', json=test_data)
    print(f'API Response: {response.status_code} - {response.text}')
except Exception as e:
    print(f'API Test failed: {e}')
"
    
    log_info "Pipeline test completed. Check logs for results."
}

validate_data() {
    log_info "Validating processed data quality..."
    
    # Check S3 for processed data
    aws s3 ls "s3://$S3_BUCKET/processed/weather_data/" --recursive | head -10
    
    # Check Data Catalog tables
    aws glue get-tables --database-name weather_analytics_db \
        --query 'TableList[].{Name:Name,UpdateTime:UpdateTime}' \
        --output table
}

# Development operations
build_images() {
    log_info "Building Docker images..."
    docker-compose build --no-cache
    log_info "Build completed."
}

run_tests() {
    log_info "Running test suite..."
    
    # Install dependencies if needed
    if [[ ! -d "venv" ]]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi
    
    # Run tests
    pytest tests/ -v --cov=services --cov-report=html
    
    log_info "Tests completed. Coverage report available in htmlcov/"
}

format_code() {
    log_info "Formatting Python code..."
    black services/ aws_glue/ tests/
    log_info "Code formatting completed."
}

lint_code() {
    log_info "Running code linting..."
    flake8 services/ aws_glue/ tests/
    mypy services/ aws_glue/
    log_info "Linting completed."
}

# Main command dispatcher
main() {
    local command=${1:-}
    
    case $command in
        # Local development
        start)
            docker_start
            ;;
        stop)
            docker_stop
            ;;
        restart)
            docker_restart
            ;;
        logs)
            docker_logs "$2"
            ;;
        status)
            docker_status
            ;;
        clean)
            docker_clean
            ;;
        
        # AWS Glue operations
        deploy-aws)
            deploy_aws_infrastructure
            ;;
        upload-scripts)
            upload_glue_scripts
            ;;
        trigger-batch)
            trigger_batch_job
            ;;
        start-streaming)
            start_streaming_job
            ;;
        job-status)
            check_job_status
            ;;
        setup-catalog)
            setup_data_catalog
            ;;
        
        # Data operations
        simulate)
            simulate_weather_data
            ;;
        test-pipeline)
            test_pipeline
            ;;
        validate-data)
            validate_data
            ;;
        
        # Development
        build)
            build_images
            ;;
        test)
            run_tests
            ;;
        format)
            format_code
            ;;
        lint)
            lint_code
            ;;
        
        # Help
        -h|--help|help)
            show_help
            ;;
        
        *)
            log_error "Unknown command: $command"
            echo "Use '$0 --help' for usage information."
            exit 1
            ;;
    esac
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
