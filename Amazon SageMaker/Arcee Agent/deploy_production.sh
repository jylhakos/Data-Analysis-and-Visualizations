#!/bin/bash
# deploy_production.sh - Production deployment script for Arcee Agent

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
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

# Configuration
PROJECT_NAME="arcee-agent"
IMAGE_NAME="arcee-agent-api"
CONTAINER_NAME="arcee-agent-production"
API_PORT="8000"
ENVIRONMENT=${ENVIRONMENT:-production}

echo "🚀 Arcee Agent Production Deployment"
echo "====================================="
echo ""

# Parse command line arguments
COMMAND=${1:-help}

case $COMMAND in
    "build")
        echo "📦 Building Production Docker Image"
        echo "=================================="
        
        # Build the Docker image
        log_info "Building Docker image..."
        docker build -t ${IMAGE_NAME}:latest .
        
        # Tag with timestamp
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        docker tag ${IMAGE_NAME}:latest ${IMAGE_NAME}:${TIMESTAMP}
        
        log_success "Docker image built successfully"
        log_info "Tagged as: ${IMAGE_NAME}:latest and ${IMAGE_NAME}:${TIMESTAMP}"
        
        # Show image size
        SIZE=$(docker images ${IMAGE_NAME}:latest --format "table {{.Size}}" | tail -n 1)
        log_info "Image size: $SIZE"
        ;;
        
    "deploy")
        echo "🚀 Deploying to Production"
        echo "=========================="
        
        # Check if .env.production exists
        if [ ! -f .env.production ]; then
            log_error ".env.production file not found"
            log_info "Creating template .env.production file..."
            
            cat << 'EOF' > .env.production
# Production Environment Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
WORKERS=4

# AWS Configuration (REQUIRED)
AWS_DEFAULT_REGION=us-east-1
SAGEMAKER_ENDPOINT_NAME=arcee-agent-endpoint
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
S3_BUCKET_NAME=your-s3-bucket-name

# Model Configuration
MODEL_NAME=arcee-agent-finetuned
MAX_TOKENS=512
TEMPERATURE=0.1
TOP_P=0.9

# Performance Settings
MAX_CONCURRENT_REQUESTS=100
REQUEST_TIMEOUT=30
CACHE_SIZE=1000

# Monitoring (optional)
GRAFANA_PASSWORD=secure-password-here
EOF
            
            log_warning "Please edit .env.production with your actual values"
            log_warning "Then run: ./deploy_production.sh deploy"
            exit 1
        fi
        
        # Load environment variables
        source .env.production
        
        # Validate required environment variables
        REQUIRED_VARS=("SAGEMAKER_ENDPOINT_NAME" "AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY")
        for var in "${REQUIRED_VARS[@]}"; do
            if [ -z "${!var}" ]; then
                log_error "Required environment variable $var is not set"
                exit 1
            fi
        done
        
        log_success "Environment variables validated"
        
        # Create necessary directories
        log_info "Creating necessary directories..."
        mkdir -p logs models config nginx/ssl monitoring/grafana/{dashboards,datasources}
        
        # Build image if it doesn't exist
        if ! docker images ${IMAGE_NAME}:latest | grep -q ${IMAGE_NAME}; then
            log_info "Docker image not found, building..."
            ./deploy_production.sh build
        fi
        
        # Deploy with Docker Compose
        log_info "Deploying with Docker Compose..."
        docker-compose -f docker-compose.prod.yml --env-file .env.production up -d
        
        # Wait for services to be ready
        log_info "Waiting for services to be ready..."
        sleep 10
        
        # Health check
        for i in {1..30}; do
            if curl -s -f http://localhost:${API_PORT}/health >/dev/null 2>&1; then
                log_success "API service is healthy"
                break
            else
                if [ $i -eq 30 ]; then
                    log_error "API service health check failed after 30 attempts"
                    exit 1
                fi
                log_info "Waiting for API service... (attempt $i/30)"
                sleep 2
            fi
        done
        
        # Show deployment status
        echo ""
        log_success "🎉 Deployment completed successfully!"
        echo ""
        echo "📋 Service URLs:"
        echo "  • API: http://localhost:${API_PORT}"
        echo "  • Health: http://localhost:${API_PORT}/health"
        echo "  • Docs: http://localhost:${API_PORT}/docs"
        echo "  • Metrics: http://localhost:${API_PORT}/metrics"
        echo "  • Grafana: http://localhost:3000 (admin/admin)"
        echo "  • Prometheus: http://localhost:9090"
        echo ""
        echo "📊 Container Status:"
        docker-compose -f docker-compose.prod.yml ps
        ;;
        
    "test")
        echo "🧪 Testing Production Deployment"
        echo "==============================="
        
        # Test API health
        log_info "Testing API health..."
        if curl -s -f http://localhost:${API_PORT}/health >/dev/null 2>&1; then
            log_success "Health check passed"
        else
            log_error "Health check failed"
            exit 1
        fi
        
        # Test function calling
        log_info "Testing function calling..."
        RESPONSE=$(curl -s -X POST "http://localhost:${API_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{
                "model": "arcee-agent-finetuned",
                "messages": [
                    {
                        "role": "user", 
                        "content": "You have access to this tool: [{\"name\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"location\": {\"type\": \"string\"}}}]. What tool should I call for: \"Weather in Paris\"?"
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 256
            }')
        
        if echo "$RESPONSE" | grep -q "choices"; then
            log_success "Function calling test passed"
            log_info "Response received: $(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null || echo "$RESPONSE")"
        else
            log_error "Function calling test failed"
            log_error "Response: $RESPONSE"
            exit 1
        fi
        
        # Test metrics endpoint
        log_info "Testing metrics endpoint..."
        if curl -s -f http://localhost:${API_PORT}/metrics >/dev/null 2>&1; then
            log_success "Metrics endpoint accessible"
        else
            log_warning "Metrics endpoint not accessible"
        fi
        
        log_success "All tests passed! 🎉"
        ;;
        
    "logs")
        echo "📜 Viewing Application Logs"
        echo "=========================="
        
        SERVICE=${2:-arcee-agent-api}
        log_info "Showing logs for service: $SERVICE"
        docker-compose -f docker-compose.prod.yml logs -f --tail=100 $SERVICE
        ;;
        
    "status")
        echo "📊 Deployment Status"
        echo "==================="
        
        echo ""
        log_info "Container Status:"
        docker-compose -f docker-compose.prod.yml ps
        
        echo ""
        log_info "Resource Usage:"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"
        
        echo ""
        log_info "API Health:"
        if curl -s -f http://localhost:${API_PORT}/health >/dev/null 2>&1; then
            HEALTH_RESPONSE=$(curl -s http://localhost:${API_PORT}/health)
            log_success "API is healthy: $HEALTH_RESPONSE"
        else
            log_error "API is not responding"
        fi
        ;;
        
    "scale")
        REPLICAS=${2:-2}
        echo "📈 Scaling API Service"
        echo "===================="
        
        log_info "Scaling arcee-agent-api to $REPLICAS replicas..."
        docker-compose -f docker-compose.prod.yml up -d --scale arcee-agent-api=$REPLICAS
        
        log_success "Scaling completed"
        docker-compose -f docker-compose.prod.yml ps arcee-agent-api
        ;;
        
    "update")
        echo "🔄 Updating Production Deployment"
        echo "================================"
        
        # Build new image
        log_info "Building updated image..."
        ./deploy_production.sh build
        
        # Rolling update
        log_info "Performing rolling update..."
        docker-compose -f docker-compose.prod.yml up -d --no-deps arcee-agent-api
        
        # Wait for health check
        sleep 10
        if curl -s -f http://localhost:${API_PORT}/health >/dev/null 2>&1; then
            log_success "Update completed successfully"
        else
            log_error "Update failed - service not healthy"
            exit 1
        fi
        ;;
        
    "stop")
        echo "🛑 Stopping Production Deployment"
        echo "================================"
        
        log_info "Stopping all services..."
        docker-compose -f docker-compose.prod.yml down
        
        log_success "All services stopped"
        ;;
        
    "cleanup")
        echo "🧹 Cleaning Up"
        echo "=============="
        
        log_warning "This will remove all containers, images, and volumes!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Stopping services..."
            docker-compose -f docker-compose.prod.yml down -v --rmi all
            
            # Clean up unused images and containers
            log_info "Cleaning up unused Docker resources..."
            docker system prune -f
            
            log_success "Cleanup completed"
        else
            log_info "Cleanup cancelled"
        fi
        ;;
        
    "backup")
        echo "💾 Creating Backup"
        echo "================="
        
        BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
        log_info "Creating backup in: $BACKUP_DIR"
        
        mkdir -p $BACKUP_DIR
        
        # Backup configuration
        cp -r .env.production docker-compose.prod.yml config logs $BACKUP_DIR/ 2>/dev/null || true
        
        # Backup Docker images
        log_info "Backing up Docker images..."
        docker save ${IMAGE_NAME}:latest | gzip > $BACKUP_DIR/${IMAGE_NAME}_latest.tar.gz
        
        # Create backup info
        cat << EOF > $BACKUP_DIR/backup_info.txt
Backup created: $(date)
Project: $PROJECT_NAME
Image: ${IMAGE_NAME}:latest
Environment: $ENVIRONMENT
Docker Compose version: $(docker-compose --version)
Docker version: $(docker --version)
EOF
        
        log_success "Backup created in: $BACKUP_DIR"
        ;;
        
    "help"|*)
        echo "📖 Arcee Agent Production Deployment Commands"
        echo "============================================="
        echo ""
        echo "Usage: $0 [COMMAND] [OPTIONS]"
        echo ""
        echo "Commands:"
        echo "  build                 Build production Docker image"
        echo "  deploy                Deploy to production environment"
        echo "  test                  Run production tests"
        echo "  logs [service]        View logs (default: arcee-agent-api)"
        echo "  status                Show deployment status"
        echo "  scale [replicas]      Scale API service (default: 2)"
        echo "  update                Update deployment with new code"
        echo "  stop                  Stop all services"
        echo "  cleanup               Remove all containers and images"
        echo "  backup                Create backup of deployment"
        echo "  help                  Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 build              # Build Docker image"
        echo "  $0 deploy             # Deploy to production"
        echo "  $0 test               # Test the deployment"
        echo "  $0 logs api           # View API logs"
        echo "  $0 scale 4            # Scale to 4 API replicas"
        echo "  $0 status             # Check deployment status"
        echo ""
        echo "Prerequisites:"
        echo "  • Docker and Docker Compose installed"
        echo "  • AWS credentials configured"
        echo "  • SageMaker endpoint deployed"
        echo "  • .env.production file configured"
        echo ""
        ;;
esac
