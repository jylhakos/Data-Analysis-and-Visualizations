#!/bin/bash

# 🐟 Fish weight prediction production deployment script
# Deploy the most meaningful components for fish weight prediction business value

set -e  # Exit on any error

echo "🐟 Fish Weight Prediction - Production Deployment"
echo "================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Docker is running
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker first."
        exit 1
    fi
    
    print_status "Docker is ready ✓"
}

# Check if Docker Compose is available
check_docker_compose() {
    print_status "Checking Docker Compose..."
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_status "Docker Compose is ready ✓"
}

# Create necessary directories
create_directories() {
    print_status "Creating necessary directories for fish prediction data..."
    
    mkdir -p data/postgres
    mkdir -p data/redis
    mkdir -p data/fish_models
    mkdir -p logs/fish
    
    # Set permissions
    chmod -R 755 data
    chmod -R 755 logs
    
    print_status "Directories created ✓"
}

# Validate fish dataset
validate_fish_dataset() {
    print_status "Validating Fish.csv dataset..."
    
    if [ ! -f "Dataset/Fish.csv" ]; then
        print_error "Fish.csv not found in Dataset/ directory!"
        print_warning "Please ensure Fish.csv is available for fish weight prediction."
        exit 1
    fi
    
    # Check if dataset has data
    lines=$(wc -l < Dataset/Fish.csv)
    if [ $lines -lt 2 ]; then
        print_error "Fish.csv appears to be empty or invalid!"
        exit 1
    fi
    
    print_status "Fish.csv validated - $lines lines found ✓"
}

# Deploy fish weight prediction stack
deploy_fish_stack() {
    print_header "🚀 Deploying Fish Weight Prediction Stack"
    
    # Stop any existing containers
    print_status "Stopping existing containers..."
    docker-compose -f docker-compose.fish-production.yml down || true
    
    # Build and start services
    print_status "Building and starting fish prediction services..."
    docker-compose -f docker-compose.fish-production.yml up -d --build
    
    print_status "Fish prediction stack deployed ✓"
}

# Wait for services to be healthy
wait_for_services() {
    print_status "Waiting for fish prediction services to be ready..."
    
    # Wait for PostgreSQL
    print_status "Waiting for PostgreSQL database..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose -f docker-compose.fish-production.yml exec -T postgres pg_isready -U airflow &> /dev/null; then
            print_status "PostgreSQL is ready ✓"
            break
        fi
        sleep 2
        ((timeout-=2))
    done
    
    if [ $timeout -le 0 ]; then
        print_error "PostgreSQL failed to start within 60 seconds"
        exit 1
    fi
    
    # Wait for Redis
    print_status "Waiting for Redis..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if docker-compose -f docker-compose.fish-production.yml exec -T redis redis-cli ping | grep PONG &> /dev/null; then
            print_status "Redis is ready ✓"
            break
        fi
        sleep 2
        ((timeout-=2))
    done
    
    # Wait for Fish API
    print_status "Waiting for Fish Weight Prediction API..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8000/health &> /dev/null; then
            print_status "Fish API is ready ✓"
            break
        fi
        sleep 3
        ((timeout-=3))
    done
    
    # Wait for Airflow
    print_status "Waiting for Airflow webserver..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -s http://localhost:8080/health &> /dev/null; then
            print_status "Airflow is ready ✓"
            break
        fi
        sleep 5
        ((timeout-=5))
    done
}

# Test fish weight prediction API
test_fish_api() {
    print_header "🧪 Testing Fish Weight Prediction API"
    
    # Test health endpoint
    print_status "Testing API health endpoint..."
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        print_status "Fish API health check passed ✓"
    else
        print_warning "Fish API health check failed - check logs"
    fi
    
    # Test fish species endpoint
    print_status "Testing fish species endpoint..."
    if curl -s http://localhost:8000/species | grep -q "Bream"; then
        print_status "Fish species endpoint working ✓"
    else
        print_warning "Fish species endpoint issues - check logs"
    fi
    
    # Test fish weight prediction (sample Bream)
    print_status "Testing fish weight prediction with sample data..."
    response=$(curl -s -X POST "http://localhost:8000/predict/fish-weight" \
        -H "Content-Type: application/json" \
        -d '{
            "species": "Bream",
            "length1": 23.2,
            "length2": 25.4,
            "length3": 30.0,
            "height": 11.52,
            "width": 4.02
        }' || echo "FAILED")
    
    if echo "$response" | grep -q "predicted_weight"; then
        predicted_weight=$(echo "$response" | grep -o '"predicted_weight":[0-9.]*' | cut -d: -f2)
        print_status "Fish weight prediction successful: ${predicted_weight}g ✓"
    else
        print_warning "Fish weight prediction test failed - model may not be loaded yet"
        print_warning "Response: $response"
    fi
}

# Display deployment summary
show_deployment_summary() {
    print_header "🎉 Fish Weight Prediction Deployment Complete!"
    echo ""
    echo "🐟 Fish Weight Prediction Services:"
    echo "   📡 Fish Weight API:    http://localhost:8000"
    echo "   📊 API Documentation: http://localhost:8000/docs"
    echo "   🌐 Airflow Web UI:    http://localhost:8080 (admin/admin)"
    echo "   🗄️  PostgreSQL:        localhost:5432"
    echo "   📬 Redis:             localhost:6379"
    echo ""
    echo "🎯 Quick Fish Weight Prediction Test:"
    echo '   curl -X POST "http://localhost:8000/predict/fish-weight" \'
    echo '     -H "Content-Type: application/json" \'
    echo '     -d '"'"'{"species":"Bream","length1":23.2,"length2":25.4,"length3":30.0,"height":11.52,"width":4.02}'"'"
    echo ""
    echo "📊 View Fish Analysis DAG:"
    echo "   1. Open http://localhost:8080"
    echo "   2. Login with admin/admin"  
    echo "   3. Enable 'fish_weight_prediction_pipeline' DAG"
    echo "   4. Trigger DAG run for fish model training"
    echo ""
    echo "🔍 Monitor Fish Prediction Services:"
    echo "   docker-compose -f docker-compose.fish-production.yml logs -f"
    echo ""
    echo "🛑 Stop Fish Prediction Services:"
    echo "   docker-compose -f docker-compose.fish-production.yml down"
}

# Main deployment process
main() {
    print_header "🐟 Starting Fish Weight Prediction Production Deployment"
    
    # Pre-deployment checks
    check_docker
    check_docker_compose
    validate_fish_dataset
    create_directories
    
    # Deploy the stack
    deploy_fish_stack
    
    # Post-deployment validation
    wait_for_services
    test_fish_api
    
    # Show results
    show_deployment_summary
    
    print_status "Fish Weight Prediction system is ready for business! 🚀"
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy"|"start")
        main
        ;;
    "stop")
        print_status "Stopping Fish Weight Prediction services..."
        docker-compose -f docker-compose.fish-production.yml down
        print_status "Fish prediction services stopped ✓"
        ;;
    "restart")
        print_status "Restarting Fish Weight Prediction services..."
        docker-compose -f docker-compose.fish-production.yml restart
        print_status "Fish prediction services restarted ✓"
        ;;
    "logs")
        print_status "Showing Fish Weight Prediction logs..."
        docker-compose -f docker-compose.fish-production.yml logs -f
        ;;
    "status")
        print_status "Fish Weight Prediction service status:"
        docker-compose -f docker-compose.fish-production.yml ps
        ;;
    "test")
        test_fish_api
        ;;
    *)
        echo "Usage: $0 {deploy|start|stop|restart|logs|status|test}"
        echo ""
        echo "Commands:"
        echo "  deploy/start - Deploy fish weight prediction stack"
        echo "  stop         - Stop all fish prediction services"
        echo "  restart      - Restart fish prediction services"
        echo "  logs         - Show service logs"
        echo "  status       - Show service status"
        echo "  test         - Test fish weight prediction API"
        exit 1
        ;;
esac
