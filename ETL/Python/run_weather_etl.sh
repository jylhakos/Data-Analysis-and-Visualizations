#!/bin/bash

# Weather ETL System Runner Script
# This script helps you start the entire weather ETL system step by step

set -e

PROJECT_DIR="/home/laptop/EXERCISES/Data Analysis and Visualization/Data-Analysis-and-Visualizations/ETL/Python"
cd "$PROJECT_DIR"

# Colors for output
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

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    netstat -tuln | grep ":$1 " >/dev/null 2>&1
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name=$1
    local port=$2
    local max_attempts=30
    local attempt=1
    
    print_status "Waiting for $service_name to be ready on port $port..."
    
    while [ $attempt -le $max_attempts ]; do
        if port_in_use $port; then
            print_status "$service_name is ready!"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    print_error "$service_name failed to start within 60 seconds"
    return 1
}

# Function to check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    local missing_deps=()
    
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi
    
    if ! command_exists docker; then
        missing_deps+=("docker")
    fi
    
    if ! command_exists docker-compose; then
        missing_deps+=("docker-compose")
    fi
    
    if ! command_exists node; then
        missing_deps+=("node")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_error "Please install them first using SETUP_GUIDE.md"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        print_warning "Python virtual environment not found. Creating one..."
        python3 -m venv venv
    fi
    
    print_status "All prerequisites met!"
}

# Function to setup environment
setup_environment() {
    print_step "Setting up environment..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install Python dependencies if needed
    if [ ! -f "venv/.dependencies_installed" ]; then
        print_status "Installing Python dependencies..."
        pip install --upgrade pip
        pip install -r requirements.txt
        touch venv/.dependencies_installed
    fi
    
    # Generate gRPC code if needed
    if [ ! -f "weather_pb2.py" ]; then
        print_status "Generating gRPC code..."
        python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. proto/weather.proto
        touch proto/__init__.py
    fi
    
    # Check if .env exists
    if [ ! -f ".env" ]; then
        print_warning ".env file not found. Copying from .env.example..."
        cp .env.example .env
        print_warning "Please edit .env file with your configuration before proceeding"
        read -p "Press Enter after editing .env file..."
    fi
    
    print_status "Environment setup complete!"
}

# Function to start infrastructure
start_infrastructure() {
    print_step "Starting infrastructure services..."
    
    # Start Docker services
    print_status "Starting PostgreSQL, Redis, and MQTT broker..."
    docker-compose up -d postgres redis mosquitto
    
    # Wait for services
    wait_for_service "PostgreSQL" 5432
    wait_for_service "Redis" 6379
    wait_for_service "MQTT Broker" 1883
    
    # Initialize database
    print_status "Initializing database..."
    source venv/bin/activate
    python -c "
try:
    from services.data_storage import DatabaseManager
    db = DatabaseManager()
    print('✅ Database tables created successfully')
except Exception as e:
    print(f'❌ Database initialization failed: {e}')
    exit(1)
"
    
    print_status "Infrastructure services started successfully!"
}

# Function to start microservices
start_microservices() {
    print_step "Starting microservices..."
    
    source venv/bin/activate
    
    # Create logs directory
    mkdir -p logs
    
    # Start services in background
    print_status "Starting Data Ingestion Service..."
    nohup python -m services.data_ingestion > logs/data_ingestion.log 2>&1 &
    echo $! > logs/data_ingestion.pid
    
    print_status "Starting ETL Processing Service..."
    nohup python -m services.etl_processing > logs/etl_processing.log 2>&1 &
    echo $! > logs/etl_processing.pid
    
    print_status "Starting Data Storage Service..."
    nohup python -m services.data_storage > logs/data_storage.log 2>&1 &
    echo $! > logs/data_storage.pid
    
    print_status "Starting API Gateway..."
    nohup python -m services.api_gateway > logs/api_gateway.log 2>&1 &
    echo $! > logs/api_gateway.pid
    
    # Wait for services to start
    sleep 5
    wait_for_service "gRPC Data Ingestion" 50051
    wait_for_service "gRPC ETL Processing" 50052
    wait_for_service "gRPC Data Storage" 50053
    wait_for_service "API Gateway" 8000
    
    print_status "All microservices started successfully!"
}

# Function to start dashboard
start_dashboard() {
    print_step "Starting dashboard..."
    
    cd dashboard
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        print_status "Installing Node.js dependencies..."
        npm install
    fi
    
    # Start development server
    print_status "Starting Next.js development server..."
    nohup npm run dev > ../logs/dashboard.log 2>&1 &
    echo $! > ../logs/dashboard.pid
    
    cd ..
    
    wait_for_service "Dashboard" 3000
    print_status "Dashboard started successfully!"
}

# Function to start weather simulator
start_simulator() {
    print_step "Starting weather station simulator..."
    
    source venv/bin/activate
    
    # Test MQTT connection first
    python simulate_weather_stations.py --test
    
    if [ $? -eq 0 ]; then
        print_status "Starting weather data simulation (5 minute intervals)..."
        nohup python simulate_weather_stations.py --duration 60 --interval 300 > logs/simulator.log 2>&1 &
        echo $! > logs/simulator.pid
        print_status "Weather simulator started!"
    else
        print_error "MQTT connection test failed. Cannot start simulator."
        return 1
    fi
}

# Function to show status
show_status() {
    print_step "System Status"
    echo
    
    # Check infrastructure
    echo "Infrastructure Services:"
    docker-compose ps postgres redis mosquitto
    echo
    
    # Check microservices
    echo "Microservices:"
    if [ -f "logs/data_ingestion.pid" ] && kill -0 $(cat logs/data_ingestion.pid) 2>/dev/null; then
        echo "✅ Data Ingestion Service (PID: $(cat logs/data_ingestion.pid))"
    else
        echo "❌ Data Ingestion Service"
    fi
    
    if [ -f "logs/etl_processing.pid" ] && kill -0 $(cat logs/etl_processing.pid) 2>/dev/null; then
        echo "✅ ETL Processing Service (PID: $(cat logs/etl_processing.pid))"
    else
        echo "❌ ETL Processing Service"
    fi
    
    if [ -f "logs/data_storage.pid" ] && kill -0 $(cat logs/data_storage.pid) 2>/dev/null; then
        echo "✅ Data Storage Service (PID: $(cat logs/data_storage.pid))"
    else
        echo "❌ Data Storage Service"
    fi
    
    if [ -f "logs/api_gateway.pid" ] && kill -0 $(cat logs/api_gateway.pid) 2>/dev/null; then
        echo "✅ API Gateway (PID: $(cat logs/api_gateway.pid))"
    else
        echo "❌ API Gateway"
    fi
    
    if [ -f "logs/dashboard.pid" ] && kill -0 $(cat logs/dashboard.pid) 2>/dev/null; then
        echo "✅ Dashboard (PID: $(cat logs/dashboard.pid))"
    else
        echo "❌ Dashboard"
    fi
    
    if [ -f "logs/simulator.pid" ] && kill -0 $(cat logs/simulator.pid) 2>/dev/null; then
        echo "✅ Weather Simulator (PID: $(cat logs/simulator.pid))"
    else
        echo "⏹️ Weather Simulator"
    fi
    
    echo
    echo "Access Points:"
    echo "• API Gateway: http://localhost:8000"
    echo "• Dashboard: http://localhost:3000"
    echo "• API Health: http://localhost:8000/health"
    echo "• API Documentation: http://localhost:8000/docs"
}

# Function to stop all services
stop_services() {
    print_step "Stopping all services..."
    
    # Stop microservices
    for service in data_ingestion etl_processing data_storage api_gateway dashboard simulator; do
        if [ -f "logs/${service}.pid" ]; then
            pid=$(cat "logs/${service}.pid")
            if kill -0 $pid 2>/dev/null; then
                print_status "Stopping $service (PID: $pid)..."
                kill $pid
                rm -f "logs/${service}.pid"
            fi
        fi
    done
    
    # Stop Docker services
    print_status "Stopping infrastructure services..."
    docker-compose down
    
    print_status "All services stopped!"
}

# Function to show logs
show_logs() {
    local service=$1
    
    if [ -z "$service" ]; then
        echo "Available logs:"
        ls -1 logs/*.log 2>/dev/null | sed 's/logs\///g' | sed 's/\.log//g' || echo "No log files found"
        return
    fi
    
    if [ -f "logs/${service}.log" ]; then
        tail -f "logs/${service}.log"
    else
        print_error "Log file logs/${service}.log not found"
    fi
}

# Main function
main() {
    echo "🌦️ Weather ETL Microservices System Runner"
    echo "==========================================="
    echo
    
    case "${1:-start}" in
        "start")
            check_prerequisites
            setup_environment
            start_infrastructure
            start_microservices
            start_dashboard
            echo
            print_status "🎉 Weather ETL system started successfully!"
            echo
            show_status
            echo
            print_status "To start generating weather data, run:"
            print_status "  $0 simulator"
            ;;
        
        "simulator")
            start_simulator
            ;;
        
        "status")
            show_status
            ;;
        
        "stop")
            stop_services
            ;;
        
        "restart")
            stop_services
            sleep 3
            main start
            ;;
        
        "logs")
            show_logs "$2"
            ;;
        
        "help"|"--help"|"-h")
            echo "Usage: $0 [command]"
            echo
            echo "Commands:"
            echo "  start      Start the entire weather ETL system (default)"
            echo "  simulator  Start weather station data simulator"
            echo "  status     Show system status"
            echo "  stop       Stop all services"
            echo "  restart    Restart all services"
            echo "  logs [service]  Show logs for a service"
            echo "  help       Show this help message"
            echo
            echo "Examples:"
            echo "  $0              # Start the system"
            echo "  $0 status       # Check system status"
            echo "  $0 logs api_gateway  # Show API gateway logs"
            echo "  $0 stop         # Stop everything"
            ;;
        
        *)
            print_error "Unknown command: $1"
            print_status "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
