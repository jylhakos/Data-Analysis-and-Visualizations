#!/bin/bash
set -e

# Environment Setup Script for IoT Temperature Forecasting Application

PROJECT_NAME="iot-temp-forecast"
AWS_REGION="us-east-1"

echo "🔧 Setting up local environment for AWS deployment..."

# Check if required tools are installed
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is not installed. Please install it first."
        return 1
    else
        echo "✅ $1 is installed"
        return 0
    fi
}

echo "📋 Checking required tools..."
check_tool docker || exit 1
check_tool aws || exit 1
check_tool python3 || exit 1

# Check Docker daemon
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running. Please start Docker."
    exit 1
else
    echo "✅ Docker daemon is running"
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure' first."
    exit 1
else
    echo "✅ AWS credentials are configured"
    aws sts get-caller-identity
fi

# Install Python dependencies locally (for development)
echo "📦 Installing Python dependencies..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt

# Create .env file for local development
echo "📝 Creating .env file for local development..."
cat > .env << EOF
# AWS Configuration
AWS_REGION=$AWS_REGION
AWS_DEFAULT_REGION=$AWS_REGION
KINESIS_STREAM_NAME=${PROJECT_NAME}-temperature-sensor-stream

# Application Configuration
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883

# Development settings
DEBUG=True
LOG_LEVEL=INFO
EOF

# Make deployment script executable
chmod +x deploy/deploy.sh

echo ""
echo "🎉 Environment setup completed!"
echo ""
echo "📋 Next steps:"
echo "1. Review and update .env file if needed"
echo "2. Test locally: docker-compose up"
echo "3. Deploy to AWS: ./deploy/deploy.sh"
echo ""
echo "🔧 Local Development Commands:"
echo "# Start local environment:"
echo "docker-compose up"
echo ""
echo "# Run FastAPI locally:"
echo "source venv/bin/activate"
echo "uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "# Run MQTT simulator:"
echo "python mqtt_simulator.py"
echo ""
echo "# Run MQTT to Kinesis ingester:"
echo "python mqtt_kinesis_ingester.py"
