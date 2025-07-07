#!/bin/bash

# Build script for the entire IoT analytics platform

echo "Building IoT Analytics Platform with Spring Boot and Machine Learning"
echo "===================================================================="

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p mosquitto/data
mkdir -p mosquitto/log
mkdir -p localstack
mkdir -p ml-service/models
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/grafana/datasources

# Build Spring Boot application
echo "Building Spring Boot application..."
if command -v mvn &> /dev/null; then
    mvn clean package -DskipTests
else
    echo "Maven not found. Please install Maven to build the Spring Boot application."
fi

# Build Docker images
echo "Building Docker images..."
docker-compose build

echo "Build completed!"
echo ""
echo "To start the platform, run:"
echo "  docker-compose up -d"
echo ""
echo "Services will be available at:"
echo "  - Spring Boot API: http://localhost:8080"
echo "  - ML Service: http://localhost:8001"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - MQTT Broker: localhost:1883"
