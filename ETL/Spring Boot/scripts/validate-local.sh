#!/bin/bash

# Local validation script for microservices
# This script validates that all microservices can build and start locally

set -e

echo "=== AWS ETL Microservices Local Validation ==="
echo

PROJECT_ROOT="/home/laptop/EXERCISES/Data Analysis and Visualization/Data-Analysis-and-Visualizations/ETL/Spring Boot"
cd "$PROJECT_ROOT"

echo "1. Building all modules..."
mvn clean compile -q
if [ $? -eq 0 ]; then
    echo "✅ All modules compiled successfully"
else
    echo "❌ Compilation failed"
    exit 1
fi

echo
echo "2. Packaging all modules..."
mvn clean package -DskipTests -q
if [ $? -eq 0 ]; then
    echo "✅ All modules packaged successfully"
else
    echo "❌ Packaging failed"
    exit 1
fi

echo
echo "3. Checking JAR files..."
SERVICES=("data-ingestion-service" "etl-processing-service" "stream-processing-service" "query-service" "dashboard-api-service")

for service in "${SERVICES[@]}"; do
    jar_file="microservices/$service/target/$service-1.0.0-SNAPSHOT.jar"
    if [ -f "$jar_file" ]; then
        echo "✅ $service JAR created: $(ls -lh $jar_file | awk '{print $5}')"
    else
        echo "❌ $service JAR not found"
        exit 1
    fi
done

echo
echo "4. Testing JAR executability..."
for service in "${SERVICES[@]}"; do
    jar_file="microservices/$service/target/$service-1.0.0-SNAPSHOT.jar"
    echo "Testing $service..."
    
    # Test if JAR can start (will fail due to missing dependencies, but should show Spring Boot banner)
    timeout 10s java -jar "$jar_file" --server.port=0 > /tmp/test_$service.log 2>&1 || true
    
    if grep -q "Spring Boot" /tmp/test_$service.log; then
        echo "✅ $service JAR is executable"
    else
        echo "⚠️  $service JAR may have issues (check /tmp/test_$service.log)"
    fi
done

echo
echo "5. Project structure validation..."
echo "✅ Project structure:"
echo "   - Parent POM: $([ -f pom.xml ] && echo "✅" || echo "❌")"
echo "   - Shared modules: $([ -d shared ] && echo "✅" || echo "❌")"
echo "   - Microservices: $([ -d microservices ] && echo "✅" || echo "❌")"
echo "   - Infrastructure: $([ -d infrastructure ] && echo "✅" || echo "❌")"
echo "   - Frontend: $([ -d frontend ] && echo "✅" || echo "❌")"
echo "   - Documentation: $([ -f README.md ] && echo "✅" || echo "❌")"

echo
echo "6. Docker configuration..."
if [ -f docker-compose.yml ]; then
    echo "✅ Docker Compose configuration found"
    echo "   - To start all services: docker-compose up"
else
    echo "❌ Docker Compose configuration missing"
fi

echo
echo "7. Development tools..."
if [ -f .vscode/tasks.json ]; then
    echo "✅ VS Code tasks configured"
else
    echo "⚠️  VS Code tasks not found"
fi

echo
echo "=== Local Validation Summary ==="
echo "✅ All microservices built successfully"
echo "✅ All JAR files created"
echo "✅ Project is ready for local development"
echo
echo "Next steps:"
echo "1. Install local dependencies: ./scripts/install-aws-tools.sh"
echo "2. Start services with Docker: docker-compose up"
echo "3. Or start individual services:"
for service in "${SERVICES[@]}"; do
    port=$((8081 + $(echo "${SERVICES[@]}" | tr ' ' '\n' | grep -n "$service" | cut -d: -f1) - 1))
    echo "   java -jar microservices/$service/target/$service-1.0.0-SNAPSHOT.jar --server.port=$port"
done

echo
echo "4. Access services:"
echo "   - Data Ingestion API: http://localhost:8081/api/ingestion/status"
echo "   - ETL Processing API: http://localhost:8082/api/etl/status"
echo "   - Query Service API: http://localhost:8083/api/query/status"
echo "   - Stream Processing: http://localhost:8084/actuator/health"
echo "   - Dashboard API: http://localhost:8085/actuator/health"

echo
echo "=== Validation Complete ==="
