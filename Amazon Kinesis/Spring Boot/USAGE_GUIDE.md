# IoT temperature analytics microservices provided by machine learning

## Overview

This comprehensive Spring Boot 3.x application demonstrates a complete IoT data processing pipeline with machine learning capabilities for temperature forecasting. The system includes:

## Features

### 🌡️ **Temperature data processing**
- Real-time IoT sensor data ingestion via MQTT
- Amazon Kinesis stream processing with KCL (Kinesis Client Library)
- DynamoDB for time-series data storage
- SQS for reliable message queuing

### 🤖 **Machine learning integration**
- Python FastAPI service with scikit-learn
- Random Forest model for temperature forecasting
- Time series prediction with confidence intervals
- Automatic model training and retraining

### 🏗️ **Microservices**
- Spring Boot 3.x with reactive programming
- Docker containerization for all services
- AWS-ready deployment with ECS support
- Monitoring with Prometheus and Grafana

## Quick start

### 1. Build and start the microservices
```bash
# Build all components
./build.sh

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### 2. Test the platform
```bash
# Install test dependencies
pip install requests

# Run comprehensive tests
python test_platform.py
```

### 3. Access the services

| Service | URL | Description |
|---------|-----|-------------|
| Spring Boot API | http://localhost:8080 | Main application APIs |
| ML Service | http://localhost:8001 | Machine learning endpoints |
| Grafana | http://localhost:3000 | Monitoring dashboard (admin/admin) |
| Prometheus | http://localhost:9090 | Metrics collection |

## API usage

### Temperature data submission
```bash
# Submit temperature reading
curl -X POST http://localhost:8080/api/temperature/data \
  -H "Content-Type: application/json" \
  -d '{
    "sensorId": "SENSOR_001",
    "temperature": 23.5,
    "humidity": 65.0,
    "location": "Living Room",
    "timestamp": "2024-01-15T10:30:00",
    "deviceType": "thermometer",
    "batteryLevel": 85,
    "signalStrength": -45
  }'
```

### Get sensor statistics
```bash
# Get sensor stats
curl http://localhost:8080/api/temperature/stats/SENSOR_001
```

### Request temperature forecast
```bash
# Request 7-day forecast
curl -X POST http://localhost:8001/api/forecast/temperature \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "SENSOR_001",
    "days_ahead": 7
  }'
```

## Data flows

```
IoT Sensors → MQTT Broker → Spring Boot MQTT Listener
     ↓
Spring Boot → Amazon Kinesis → KCL Consumer
     ↓
DynamoDB (Storage) + SQS (Processing Queue)
     ↓
ML Service (Analysis & Forecasting)
     ↓
REST API → Web Clients
```

## Production

### AWS ECS deployment
```yaml
# docker-compose.production.yml
version: '3.8'
services:
  kinesis-iot-processor:
    image: your-registry/kinesis-iot-processor:latest
    environment:
      - SPRING_PROFILES_ACTIVE=production
      - AWS_REGION=us-east-1
      - RDS_ENDPOINT=your-rds-endpoint
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
```

### Environment variables for production
```bash
# AWS Configuration
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your-access-key
export AWS_SECRET_ACCESS_KEY=your-secret-key

# Database
export DB_ENDPOINT=your-rds-endpoint
export DB_USERNAME=your-db-user
export DB_PASSWORD=your-db-password

# MQTT Broker
export MQTT_BROKER_URL=ssl://your-mqtt-broker:8883
export MQTT_USERNAME=your-mqtt-user
export MQTT_PASSWORD=your-mqtt-password

# ML Service
export ML_SERVICE_URL=https://your-ml-service-url
```

## Configuration for different environments

### Local development (application-local.yml)
```yaml
spring:
  datasource:
    url: jdbc:h2:mem:testdb
    
aws:
  endpoint-override: http://localstack:4566
  
mqtt:
  broker:
    url: tcp://localhost:1883
```

### Production (application-production.yml)
```yaml
spring:
  datasource:
    url: jdbc:postgresql://${DB_ENDPOINT}:5432/iotanalytics
    
aws:
  region: ${AWS_REGION}
  
mqtt:
  broker:
    url: ssl://${MQTT_BROKER_URL}:8883
```

## Monitoring and alerting

### Grafana Dashboards
- Temperature trends by sensor
- System performance metrics
- ML model accuracy tracking
- Alert rules for anomalies

### Prometheus metrics
```yaml
# Custom metrics exposed
- temperature_readings_total
- ml_predictions_total
- kinesis_records_processed
- mqtt_messages_received
```

## Scaling

### Horizontal scaling
- Multiple Spring Boot instances behind load balancer
- Kinesis shard scaling based on throughput
- ML service horizontal scaling with shared model storage

### Vertical scaling
- Increase JVM heap size for Spring Boot
- Scale DynamoDB read/write capacity
- Optimize ML model complexity vs. accuracy

## Security

### API security
```java
// JWT authentication example
@Security({@SecurityRequirement(name = "bearer-jwt")})
@GetMapping("/api/temperature/secure-endpoint")
public ResponseEntity<?> secureEndpoint(Authentication auth) {
    // Secured endpoint implementation
}
```

### MQTT security
```yaml
# TLS/SSL configuration
mqtt:
  broker:
    url: ssl://secure-broker:8883
    ssl:
      truststore: /path/to/truststore.jks
      keystore: /path/to/keystore.jks
```

## Troubleshooting

### Issues and solutions

1. **MQTT connections**
   ```bash
   # Check broker connectivity
   docker-compose logs mosquitto
   
   # Test MQTT connection
   mosquitto_pub -h localhost -p 1883 -t test/topic -m "test message"
   ```

2. **Kinesis streams**
   ```bash
   # Check LocalStack logs
   docker-compose logs localstack
   
   # Verify stream creation
   aws --endpoint-url=http://localhost:4566 kinesis list-streams
   ```

3. **ML services**
   ```bash
   # Check ML service logs
   docker-compose logs ml-service
   
   # Test ML service health
   curl http://localhost:8001/health
   ```

### Performance optimization

1. **JVM tuning**
   ```bash
   # Add to Spring Boot startup
   -Xmx2g -Xms1g -XX:+UseG1GC -XX:MaxGCPauseMillis=200
   ```

2. **Database optimization**
   ```sql
   -- Index for time-series queries
   CREATE INDEX idx_sensor_timestamp ON temperature_data(sensor_id, timestamp DESC);
   ```

3. **Kinesis optimization**
   ```yaml
   # Increase batch size for better throughput
   spring.cloud.stream.kinesis.binder:
     kinesis-producer-properties:
       record-max-buffered-time: 1000
       max-connections: 24
   ```

## Development

### Adding new sensor types
1. Extend `TemperatureData` model
2. Update MQTT listener for new topic patterns
3. Add new ML features for sensor-specific data
4. Update REST APIs for new data types

### Custom ML models
```python
# Add new model in ml-service/models/
class CustomTemperatureModel:
    def __init__(self):
        self.model = YourCustomModel()
    
    def train(self, data):
        # Custom training logic
        pass
    
    def predict(self, features):
        # Custom prediction logic
        pass
```

### Integration testing
```java
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@TestPropertySource(properties = {
    "spring.profiles.active=test",
    "mqtt.broker.url=tcp://localhost:1883"
})
class IntegrationTest {
    // Test implementation
}
```

## Documentation

### Additional resources
- [Spring Boot 3.x Documentation](https://spring.io/projects/spring-boot)
- [Spring Cloud Stream Kinesis](https://docs.awspring.io/spring-cloud-aws/docs/current/reference/html/index.html)
- [AWS Kinesis Client Library](https://docs.aws.amazon.com/kinesis/latest/dev/kinesis-record-processor-implementation-app-java.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Help
1. Check the logs first: `docker-compose logs [service-name]`
2. Verify configuration files
3. Test individual components
4. Review the troubleshooting section
5. Check AWS service limits and quotas

This IoT analytics platform demonstrates real-time data processing, machine learning integration, and microservices architecture using modern Spring Boot 3.x features.
