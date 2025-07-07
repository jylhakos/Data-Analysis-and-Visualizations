# IoT Temperature Analytics Platform with Spring Boot 3.x and Machine Learning

## Architecture Overview

This project demonstrates a comprehensive IoT data processing and machine learning platform using:

- **Spring Boot 3.x** with Java 17 for microservices
- **Amazon Kinesis** for real-time data streaming with KCL (Kinesis Client Library)
- **Amazon DynamoDB** for data storage
- **Amazon SQS** for message queuing
- **MQTT** (Mosquitto) for IoT device communication
- **Machine Learning** with Python/FastAPI and scikit-learn for temperature forecasting
- **Docker** for containerization
- **AWS ECS** ready deployment

## Components

### 1. Spring Boot Application (`kinesis-iot-processor`)
- **Main Application**: `KinesisIotProcessorApplication.java`
- **Models**: Temperature data and forecast models
- **Services**:
  - `KinesisProducerService`: Publishes data to Kinesis streams
  - `KinesisConsumerService`: Consumes data from Kinesis using KCL
  - `TemperatureDataProcessor`: Processes and validates sensor data
  - `DynamoDbService`: Handles data persistence
  - `SqsService`: Manages message queuing
  - `MLServiceClient`: Communicates with ML service
- **Controllers**: REST APIs for data access and forecasting
- **MQTT Integration**: Receives data from IoT sensors

### 2. Machine Learning Service (`ml-service/`)
- **FastAPI** application for temperature forecasting
- **scikit-learn** Random Forest model for time series prediction
- **Features**:
  - Real-time temperature data analysis
  - Time series forecasting with confidence intervals
  - Model training with historical data
  - RESTful API for predictions

### 3. IoT Sensor Simulator (`sensor-simulator/`)
- Simulates multiple temperature sensors
- Generates realistic temperature data with:
  - Daily and seasonal patterns
  - Random noise and variations
  - Battery level simulation
  - MQTT publishing

### 4. Infrastructure Components
- **MQTT Broker**: Eclipse Mosquitto for IoT communication
- **LocalStack**: Local AWS services simulation
- **Monitoring**: Prometheus and Grafana
- **Docker Compose**: Complete multi-service deployment

## Quick Start

### Prerequisites
- Java 17+
- Maven 3.6+
- Docker & Docker Compose
- Python 3.9+ (for local ML service development)

### 1. Build the Project
```bash
chmod +x build.sh
./build.sh
```

### 2. Start the Platform
```bash
docker-compose up -d
```

### 3. Verify Services
- **Spring Boot API**: http://localhost:8080/api/temperature/health
- **ML Service**: http://localhost:8001/health
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## API Endpoints

### Spring Boot Application (Port 8080)
- `GET /api/temperature/sensors` - List all sensors
- `GET /api/temperature/latest/{sensorId}` - Latest temperature reading
- `GET /api/temperature/history/{sensorId}` - Historical data
- `GET /api/temperature/stats/{sensorId}` - Sensor statistics
- `POST /api/temperature/data` - Submit temperature data
- `GET /api/temperature/health` - Health check

### ML Service (Port 8001)
- `GET /health` - Health check
- `POST /api/temperature/analyze` - Analyze temperature data
- `POST /api/forecast/temperature` - Generate forecast
- `POST /api/model/train` - Train ML model
- `GET /api/sensors` - List sensors with data
- `GET /api/sensors/{sensorId}/data` - Get sensor data

## Configuration

### Spring Boot Configuration (`application.yml`)
```yaml
spring:
  application:
    name: kinesis-iot-processor
  cloud:
    stream:
      bindings:
        temperatureData-in-0:
          destination: temperature-stream
          group: temperature-processor-group

aws:
  region: us-east-1
  kinesis:
    stream:
      name: temperature-stream

mqtt:
  broker:
    url: tcp://localhost:1883
  topics:
    temperature: sensor/temperature/+
```

### Environment Variables
- `AWS_REGION`: AWS region (default: us-east-1)
- `MQTT_BROKER_URL`: MQTT broker URL
- `ML_SERVICE_URL`: ML service URL
- `SENSOR_COUNT`: Number of simulated sensors
- `INTERVAL_SECONDS`: Data publishing interval

## Data Flow

1. **IoT Sensors** → MQTT → **Spring Boot MQTT Listener**
2. **Spring Boot** → Amazon Kinesis → **KCL Consumer**
3. **KCL Consumer** → DynamoDB (storage) + SQS (processing queue)
4. **Temperature Data** → **ML Service** (analysis & forecasting)
5. **Web Clients** → **REST API** → **Forecast Results**

## Machine Learning Features

### Temperature Forecasting Model
- **Algorithm**: Random Forest Regressor
- **Features**:
  - Time-based features (hour, day, month)
  - Lag features (previous temperatures)
  - Rolling averages (3, 6, 12 hour windows)
  - Humidity data (when available)

### Model Training
- Automatic feature engineering
- Cross-validation with train/test split
- Model persistence with joblib
- Performance metrics (MAE, MSE)

### Forecasting
- Multi-day ahead predictions
- Hourly granularity
- Confidence intervals
- Uncertainty quantification

## AWS Integration

### Amazon Kinesis
- Real-time data streaming
- Automatic scaling with shard management
- KCL for distributed processing

### Amazon DynamoDB
- NoSQL storage for time-series data
- Automatic scaling
- Global secondary indexes

### Amazon SQS
- Message queuing for reliable processing
- Dead letter queues for error handling
- Batch processing support

## Production Deployment

### AWS ECS Deployment
```yaml
# Task definition for Spring Boot service
{
  "family": "kinesis-iot-processor",
  "networkMode": "awsvpc",
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [...]
}
```

### Environment-Specific Configurations
- **Local**: H2 database, LocalStack
- **Development**: PostgreSQL, AWS dev environment
- **Production**: RDS, full AWS services

## Monitoring and Observability

### Metrics (Prometheus)
- Application metrics
- JVM metrics
- Custom business metrics

### Visualization (Grafana)
- Real-time temperature dashboards
- System performance metrics
- Alert configurations

### Logging
- Structured logging with Logback
- Application logs
- AWS CloudWatch integration

## Testing

### Unit Tests
```bash
mvn test
```

### Integration Tests
```bash
mvn verify
```

### Load Testing
- JMeter scripts for API testing
- MQTT load testing tools

## Development

### Local Development Setup
1. Start LocalStack for AWS services
2. Start Mosquitto MQTT broker
3. Run Spring Boot application
4. Start ML service
5. Run sensor simulator

### Adding New Features
1. **New Sensor Types**: Extend TemperatureData model
2. **ML Models**: Add new algorithms in ML service
3. **APIs**: Create new REST controllers
4. **Processors**: Implement new stream processors

## Troubleshooting

### Common Issues
1. **MQTT Connection**: Check broker URL and port
2. **AWS Credentials**: Verify LocalStack or AWS setup
3. **ML Service**: Check Python dependencies
4. **Docker**: Ensure sufficient resources

### Logs Location
- Spring Boot: `logs/kinesis-iot-processor.log`
- Docker logs: `docker-compose logs [service-name]`

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Spring Boot Documentation](https://spring.io/projects/spring-boot)
- [Spring Cloud Stream](https://spring.io/projects/spring-cloud-stream)
- [Amazon Kinesis](https://aws.amazon.com/kinesis/)
- [Eclipse Mosquitto](https://mosquitto.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [scikit-learn](https://scikit-learn.org/)

