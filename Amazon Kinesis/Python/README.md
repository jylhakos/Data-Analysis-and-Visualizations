# Example: IoT temperature forecasting with Amazon Kinesis and FastAPI

## Overview

A complete IoT temperature forecasting application that ingests sensor data through MQTT, processes it with Amazon Kinesis, and provides ML-powered temperature predictions via FastAPI.

The application is containerized with Docker and deployable on AWS ECS.

## Architecture

```
[IoT Sensors] → [MQTT Broker] → [Kinesis Ingester] → [Amazon Kinesis] → [FastAPI + ML Model] → [Predictions API]
```

### Components

- **MQTT simulator**: Simulates IoT temperature sensors
- **MQTT to Kinesis Ingester**: Processes MQTT messages and sends to Kinesis
- **Amazon Kinesis Data Streams**: Real-time data streaming
- **FastAPI application**: REST API with ML temperature forecasting
- **Machine learning model**: Scikit-learn RandomForest for temperature prediction
- **Docker containerization**: Easy deployment and scaling
- **AWS ECS deployment**: Production-ready container orchestration

## Quick start

### Prerequisites

- Docker and Docker Compose
- AWS CLI configured with appropriate permissions
- Python 3.11+
- Git

### Local development

1. **Setup environment**
   ```bash
   chmod +x setup-environment.sh
   ./setup-environment.sh
   ```

2. **Start local stack**
   ```bash
   docker-compose up -d
   ```

3. **Test the API**
   ```bash
   curl http://localhost:8000/
   curl http://localhost:8000/docs  # API documentation
   ```

4. **Simulate IoT sensors**
   ```bash
   chmod +x simulate-sensors.sh
   ./simulate-sensors.sh
   ```

### AWS deployment

1. **Deploy infrastructure and application**
   ```bash
   chmod +x deploy/deploy.sh
   ./deploy/deploy.sh
   ```

2. **Access your application**
   - The deployment script will output the Application Load Balancer URL
   - API documentation available at: `{ALB_URL}/docs`

## Project structure

```
├── fastapi_app.py              # Main FastAPI application
├── temperature_ml_model.py     # ML model for temperature forecasting
├── mqtt_kinesis_ingester.py    # MQTT to Kinesis data pipeline
├── kinesis_processor.py        # Kinesis stream processor
├── mqtt_simulator.py           # IoT sensor simulator
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Development container
├── Dockerfile.production       # Production container
├── docker-compose.yml          # Local development stack
├── mosquitto.conf             # MQTT broker configuration
├── setup-environment.sh       # Environment setup script
├── simulate-sensors.sh         # Sensor simulation script
└── deploy/
    ├── aws-infrastructure.yaml # CloudFormation template
    └── deploy.sh              # Deployment script
```

## API endpoints

### RESTful API

- `GET /` - Health check and API info
- `GET /docs` - Interactive API documentation
- `POST /sensor-data` - Submit sensor readings
- `POST /forecast` - Get temperature predictions
- `GET /historical-data` - Retrieve historical sensor data
- `GET /sensors` - List all sensors
- `POST /model/train` - Train ML model
- `GET /model/status` - Model training status

### Example usage

**Submit sensor data:**
```bash
curl -X POST "http://localhost:8000/sensor-data" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "sensor_001",
    "temperature": 25.5,
    "humidity": 60.0,
    "location": "Office_A"
  }'
```

**Get temperature forecast:**
```bash
curl -X POST "http://localhost:8000/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "sensor_001",
    "hours_ahead": 24
  }'
```

## AWS services

- **Amazon ECS (Fargate)**: Container orchestration
- **Amazon Kinesis Data Streams**: Real-time data streaming
- **Application Load Balancer**: Load balancing and SSL termination
- **Amazon ECR**: Container registry
- **CloudWatch**: Logging and monitoring
- **IAM**: Security and access control
- **VPC**: Network isolation

## ML model features

The temperature forecasting model uses:

- **Time-based features**: Hour, day of week, month, day of year
- **Cyclical encoding**: Sine/cosine transformations for time features
- **Lag features**: Previous 1, 2, 3, 6, 12, 24 hours
- **Rolling statistics**: Moving averages and standard deviations
- **Difference features**: Temperature changes over time
- **Weather patterns**: Humidity correlations

## Monitoring and troubleshooting

### CloudWatch Logs
```bash
aws logs tail /ecs/iot-temp-forecast --follow
```

### ECS service status
```bash
aws ecs describe-services --cluster iot-temp-forecast-cluster --services iot-temp-forecast-service
```

### Kinesis stream monitoring
```bash
aws kinesis describe-stream --stream-name iot-temp-forecast-temperature-sensor-stream
```

## Scaling and performance

- **Horizontal scaling**: Adjust ECS service desired count
- **Kinesis scaling**: Add more shards for higher throughput
- **Load balancer**: Handles traffic distribution automatically
- **Auto-scaling**: Configure based on CPU/memory metrics

## Security

- ✅ Non-root container user
- ✅ IAM roles with least privilege
- ✅ VPC network isolation
- ✅ Security groups for access control
- ✅ HTTPS/SSL termination at load balancer
- ✅ Container image scanning

## Cost optimization

- Use Fargate Spot for non-critical workloads
- Configure appropriate Kinesis retention periods
- Set up CloudWatch log retention policies
- Monitor and optimize container resource allocation

## References

- [Amazon Kinesis Data Streams](https://docs.aws.amazon.com/streams/latest/dev/introduction.html)
- [AWS ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
