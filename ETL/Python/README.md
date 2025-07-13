# Weather data ETL microservices

## Architecture

This solution implements a real-time ETL pipeline for weather station data.

- **gRPC Microservices** for inter-service communication
- **MQTT** for real-time data ingestion from weather stations
- **Python FastAPI** for REST API endpoints
- **AWS Services** for deployment and scaling
- **Next.js Dashboard** for data visualization

## Components

### Services
1. **Data ingestion** - MQTT client for weather station data
2. **ETL processing** - Data transformation and validation
3. **Data storage** - Database operations and caching
4. **API Gateway** - REST API endpoints for frontend
5. **Notifications** - Alerts and monitoring

### Frontend
- **Next.js dashboard** - Real-time weather data visualization
- **React components** - Interactive charts and maps
- **Server-Side Rendering (SSR)** - Optimized for AWS CloudFront

## AWS infrastructure

### AWS services
- **ECS/Fargate** - Container orchestration for microservices
- **S3** - Data lake for historical weather data
- **RDS/Aurora** - Structured data storage
- **ElastiCache** - Redis for real-time caching
- **CloudFront** - CDN for frontend delivery
- **API Gateway** - Managed API endpoints
- **IoT Core** - MQTT message broker
- **Lambda** - Serverless data processing
- **CloudWatch** - Monitoring and logging

### Required IAM roles
1. **ECS task execution role**
2. **S3 data access role**
3. **RDS access role**
4. **IoT core access role**
5. **CloudWatch logs role**

## Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Configure AWS credentials
3. Set up environment variables
4. Deploy infrastructure with Terraform/CDK
5. Start local development servers

## Data flow

```
Weather Stations → MQTT/IoT Core → Ingestion Service → ETL Service → Storage Service → API Gateway → Dashboard
```

## Real-time processing

- Data collected every hour from 5 weather stations
- 100km radius coverage area
- JSON format with temperature, humidity, pressure, wind data
- Real-time alerts for extreme weather conditions
