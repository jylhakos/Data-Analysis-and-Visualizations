# Weather ETL microservices architecture

## 🏗️ System Architecture

This solution implements a production-ready, cloud-native ETL pipeline for processing real-time weather data from multiple weather stations using modern microservices architecture.

### 🎯 Features
- **Real-time data processing**: Handles weather data from 5 stations every hour
- **Microservices architecture**: gRPC-based inter-service communication
- **Scalable**: Ready for AWS deployment with auto-scaling
- **Data lake integration**: Long-term storage in S3 with lifecycle policies
- **Real-time dashboard**: Next.js SPA with live data visualization
- **Production**: Monitoring, logging, and error handling

## 📋 Component architecture

### Core microservices

| Service | Purpose | Technology | Port | Dependencies |
|---------|---------|------------|------|--------------|
| **Data Ingestion** | MQTT client, data reception | Python, gRPC, MQTT | 50051 | MQTT Broker |
| **ETL Processing** | Data validation, transformation | Python, gRPC, Pandas | 50052 | None |
| **Data Storage** | Database operations, caching | Python, gRPC, SQLAlchemy | 50053 | PostgreSQL, Redis, S3 |
| **API Gateway** | REST API endpoints | Python, FastAPI | 8000 | All gRPC services |

### Infrastructure components

| Component | Purpose | Technology | Configuration |
|-----------|---------|------------|---------------|
| **PostgreSQL** | Structured data storage | PostgreSQL 15 | Primary database with indexes |
| **Redis** | Real-time caching | Redis 7 | Session cache, latest readings |
| **MQTT Broker** | Message broker | Eclipse Mosquitto | Weather station data ingestion |
| **S3 Data Lake** | Long-term storage | AWS S3 | Partitioned by date/station |

### Frontend dashboard

| Component | Purpose | Technology | Features |
|-----------|---------|------------|----------|
| **Dashboard** | Data visualization | Next.js, React, TypeScript | Real-time charts, maps, alerts |
| **API client** | Data fetching | Axios, SWR | Server-side rendering, caching |
| **UI components** | Interactive elements | Tailwind CSS, Chart.js | Responsive design |

## 🚀 Quick start

### Prerequisites
```bash
# Install system dependencies
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
sudo apt install -y docker.io docker-compose nodejs npm

# Clone and setup project
cd /home/laptop/EXERCISES/Data\ Analysis\ and\ Visualization/Data-Analysis-and-Visualizations/ETL/Python
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start the system
```bash
# Start entire system with one command
./run_weather_etl.sh start

# Check system status
./run_weather_etl.sh status

# Start weather data simulation
./run_weather_etl.sh simulator

# View logs
./run_weather_etl.sh logs api_gateway

# Stop system
./run_weather_etl.sh stop
```

### Access endpoints
- **Dashboard**: http://localhost:3000
- **API Gateway**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## ☁️ AWS deployment

### Required AWS services

#### Compute & container Services
- **Amazon ECS with Fargate**: Serverless container orchestration
- **Application Load Balancer**: Traffic distribution and health checks
- **Auto Scaling Groups**: Automatic scaling based on metrics

#### Storage services
- **Amazon RDS PostgreSQL**: Managed relational database
- **Amazon ElastiCache**: Managed Redis for caching
- **Amazon S3**: Data lake for historical weather data

#### Networking & security
- **Amazon VPC**: Isolated network environment
- **Security Groups**: Fine-grained network access control
- **AWS IoT Core**: Managed MQTT broker for device connectivity

#### Monitoring
- **Amazon CloudWatch**: Metrics, logs, and alarms
- **AWS X-Ray**: Distributed tracing for microservices
- **AWS CloudTrail**: API call auditing

#### Content delivery
- **Amazon CloudFront**: Global CDN for dashboard
- **Route 53**: DNS management and health checks

### Essential IAM roles

#### 1. ECS Task Execution Role
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**Policies**: 
- `AmazonECSTaskExecutionRolePolicy`
- Custom weather ETL policy (S3, RDS, IoT access)

#### 2. Weather ETL service role
**Permissions**:
- S3: Read/Write to data lake bucket
- RDS: Connect to PostgreSQL database  
- ElastiCache: Access Redis cache
- IoT Core: MQTT publish/subscribe
- CloudWatch: Write logs and custom metrics

#### 3. CloudFront OAIrole
**Purpose**: Allow CloudFront to access S3 bucket for dashboard assets

### AWS Deployment commands

```bash
# Deploy infrastructure
chmod +x aws/deploy-infrastructure.sh
./aws/deploy-infrastructure.sh

# Build and push containers to ECR
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build and push all microservices
docker build -f Dockerfile.data-ingestion -t $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/weather-etl/data-ingestion:latest .
docker push $ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/weather-etl/data-ingestion:latest
```

## 📊 Data flow

```
Weather Stations → MQTT/IoT Core → Data Ingestion → ETL Processing → Data Storage
                                                                           ↓
Dashboard ← API Gateway ← ←←←←←←←←←←←←←←←←←←←←←←←←←←←←← ← PostgreSQL/Redis/S3
```

### Data processing pipeline

1. **Data Collection**: Weather stations send JSON data via MQTT every hour
2. **Ingestion**: Data Ingestion service receives and forwards to ETL
3. **Processing**: ETL service validates, transforms, and enriches data
4. **Storage**: Data Storage service saves to PostgreSQL, Redis, and S3
5. **API**: API Gateway provides REST endpoints for dashboard
6. **Visualization**: Next.js dashboard displays real-time data and charts

### Weather data format
```json
{
  "station_id": "WS001",
  "latitude": 40.7589,
  "longitude": -73.9851,
  "temperature": 22.5,
  "humidity": 65.2,
  "pressure": 1013.8,
  "wind_speed": 12.3,
  "wind_direction": 225.0,
  "timestamp": "2025-01-13T10:00:00Z",
  "metadata": {
    "heat_index": 24.1,
    "wind_chill": 22.5,
    "dew_point": 15.8
  }
}
```

## 🔧 Local development

### Tools
```bash
# Python development
pip install grpcio-tools black flake8 pytest mypy

# Container development
docker --version
docker-compose --version

# AWS development
aws --version
npm install -g aws-cdk
```

### AWS CLI tools for local machine
```bash
# Essential AWS tools
aws configure                    # Configure credentials
aws-cli/2.x                     # Latest AWS CLI
aws-cdk                         # Infrastructure as Code
terraform                       # Alternative to CDK
kubectl                         # Kubernetes management
eksctl                          # EKS cluster management
helm                            # Kubernetes package manager
```

## 📈 Performance and scaling

### Performance metrics
- **Latency**: <100ms for API responses
- **Throughput**: 1000+ requests/second
- **Data Processing**: Real-time with <5 second lag
- **Availability**: 99.9% uptime target

### Scaling configuration
```yaml
# ECS Auto Scaling
TargetTrackingScaling:
  TargetValue: 70.0
  ScaleInCooldown: 300s
  ScaleOutCooldown: 60s
  MetricType: ECSServiceAverageCPUUtilization

# RDS Auto Scaling
Storage:
  MaxAllocatedStorage: 1000
  MonitoringInterval: 60
```

## 💰 Cost estimation (AWS monthly)

| Service | Instance Type | Estimated Cost |
|---------|---------------|----------------|
| ECS Fargate (4 services) | 0.25 vCPU, 0.5 GB | $35-50 |
| RDS PostgreSQL | db.t3.micro | $15 |
| ElastiCache Redis | cache.t3.micro | $12 |
| Application Load Balancer | - | $16 |
| S3 Storage (100GB) | Standard | $2-5 |
| CloudFront | 1TB transfer | $5-10 |
| IoT Core | 1M messages | $1-2 |
| **Total Estimated** | | **$85-120/month** |

### Cost optimization
1. **Reserved Instances**: 30-60% savings for predictable workloads
2. **Spot Instances**: Up to 90% savings for fault-tolerant workloads
3. **S3 Lifecycle Policies**: Automatic archival to Glacier/Deep Archive
4. **Auto Scaling**: Scale down during low-traffic periods
5. **Right-sizing**: Monitor and adjust instance sizes based on usage

## 🔍 Monitoring & observability

### CloudWatch dashboards
- **System Health**: Service status, error rates, response times
- **Business Metrics**: Data processing volume, station connectivity
- **Infrastructure**: CPU, memory, disk utilization
- **Cost Tracking**: Service-wise cost breakdown

### Alerting
```yaml
Critical Alerts:
  - Service downtime > 5 minutes
  - Error rate > 5%
  - Database connection failures
  
Warning Alerts:
  - High CPU utilization > 80%
  - Missing weather data > 2 hours
  - API response time > 2 seconds
```

## 🔐 Security

### Network security
- **VPC Isolation**: Private subnets for databases
- **Security Groups**: Least-privilege access rules
- **WAF Protection**: Web Application Firewall for APIs

### Data security
- **Encryption at Rest**: RDS, S3, EBS encryption
- **Encryption in Transit**: TLS for all communications
- **IAM Roles**: No hardcoded credentials
- **Secrets Manager**: Secure credential storage

### Application security
- **API Rate Limiting**: Prevent abuse and DDoS
- **Input Validation**: Comprehensive data validation
- **CORS Configuration**: Restrict cross-origin requests
- **Container Scanning**: Automated vulnerability scanning

## 🚦 Steps

### 1. Local development
```bash
# Complete local setup
./run_weather_etl.sh start
./run_weather_etl.sh simulator
```

### 2. AWS account setup
```bash
# Configure AWS credentials
aws configure

# Deploy infrastructure
./aws/deploy-infrastructure.sh
```

### 3. Production deployment
```bash
# Build and deploy containers
docker build -t weather-etl/api-gateway .
docker tag weather-etl/api-gateway $ECR_REGISTRY/weather-etl/api-gateway:latest
docker push $ECR_REGISTRY/weather-etl/api-gateway:latest
```

### 4. Dashboard deployment
```bash
cd dashboard
npm run build
aws s3 sync out/ s3://weather-dashboard-bucket/
```

This solution provides a production-ready, scalable ETL system for real-time weather data processing with modern cloud-native architecture, complete monitoring, and cost-effective AWS deployment.
