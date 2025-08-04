# ETL for Weather microservices setup

> **Deployment options:**
> - **This document (SETUP.md)**: Traditional microservices deployment with ECS/Fargate, RDS, and local development (Linux)
> - **AWS Glue deployment (README.md)**: Serverless ETL using AWS Glue, S3 Data Lake, and Kinesis streams
>
> Choose the approach that best fits your requirements:
> - **Traditional (this document)**: Full control, container-based, higher operational overhead
> - **AWS Glue (README.md)**: Serverless, auto-scaling, lower cost and maintenance

This document provides step-by-step instructions for setting up the ETL for Weather microservices system using **traditional AWS services** on your local machine and deploying to AWS with containerized microservices.

## Comparison

| Aspect | Traditional Setup (SETUP.md) | AWS Glue Setup (README.md) |
|--------|--------------------------------|----------------------------|
| **ETL processing** | Custom Python microservices | Serverless AWS Glue jobs |
| **Data Storage** | RDS PostgreSQL + S3 | S3 Data Lake (primary) |
| **Real-time processing** | Custom streaming services | Kinesis + Glue Streaming |
| **Infrastructure** | ECS/Fargate containers | Serverless (Glue, Lambda) |
| **Monthly cost** | ~$75-115 | ~$20-50 |
| **Operational overhead** | High (manage containers/DBs) | Low (managed services) |
| **Scalability** | Manual/auto-scaling groups | Automatic serverless scaling |
| **Development complexity** | Medium | Low-Medium |
| **Local development** | Full local stack | Hybrid (local + cloud) |
| **Best For** | Full control, custom logic | Cost efficiency, simplicity |

## Prerequisites

### System requirements
- Linux/Ubuntu (tested on Ubuntu 20.04+)
- Python 3.11+
- Node.js 18+
- Docker and Docker Compose
- Git

### Required accounts
- AWS Account with appropriate permissions
- GitHub account (for code repository)

## Local development

### 1. System dependencies

```bash
# Update package list
sudo apt update

# Install Python and development tools
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
sudo apt install -y build-essential libpq-dev curl wget git

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Log out and back in to activate Docker group membership
newgrp docker
```

### 2. Install AWS CLI and tools

```bash
# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installation
aws --version

# Install AWS CDK (optional but recommended)
npm install -g aws-cdk

# Install Terraform (alternative to CDK)
wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmit -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform
```

### 3. Configure Amazon AWS credentials

```bash
# Configure AWS CLI with your credentials
aws configure

# You'll be prompted for:
# AWS Access Key ID: [Your Access Key]
# AWS Secret Access Key: [Your Secret Key]  
# Default region name: us-east-1
# Default output format: json

# Verify configuration
aws sts get-caller-identity
```

### 4. Clone and setup project

```bash
# Navigate to your project directory
cd /home/laptop/EXERCISES/Data\ Analysis\ and\ Visualization/Data-Analysis-and-Visualizations/ETL/Python

# Create Python virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Edit environment variables
nano .env
```

### 5. Generate gRPC code

```bash
# Install grpc tools if not already installed
pip install grpcio-tools

# Generate Python gRPC code from protobuf definitions
python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. proto/weather.proto

# Create __init__.py files for proper imports
touch proto/__init__.py
```

### 6. Configure environment variables

Edit the `.env` file with your specific settings:

```bash
# Database Configuration
DATABASE_URL=postgresql://weather_user:weather_pass@localhost:5432/weather_db
REDIS_URL=redis://localhost:6379/0

# MQTT Configuration
MQTT_BROKER_HOST=localhost
MQTT_BROKER_PORT=1883
MQTT_USERNAME=weather_user
MQTT_PASSWORD=weather_pass

# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
S3_BUCKET_NAME=weather-data-lake-unique-suffix

# gRPC and API Configuration
GRPC_HOST=localhost
API_HOST=0.0.0.0
API_PORT=8000

# Weather Station Configuration
STATION_COUNT=5
COLLECTION_INTERVAL_HOURS=1
COVERAGE_RADIUS_KM=100
```

### 7. Start local development environment

```bash
# Start infrastructure services (PostgreSQL, Redis, MQTT)
docker-compose up -d postgres redis mosquitto

# Wait for services to be ready
sleep 30

# Create database tables (run this once)
python -c "
from services.data_storage import DatabaseManager
db = DatabaseManager()
print('Database tables created successfully')
"

# Start microservices in separate terminals

# Terminal 1: Data Ingestion Service
source venv/bin/activate
python -m services.data_ingestion

# Terminal 2: ETL Processing Service  
source venv/bin/activate
python -m services.etl_processing

# Terminal 3: Data Storage Service
source venv/bin/activate  
python -m services.data_storage

# Terminal 4: API Gateway
source venv/bin/activate
python -m services.api_gateway
```

### 8. Setup dashboard (Next.js)

```bash
# Navigate to dashboard directory
cd dashboard

# Install Node.js dependencies
npm install

# Start development server
npm run dev

# Dashboard will be available at http://localhost:3000
```

## Amazon AWS deployment

> **💡 Alternative: AWS Glue Serverless Deployment**
>
> For a serverless, cost-effective alternative, see **README.md** which covers:
> - AWS Glue ETL jobs (batch and streaming)
> - S3 Data Lake architecture
> - Kinesis Data Streams integration
> - ~60% cost reduction compared to this traditional setup
> - Automatic scaling and managed infrastructure
>
> Continue below for traditional containerized deployment on AWS.

This document provides steps needed to develop, deploy, and operate the ETL for Weather  microservices.

### 1. Essential IAM roles

The following IAM roles are required for AWS deployment:

#### ECS Task execution role
- **Purpose**: Allows ECS tasks to pull container images and write logs
- **Policies**:
  - `AmazonECSTaskExecutionRolePolicy`
  - Custom weather ETL policy (see `aws/weather-etl-policy.json`)

#### Weather ETL service role
- **Purpose**: Allows microservices to access AWS resources
- **Permissions**:
  - S3: Read/Write to data lake bucket
  - RDS: Connect to PostgreSQL database
  - ElastiCache: Access Redis cache
  - IoT Core: Publish/Subscribe to MQTT topics
  - CloudWatch: Write logs and metrics

#### CloudFront distribution role
- **Purpose**: Allows CloudFront to access S3 bucket for dashboard assets
- **Policies**: S3 read access to dashboard bucket

### 2. Required Amazon AWS services

| Service | Purpose | Configuration |
|---------|---------|---------------|
| **ECS/Fargate** | Container orchestration | 4 services (ingestion, ETL, storage, API) |
| **RDS PostgreSQL** | Structured data storage | db.t3.micro, Multi-AZ for production |
| **ElastiCache Redis** | Real-time caching | cache.t3.micro cluster |
| **S3** | Data lake and static hosting | 2 buckets (data lake, dashboard) |
| **CloudFront** | CDN for dashboard | Global distribution |
| **Application Load Balancer** | API gateway load balancing | Target groups for each service |
| **IoT Core** | MQTT message broker | Device certificates and policies |
| **CloudWatch** | Monitoring and logging | Log groups and dashboards |
| **Route 53** | DNS management | Custom domain for dashboard |

### 3. Deploy infrastructure

```bash
# Make deployment script executable
chmod +x aws/deploy-infrastructure.sh

# Run infrastructure deployment
./aws/deploy-infrastructure.sh

# This script will create:
# - VPC with public/private subnets
# - RDS PostgreSQL instance
# - ElastiCache Redis cluster  
# - ECS cluster
# - S3 buckets
# - IAM roles and policies
# - IoT Core configuration
```

### 4. Build and deploy containers

```bash
# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

# Create ECR repositories
aws ecr create-repository --repository-name weather-etl/data-ingestion --region $REGION
aws ecr create-repository --repository-name weather-etl/etl-processing --region $REGION  
aws ecr create-repository --repository-name weather-etl/data-storage --region $REGION
aws ecr create-repository --repository-name weather-etl/api-gateway --region $REGION

# Login to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# Build and push images
docker build -f Dockerfile.data-ingestion -t $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/weather-etl/data-ingestion:latest .
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/weather-etl/data-ingestion:latest

# Repeat for other services...
```

### 5. Deploy dashboard to S3/CloudFront

```bash
cd dashboard

# Build for production
npm run build

# Create S3 bucket for dashboard
aws s3 mb s3://weather-dashboard-unique-suffix --region us-east-1

# Enable static website hosting
aws s3 website s3://weather-dashboard-unique-suffix --index-document index.html

# Upload build files
aws s3 sync out/ s3://weather-dashboard-unique-suffix --delete

# Create CloudFront distribution
aws cloudfront create-distribution --distribution-config file://cloudfront-config.json
```

## Essential AWS tools for local development

### Install the following tools on your Linux machine:

1. **AWS CLI v2** - Command line interface for AWS services
2. **AWS CDK** - Infrastructure as Code framework  
3. **Terraform** - Alternative Infrastructure as Code tool
4. **Docker** - Container runtime for local development
5. **kubectl** - Kubernetes CLI (if using EKS)
6. **Helm** - Kubernetes package manager
7. **eksctl** - EKS cluster management tool

```bash
# Install additional tools
pip install awscli-local  # LocalStack support
npm install -g @aws-cdk/aws-lambda @aws-cdk/aws-ecs @aws-cdk/aws-rds

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Install Helm
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update && sudo apt-get install helm
```

## Cost optimization

### AWS cost estimates (Monthly)
- **RDS db.t3.micro**: ~$15
- **ElastiCache cache.t3.micro**: ~$12  
- **ECS Fargate (4 services)**: ~$30-50
- **Application Load Balancer**: ~$16
- **S3 Storage**: ~$1-5 (depending on data volume)
- **CloudFront**: ~$1-10 (depending on traffic)
- **Data Transfer**: Variable

**Total estimated monthly cost: $75-115**

### Cost reduction
1. Use Spot instances for non-critical workloads
2. Implement S3 lifecycle policies for data archival
3. Use CloudWatch scheduled scaling for ECS services
4. Enable RDS automatic scaling
5. Use Reserved Instances for predictable workloads

## Monitoring and observability

### CloudWatch configuration
- **Log Groups**: One per microservice
- **Metrics**: Custom business metrics
- **Alarms**: System health and business alerts
- **Dashboards**: Real-time operational views

### Prometheus and grafana (Optional)
```bash
# Deploy monitoring stack
kubectl apply -f monitoring/prometheus-operator.yaml
kubectl apply -f monitoring/grafana-dashboard.yaml
```

## Security

1. **Network security**
   - Use VPC with private subnets for databases
   - Security groups with minimal required access
   - WAF for API Gateway protection

2. **Data security**
   - Encrypt data at rest (RDS, S3, EBS)
   - Use IAM roles instead of access keys
   - Enable CloudTrail for audit logging

3. **Application security**
   - Use secrets manager for sensitive data
   - Implement API rate limiting
   - Enable container image scanning

## Troubleshooting

### Common issues

1. **gRPC connection errors**
   ```bash
   # Check service health
   grpc_health_probe -addr=localhost:50051
   
   # Check network connectivity
   telnet localhost 50051
   ```

2. **Database connection**
   ```bash
   # Test PostgreSQL connection
   psql postgresql://weather_user:weather_pass@localhost:5432/weather_db
   
   # Check database logs
   docker logs weather-etl_postgres_1
   ```

3. **MQTT connection**
   ```bash
   # Test MQTT broker
   mosquitto_pub -h localhost -t test -m "hello"
   mosquitto_sub -h localhost -t test
   ```

4. **AWS deployment**
   ```bash
   # Check ECS service status
   aws ecs describe-services --cluster weather-etl-cluster --services data-ingestion
   
   # View CloudWatch logs
   aws logs tail /aws/ecs/weather-etl --follow
   ```

## Performance

### Database optimization
- Create indexes on frequently queried columns
- Use connection pooling
- Configure appropriate buffer sizes

### Redis caching
- Implement cache-aside pattern
- Set appropriate TTL values
- Monitor cache hit ratios

### Microservices scaling
- Configure auto-scaling policies
- Implement circuit breakers
- Use async processing where possible
