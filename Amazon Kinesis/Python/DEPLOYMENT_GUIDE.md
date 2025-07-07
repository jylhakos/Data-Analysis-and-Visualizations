# 🚀 AWS Deployment Guide for IoT Temperature Forecasting Application

This guide walks you through deploying your FastAPI application with Amazon Kinesis and scikit-learn ML model to AWS using Docker containers.

## 📋 Prerequisites

### Local Environment Setup

1. **Install Required Tools:**
   ```bash
   # AWS CLI
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install
   
   # Docker
   sudo apt-get update
   sudo apt-get install docker.io docker-compose
   sudo usermod -aG docker $USER
   # Log out and back in for group changes
   
   # Python 3.11+
   sudo apt-get install python3 python3-pip python3-venv
   ```

2. **Configure AWS Credentials:**
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, Region (us-east-1), and output format (json)
   
   # Verify configuration
   aws sts get-caller-identity
   ```

3. **Required IAM Permissions:**
   Your AWS user/role needs these permissions:
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "cloudformation:*",
                   "ecs:*",
                   "ecr:*",
                   "kinesis:*",
                   "ec2:*",
                   "elasticloadbalancing:*",
                   "iam:CreateRole",
                   "iam:AttachRolePolicy",
                   "iam:PassRole",
                   "iam:GetRole",
                   "logs:*"
               ],
               "Resource": "*"
           }
       ]
   }
   ```

## 🔧 Environment Setup

1. **Run the setup script:**
   ```bash
   ./setup-environment.sh
   ```

   This script will:
   - Check all required tools are installed
   - Verify AWS credentials
   - Create Python virtual environment
   - Install dependencies
   - Create `.env` file for configuration
   - Make deployment scripts executable

## 🏗️ Local Development & Testing

1. **Start the local development stack:**
   ```bash
   docker-compose up -d
   ```

2. **Test the FastAPI application:**
   ```bash
   # Check API health
   curl http://localhost:8000/
   
   # View API documentation
   curl http://localhost:8000/docs
   ```

3. **Simulate IoT sensors:**
   ```bash
   # In a new terminal
   ./simulate-sensors.sh
   ```

4. **Test the complete flow:**
   ```bash
   # Submit sensor data
   curl -X POST "http://localhost:8000/sensor-data" \
     -H "Content-Type: application/json" \
     -d '{
       "sensor_id": "test-001",
       "temperature": 25.5,
       "humidity": 60.0,
       "location": "TestLab"
     }'
   
   # Train the model (after collecting some data)
   curl -X POST "http://localhost:8000/model/train"
   
   # Get forecast
   curl -X POST "http://localhost:8000/forecast" \
     -H "Content-Type: application/json" \
     -d '{
       "sensor_id": "test-001",
       "hours_ahead": 24
     }'
   ```

## ☁️ AWS Deployment

### Automated Deployment

Run the automated deployment script:

```bash
./deploy/deploy.sh
```

This script will:
1. 🏗️ Deploy AWS infrastructure using CloudFormation
2. 🐳 Build and push Docker image to ECR
3. 🚀 Deploy to ECS Fargate
4. 🌐 Configure Application Load Balancer
5. 📊 Set up monitoring and logging

### Manual Deployment Steps

If you prefer to deploy manually:

#### 1. Deploy Infrastructure
```bash
aws cloudformation deploy \
    --template-file deploy/aws-infrastructure.yaml \
    --stack-name iot-temp-forecast-infrastructure \
    --parameter-overrides EnvironmentName=iot-temp-forecast \
    --capabilities CAPABILITY_IAM \
    --region us-east-1
```

#### 2. Build and Push Docker Image
```bash
# Get ECR repository URI
ECR_REPO_URI=$(aws cloudformation describe-stacks \
    --stack-name iot-temp-forecast-infrastructure \
    --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryURI`].OutputValue' \
    --output text)

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REPO_URI

# Build and push
docker build -f Dockerfile.production -t iot-temp-forecast .
docker tag iot-temp-forecast:latest $ECR_REPO_URI:latest
docker push $ECR_REPO_URI:latest
```

#### 3. Update ECS Service
```bash
aws ecs update-service \
    --cluster iot-temp-forecast-cluster \
    --service iot-temp-forecast-service \
    --force-new-deployment \
    --region us-east-1
```

## 🌐 Accessing Your Application

After deployment, get your application URL:

```bash
aws cloudformation describe-stacks \
    --stack-name iot-temp-forecast-infrastructure \
    --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' \
    --output text
```

## 📊 AWS Architecture Overview

### Services Deployed:

1. **Amazon ECS (Fargate)**
   - Serverless container orchestration
   - Auto-scaling based on demand
   - Health checks and automatic recovery

2. **Amazon Kinesis Data Streams**
   - Real-time data streaming
   - 2 shards for parallel processing
   - 24-hour data retention

3. **Application Load Balancer**
   - HTTP/HTTPS traffic distribution
   - Health checks
   - SSL termination ready

4. **Amazon ECR**
   - Private Docker registry
   - Image vulnerability scanning
   - Lifecycle policies

5. **VPC & Networking**
   - Isolated network environment
   - Public/private subnets
   - Security groups for access control

6. **CloudWatch**
   - Application logs
   - Metrics and monitoring
   - Alerts and dashboards

### Cost Estimation (Monthly):

- **ECS Fargate (2 tasks)**: ~$25-50
- **Application Load Balancer**: ~$20
- **Kinesis Data Streams**: ~$20-40
- **ECR Storage**: ~$1-5
- **CloudWatch Logs**: ~$5-10
- **Data Transfer**: ~$5-15

**Total estimated cost**: $75-140/month for moderate usage

## 🔍 Monitoring and Troubleshooting

### CloudWatch Logs
```bash
# View application logs
aws logs tail /ecs/iot-temp-forecast --follow --region us-east-1

# View specific log stream
aws logs describe-log-streams \
    --log-group-name /ecs/iot-temp-forecast \
    --region us-east-1
```

### ECS Service Health
```bash
# Check service status
aws ecs describe-services \
    --cluster iot-temp-forecast-cluster \
    --services iot-temp-forecast-service \
    --region us-east-1

# List running tasks
aws ecs list-tasks \
    --cluster iot-temp-forecast-cluster \
    --service-name iot-temp-forecast-service \
    --region us-east-1
```

### Kinesis Stream Monitoring
```bash
# Stream status
aws kinesis describe-stream \
    --stream-name iot-temp-forecast-temperature-sensor-stream \
    --region us-east-1

# Put sample record
aws kinesis put-record \
    --stream-name iot-temp-forecast-temperature-sensor-stream \
    --partition-key sensor-001 \
    --data '{"sensor_id":"sensor-001","temperature":25.0,"humidity":60.0,"timestamp":"2024-01-01T12:00:00Z"}' \
    --region us-east-1
```

## 🔄 Updates and Maintenance

### Update Application
```bash
# Build and push new image
docker build -f Dockerfile.production -t iot-temp-forecast:v2 .
docker tag iot-temp-forecast:v2 $ECR_REPO_URI:v2
docker push $ECR_REPO_URI:v2

# Update ECS service
aws ecs update-service \
    --cluster iot-temp-forecast-cluster \
    --service iot-temp-forecast-service \
    --force-new-deployment \
    --region us-east-1
```

### Scale Application
```bash
# Scale to 4 tasks
aws ecs update-service \
    --cluster iot-temp-forecast-cluster \
    --service iot-temp-forecast-service \
    --desired-count 4 \
    --region us-east-1
```

### Backup and Recovery
- **Model files**: Stored in ECS volumes (consider S3 for persistence)
- **Configuration**: Version controlled in CloudFormation
- **Data**: Kinesis provides built-in durability and replay capabilities

## 🧹 Cleanup

To avoid ongoing charges, delete the stack when done:

```bash
aws cloudformation delete-stack \
    --stack-name iot-temp-forecast-infrastructure \
    --region us-east-1
```

## 🚨 Common Issues and Solutions

### Issue: ECS tasks keep restarting
**Solution**: Check CloudWatch logs for application errors

### Issue: Cannot connect to Kinesis
**Solution**: Verify IAM roles have Kinesis permissions

### Issue: Load balancer health checks failing
**Solution**: Ensure FastAPI app responds to `GET /` on port 8000

### Issue: Docker build fails
**Solution**: Check local Docker installation and network connectivity

## 📈 Production Best Practices

1. **Security**:
   - Use HTTPS with SSL certificates
   - Implement API authentication
   - Enable VPC Flow Logs
   - Regular security updates

2. **Performance**:
   - Use CloudFront for global distribution
   - Implement Redis for caching
   - Optimize ML model inference
   - Database for persistent storage

3. **Reliability**:
   - Multi-AZ deployment
   - Auto-scaling policies
   - Backup strategies
   - Disaster recovery plan

4. **Cost Optimization**:
   - Use Spot instances for batch processing
   - Implement lifecycle policies
   - Monitor and optimize resource usage
   - Reserved capacity for predictable workloads

## 📚 Additional Resources

- [AWS ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [Amazon Kinesis Developer Guide](https://docs.aws.amazon.com/streams/latest/dev/)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)

## 🆘 Support

For issues with this deployment:
1. Check CloudWatch logs first
2. Verify AWS service limits
3. Review IAM permissions
4. Check network connectivity
5. Consult AWS documentation
