# 📋 AWS Deployment Checklist

## Pre-Deployment Checklist

### ✅ Local Environment Setup
- [ ] AWS CLI installed and configured
- [ ] Docker installed and running
- [ ] Python 3.11+ installed
- [ ] Git installed
- [ ] Required IAM permissions configured

### ✅ AWS Account Preparation
- [ ] AWS account with billing enabled
- [ ] IAM user/role with deployment permissions
- [ ] Service limits checked (ECS, Kinesis, ECR)
- [ ] Default VPC available or custom VPC planned
- [ ] AWS region selected (default: us-east-1)

### ✅ Project Preparation
- [ ] All source code files present
- [ ] Dependencies listed in requirements.txt
- [ ] Dockerfile.production reviewed
- [ ] Environment variables configured

## Deployment Steps

### 🔧 Step 1: Environment Setup
```bash
# Run the setup script
./setup-environment.sh

# Verify output shows:
# ✅ Docker daemon is running
# ✅ AWS credentials are configured
# ✅ Python dependencies installed
```
**Status**: [ ] Complete

### 🐳 Step 2: Local Testing (Optional but Recommended)
```bash
# Start local environment
docker-compose up -d

# Test the API
./test-api.sh http://localhost:8000

# Stop local environment
docker-compose down
```
**Status**: [ ] Complete / [ ] Skipped

### ☁️ Step 3: AWS Infrastructure Deployment
```bash
# Automated deployment
./deploy/deploy.sh

# OR manual deployment
aws cloudformation deploy \
    --template-file deploy/aws-infrastructure.yaml \
    --stack-name iot-temp-forecast-infrastructure \
    --parameter-overrides EnvironmentName=iot-temp-forecast \
    --capabilities CAPABILITY_IAM \
    --region us-east-1
```
**Expected Output**: CloudFormation stack creation success
**Status**: [ ] Complete

### 🐳 Step 4: Container Image Build & Push
```bash
# Get ECR repository URI
ECR_REPO_URI=$(aws cloudformation describe-stacks \
    --stack-name iot-temp-forecast-infrastructure \
    --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryURI`].OutputValue' \
    --output text)

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin $ECR_REPO_URI

# Build and push
docker build -f Dockerfile.production -t iot-temp-forecast .
docker tag iot-temp-forecast:latest $ECR_REPO_URI:latest
docker push $ECR_REPO_URI:latest
```
**Expected Output**: "The push refers to repository" with successful layers
**Status**: [ ] Complete

### 🚀 Step 5: ECS Service Deployment
```bash
# Force new deployment
aws ecs update-service \
    --cluster iot-temp-forecast-cluster \
    --service iot-temp-forecast-service \
    --force-new-deployment \
    --region us-east-1
```
**Expected Output**: Service update initiated
**Status**: [ ] Complete

## Post-Deployment Verification

### 🌐 Application Access
```bash
# Get Application URL
APP_URL=$(aws cloudformation describe-stacks \
    --stack-name iot-temp-forecast-infrastructure \
    --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' \
    --output text)

echo "Application URL: $APP_URL"
```
**Application URL**: ________________________________
**Status**: [ ] Accessible

### 🧪 API Testing
```bash
# Run comprehensive API tests
./test-api.sh $APP_URL
```
**Expected Results**:
- [ ] ✅ Health check passed
- [ ] ✅ Sensor data submission passed
- [ ] ✅ Historical data retrieval passed
- [ ] ✅ Model status check passed

### 📊 Service Health Checks

#### ECS Service Status
```bash
aws ecs describe-services \
    --cluster iot-temp-forecast-cluster \
    --services iot-temp-forecast-service \
    --region us-east-1
```
**Expected**: 
- [ ] Service status: ACTIVE
- [ ] Running count: 2
- [ ] Desired count: 2

#### Load Balancer Health
```bash
# Check target group health
aws elbv2 describe-target-health \
    --target-group-arn $(aws elbv2 describe-target-groups \
        --names iot-temp-forecast-tg \
        --query 'TargetGroups[0].TargetGroupArn' \
        --output text) \
    --region us-east-1
```
**Expected**: 
- [ ] Target health: healthy
- [ ] Both targets responding

#### Kinesis Stream Status
```bash
aws kinesis describe-stream \
    --stream-name iot-temp-forecast-temperature-sensor-stream \
    --region us-east-1
```
**Expected**: 
- [ ] Stream status: ACTIVE
- [ ] Shard count: 2

### 📝 CloudWatch Logs
```bash
# View application logs
aws logs tail /ecs/iot-temp-forecast --follow --region us-east-1
```
**Expected**: 
- [ ] No error messages
- [ ] Application startup logs visible
- [ ] API request logs appearing

## 🎉 Deployment Success Criteria

### Functional Requirements
- [ ] API responds to health checks
- [ ] Can submit sensor data via POST /sensor-data
- [ ] Can retrieve historical data via GET /historical-data
- [ ] Can access API documentation at /docs
- [ ] Model training can be initiated
- [ ] Forecasting works when model is trained

### Non-Functional Requirements
- [ ] Application load balancer distributes traffic
- [ ] Multiple ECS tasks running (high availability)
- [ ] CloudWatch logs are being generated
- [ ] All AWS services show as healthy
- [ ] Response times are acceptable (< 2 seconds)

### Security Requirements
- [ ] Application runs with non-root user
- [ ] Security groups properly configured
- [ ] IAM roles follow least privilege principle
- [ ] No sensitive data in logs

## 📋 Access Information

Once deployment is complete, record these details:

| Item | Value |
|------|-------|
| Application URL | _________________________ |
| API Documentation | {APP_URL}/docs |
| ECS Cluster | iot-temp-forecast-cluster |
| Kinesis Stream | iot-temp-forecast-temperature-sensor-stream |
| CloudWatch Log Group | /ecs/iot-temp-forecast |
| ECR Repository | iot-temp-forecast-fastapi |

## 🔧 Common Issues & Solutions

### Issue: ECS tasks not starting
**Check**: 
- [ ] ECR image exists and is accessible
- [ ] IAM task role has correct permissions
- [ ] CloudWatch logs for error messages

### Issue: Load balancer health checks failing
**Check**: 
- [ ] Application responds on port 8000
- [ ] Security groups allow ALB to reach ECS tasks
- [ ] Health check path (/) returns 200

### Issue: Can't push to ECR
**Check**: 
- [ ] ECR login command successful
- [ ] Repository exists
- [ ] IAM permissions for ECR

### Issue: Kinesis access denied
**Check**: 
- [ ] ECS task role has Kinesis permissions
- [ ] Stream name is correct
- [ ] Region is correct

## 🧹 Cleanup Instructions

When you're done testing, clean up to avoid charges:

```bash
# Delete CloudFormation stack
aws cloudformation delete-stack \
    --stack-name iot-temp-forecast-infrastructure \
    --region us-east-1

# Verify deletion
aws cloudformation describe-stacks \
    --stack-name iot-temp-forecast-infrastructure \
    --region us-east-1
```

**Status**: [ ] Cleanup Complete

## 📞 Support Resources

- **AWS Documentation**: https://docs.aws.amazon.com/
- **ECS Troubleshooting**: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/troubleshooting.html
- **Kinesis Developer Guide**: https://docs.aws.amazon.com/streams/latest/dev/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Project Documentation**: README.md and DEPLOYMENT_GUIDE.md

---

**Deployment Date**: _______________
**Deployed By**: _______________
**Environment**: _______________
**Notes**: _______________
