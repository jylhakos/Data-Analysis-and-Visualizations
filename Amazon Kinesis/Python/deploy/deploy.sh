#!/bin/bash
set -e

# Configuration
PROJECT_NAME="iot-temp-forecast"
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "🚀 Starting deployment of IoT Temperature Forecasting Application"
echo "Project: $PROJECT_NAME"
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"

# Step 1: Deploy AWS Infrastructure
echo "📦 Step 1: Deploying AWS Infrastructure..."
aws cloudformation deploy \
    --template-file deploy/aws-infrastructure.yaml \
    --stack-name ${PROJECT_NAME}-infrastructure \
    --parameter-overrides EnvironmentName=$PROJECT_NAME \
    --capabilities CAPABILITY_IAM \
    --region $AWS_REGION

echo "✅ Infrastructure deployed successfully"

# Step 2: Get ECR Repository URI
echo "📋 Step 2: Getting ECR Repository URI..."
ECR_REPO_URI=$(aws cloudformation describe-stacks \
    --stack-name ${PROJECT_NAME}-infrastructure \
    --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryURI`].OutputValue' \
    --output text \
    --region $AWS_REGION)

echo "ECR Repository: $ECR_REPO_URI"

# Step 3: Login to ECR
echo "🔐 Step 3: Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO_URI

# Step 4: Build and Push Docker Image
echo "🔨 Step 4: Building Docker image..."
docker build -t ${PROJECT_NAME}-fastapi .

echo "📤 Step 5: Tagging and pushing to ECR..."
docker tag ${PROJECT_NAME}-fastapi:latest $ECR_REPO_URI:latest
docker push $ECR_REPO_URI:latest

# Step 6: Update ECS Service
echo "🔄 Step 6: Updating ECS Service..."
aws ecs update-service \
    --cluster ${PROJECT_NAME}-cluster \
    --service ${PROJECT_NAME}-service \
    --force-new-deployment \
    --region $AWS_REGION

# Step 7: Get Application URL
echo "🌐 Step 7: Getting Application URL..."
APP_URL=$(aws cloudformation describe-stacks \
    --stack-name ${PROJECT_NAME}-infrastructure \
    --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerURL`].OutputValue' \
    --output text \
    --region $AWS_REGION)

echo ""
echo "🎉 Deployment completed successfully!"
echo "Application URL: $APP_URL"
echo "API Documentation: $APP_URL/docs"
echo ""
echo "📊 Monitoring and Management:"
echo "ECS Cluster: https://console.aws.amazon.com/ecs/home?region=$AWS_REGION#/clusters/${PROJECT_NAME}-cluster"
echo "Kinesis Stream: https://console.aws.amazon.com/kinesis/home?region=$AWS_REGION#/streams/details/${PROJECT_NAME}-temperature-sensor-stream"
echo "CloudWatch Logs: https://console.aws.amazon.com/cloudwatch/home?region=$AWS_REGION#logsV2:log-groups/log-group/%2Fecs%2F${PROJECT_NAME}"
echo ""
echo "🧪 Test the API:"
echo "curl $APP_URL/"
echo "curl -X POST $APP_URL/sensor-data -H 'Content-Type: application/json' -d '{\"sensor_id\":\"test-001\",\"temperature\":25.5,\"humidity\":60.0}'"
