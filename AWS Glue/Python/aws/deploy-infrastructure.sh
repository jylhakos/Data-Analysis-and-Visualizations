#!/bin/bash

# AWS Weather ETL Deployment Script
# This script sets up the required AWS infrastructure for the Weather ETL microservices

set -e

# Configuration
PROJECT_NAME="weather-etl"
REGION="us-east-1"
CLUSTER_NAME="${PROJECT_NAME}-cluster"
VPC_NAME="${PROJECT_NAME}-vpc"
S3_BUCKET_NAME="${PROJECT_NAME}-data-lake-$(date +%s)"
RDS_INSTANCE_NAME="${PROJECT_NAME}-db"
ELASTICACHE_CLUSTER_NAME="${PROJECT_NAME}-cache"

echo "🚀 Starting Weather ETL AWS Deployment"
echo "Project: ${PROJECT_NAME}"
echo "Region: ${REGION}"

# Check AWS CLI installation
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed. Please install it first."
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Please run 'aws configure'"
    exit 1
fi

echo "✅ AWS CLI configured and credentials verified"

# Function to create IAM roles
create_iam_roles() {
    echo "📝 Creating IAM roles..."
    
    # ECS Task Execution Role
    aws iam create-role \
        --role-name ${PROJECT_NAME}-ecs-task-execution-role \
        --assume-role-policy-document file://aws/ecs-task-execution-role-trust-policy.json \
        --region ${REGION} || echo "Role may already exist"
    
    # Attach AWS managed policy for ECS task execution
    aws iam attach-role-policy \
        --role-name ${PROJECT_NAME}-ecs-task-execution-role \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy \
        --region ${REGION}
    
    # Create custom policy for weather ETL services
    aws iam create-policy \
        --policy-name ${PROJECT_NAME}-policy \
        --policy-document file://aws/weather-etl-policy.json \
        --region ${REGION} || echo "Policy may already exist"
    
    # Get account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    
    # Attach custom policy to role
    aws iam attach-role-policy \
        --role-name ${PROJECT_NAME}-ecs-task-execution-role \
        --policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-policy \
        --region ${REGION}
    
    echo "✅ IAM roles created successfully"
}

# Function to create VPC and networking
create_vpc() {
    echo "🌐 Creating VPC and networking..."
    
    # Create VPC
    VPC_ID=$(aws ec2 create-vpc \
        --cidr-block 10.0.0.0/16 \
        --tag-specifications "ResourceType=vpc,Tags=[{Key=Name,Value=${VPC_NAME}}]" \
        --query 'Vpc.VpcId' \
        --output text \
        --region ${REGION})
    
    echo "VPC created: ${VPC_ID}"
    
    # Create Internet Gateway
    IGW_ID=$(aws ec2 create-internet-gateway \
        --tag-specifications "ResourceType=internet-gateway,Tags=[{Key=Name,Value=${VPC_NAME}-igw}]" \
        --query 'InternetGateway.InternetGatewayId' \
        --output text \
        --region ${REGION})
    
    # Attach Internet Gateway to VPC
    aws ec2 attach-internet-gateway \
        --internet-gateway-id ${IGW_ID} \
        --vpc-id ${VPC_ID} \
        --region ${REGION}
    
    # Create public subnets
    SUBNET1_ID=$(aws ec2 create-subnet \
        --vpc-id ${VPC_ID} \
        --cidr-block 10.0.1.0/24 \
        --availability-zone ${REGION}a \
        --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${VPC_NAME}-public-1}]" \
        --query 'Subnet.SubnetId' \
        --output text \
        --region ${REGION})
    
    SUBNET2_ID=$(aws ec2 create-subnet \
        --vpc-id ${VPC_ID} \
        --cidr-block 10.0.2.0/24 \
        --availability-zone ${REGION}b \
        --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${VPC_NAME}-public-2}]" \
        --query 'Subnet.SubnetId' \
        --output text \
        --region ${REGION})
    
    # Create private subnets
    PRIVATE_SUBNET1_ID=$(aws ec2 create-subnet \
        --vpc-id ${VPC_ID} \
        --cidr-block 10.0.3.0/24 \
        --availability-zone ${REGION}a \
        --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${VPC_NAME}-private-1}]" \
        --query 'Subnet.SubnetId' \
        --output text \
        --region ${REGION})
    
    PRIVATE_SUBNET2_ID=$(aws ec2 create-subnet \
        --vpc-id ${VPC_ID} \
        --cidr-block 10.0.4.0/24 \
        --availability-zone ${REGION}b \
        --tag-specifications "ResourceType=subnet,Tags=[{Key=Name,Value=${VPC_NAME}-private-2}]" \
        --query 'Subnet.SubnetId' \
        --output text \
        --region ${REGION})
    
    # Create route table for public subnets
    ROUTE_TABLE_ID=$(aws ec2 create-route-table \
        --vpc-id ${VPC_ID} \
        --tag-specifications "ResourceType=route-table,Tags=[{Key=Name,Value=${VPC_NAME}-public-rt}]" \
        --query 'RouteTable.RouteTableId' \
        --output text \
        --region ${REGION})
    
    # Add route to Internet Gateway
    aws ec2 create-route \
        --route-table-id ${ROUTE_TABLE_ID} \
        --destination-cidr-block 0.0.0.0/0 \
        --gateway-id ${IGW_ID} \
        --region ${REGION}
    
    # Associate public subnets with route table
    aws ec2 associate-route-table \
        --subnet-id ${SUBNET1_ID} \
        --route-table-id ${ROUTE_TABLE_ID} \
        --region ${REGION}
    
    aws ec2 associate-route-table \
        --subnet-id ${SUBNET2_ID} \
        --route-table-id ${ROUTE_TABLE_ID} \
        --region ${REGION}
    
    echo "✅ VPC and networking created successfully"
    
    # Export variables for use in other functions
    export VPC_ID SUBNET1_ID SUBNET2_ID PRIVATE_SUBNET1_ID PRIVATE_SUBNET2_ID
}

# Function to create S3 bucket
create_s3_bucket() {
    echo "🪣 Creating S3 data lake bucket..."
    
    aws s3 mb s3://${S3_BUCKET_NAME} --region ${REGION}
    
    # Enable versioning
    aws s3api put-bucket-versioning \
        --bucket ${S3_BUCKET_NAME} \
        --versioning-configuration Status=Enabled \
        --region ${REGION}
    
    # Set up lifecycle policy for cost optimization
    cat > /tmp/lifecycle-policy.json << EOF
{
    "Rules": [
        {
            "ID": "weather-data-lifecycle",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "weather-data/"
            },
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                },
                {
                    "Days": 365,
                    "StorageClass": "DEEP_ARCHIVE"
                }
            ]
        }
    ]
}
EOF
    
    aws s3api put-bucket-lifecycle-configuration \
        --bucket ${S3_BUCKET_NAME} \
        --lifecycle-configuration file:///tmp/lifecycle-policy.json \
        --region ${REGION}
    
    echo "✅ S3 bucket created: ${S3_BUCKET_NAME}"
    export S3_BUCKET_NAME
}

# Function to create RDS database
create_rds() {
    echo "🗄️ Creating RDS PostgreSQL database..."
    
    # Create DB subnet group
    aws rds create-db-subnet-group \
        --db-subnet-group-name ${PROJECT_NAME}-db-subnet-group \
        --db-subnet-group-description "Subnet group for weather ETL database" \
        --subnet-ids ${PRIVATE_SUBNET1_ID} ${PRIVATE_SUBNET2_ID} \
        --region ${REGION} || echo "Subnet group may already exist"
    
    # Create security group for RDS
    RDS_SG_ID=$(aws ec2 create-security-group \
        --group-name ${PROJECT_NAME}-rds-sg \
        --description "Security group for weather ETL RDS" \
        --vpc-id ${VPC_ID} \
        --query 'GroupId' \
        --output text \
        --region ${REGION})
    
    # Allow PostgreSQL access from VPC
    aws ec2 authorize-security-group-ingress \
        --group-id ${RDS_SG_ID} \
        --protocol tcp \
        --port 5432 \
        --cidr 10.0.0.0/16 \
        --region ${REGION}
    
    # Create RDS instance
    aws rds create-db-instance \
        --db-instance-identifier ${RDS_INSTANCE_NAME} \
        --db-instance-class db.t3.micro \
        --engine postgres \
        --engine-version 15.4 \
        --master-username weather_user \
        --master-user-password WeatherPass123! \
        --allocated-storage 20 \
        --db-name weather_db \
        --vpc-security-group-ids ${RDS_SG_ID} \
        --db-subnet-group-name ${PROJECT_NAME}-db-subnet-group \
        --backup-retention-period 7 \
        --multi-az false \
        --storage-encrypted \
        --region ${REGION} || echo "RDS instance may already exist"
    
    echo "✅ RDS database creation initiated (this may take several minutes)"
}

# Function to create ElastiCache cluster
create_elasticache() {
    echo "⚡ Creating ElastiCache Redis cluster..."
    
    # Create ElastiCache subnet group
    aws elasticache create-cache-subnet-group \
        --cache-subnet-group-name ${PROJECT_NAME}-cache-subnet-group \
        --cache-subnet-group-description "Subnet group for weather ETL cache" \
        --subnet-ids ${PRIVATE_SUBNET1_ID} ${PRIVATE_SUBNET2_ID} \
        --region ${REGION} || echo "Cache subnet group may already exist"
    
    # Create security group for ElastiCache
    CACHE_SG_ID=$(aws ec2 create-security-group \
        --group-name ${PROJECT_NAME}-cache-sg \
        --description "Security group for weather ETL cache" \
        --vpc-id ${VPC_ID} \
        --query 'GroupId' \
        --output text \
        --region ${REGION})
    
    # Allow Redis access from VPC
    aws ec2 authorize-security-group-ingress \
        --group-id ${CACHE_SG_ID} \
        --protocol tcp \
        --port 6379 \
        --cidr 10.0.0.0/16 \
        --region ${REGION}
    
    # Create Redis cluster
    aws elasticache create-cache-cluster \
        --cache-cluster-id ${ELASTICACHE_CLUSTER_NAME} \
        --cache-node-type cache.t3.micro \
        --engine redis \
        --num-cache-nodes 1 \
        --cache-subnet-group-name ${PROJECT_NAME}-cache-subnet-group \
        --security-group-ids ${CACHE_SG_ID} \
        --region ${REGION} || echo "ElastiCache cluster may already exist"
    
    echo "✅ ElastiCache cluster creation initiated"
}

# Function to create ECS cluster
create_ecs_cluster() {
    echo "🐳 Creating ECS cluster..."
    
    aws ecs create-cluster \
        --cluster-name ${CLUSTER_NAME} \
        --capacity-providers FARGATE \
        --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1 \
        --region ${REGION}
    
    echo "✅ ECS cluster created: ${CLUSTER_NAME}"
    export CLUSTER_NAME
}

# Function to create IoT Core resources
create_iot_core() {
    echo "📡 Setting up AWS IoT Core..."
    
    # Create IoT policy
    cat > /tmp/iot-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "iot:Connect",
                "iot:Publish",
                "iot:Subscribe",
                "iot:Receive"
            ],
            "Resource": "*"
        }
    ]
}
EOF
    
    aws iot create-policy \
        --policy-name ${PROJECT_NAME}-iot-policy \
        --policy-document file:///tmp/iot-policy.json \
        --region ${REGION} || echo "IoT policy may already exist"
    
    echo "✅ AWS IoT Core configured"
}

# Function to output deployment information
output_deployment_info() {
    echo ""
    echo "🎉 AWS Weather ETL Infrastructure Deployment Complete!"
    echo ""
    echo "📋 Deployment Summary:"
    echo "====================="
    echo "Project Name: ${PROJECT_NAME}"
    echo "Region: ${REGION}"
    echo "VPC ID: ${VPC_ID}"
    echo "ECS Cluster: ${CLUSTER_NAME}"
    echo "S3 Bucket: ${S3_BUCKET_NAME}"
    echo "RDS Instance: ${RDS_INSTANCE_NAME}"
    echo "ElastiCache Cluster: ${ELASTICACHE_CLUSTER_NAME}"
    echo ""
    echo "🔧 Next Steps:"
    echo "1. Wait for RDS and ElastiCache to become available (5-10 minutes)"
    echo "2. Build and push Docker images to ECR"
    echo "3. Create ECS task definitions and services"
    echo "4. Set up Application Load Balancer"
    echo "5. Configure Route 53 for custom domain"
    echo "6. Set up CloudWatch dashboards and alarms"
    echo ""
    echo "📄 Save this information for the next deployment steps!"
    
    # Save configuration to file
    cat > aws-deployment-config.env << EOF
PROJECT_NAME=${PROJECT_NAME}
REGION=${REGION}
VPC_ID=${VPC_ID}
SUBNET1_ID=${SUBNET1_ID}
SUBNET2_ID=${SUBNET2_ID}
PRIVATE_SUBNET1_ID=${PRIVATE_SUBNET1_ID}
PRIVATE_SUBNET2_ID=${PRIVATE_SUBNET2_ID}
CLUSTER_NAME=${CLUSTER_NAME}
S3_BUCKET_NAME=${S3_BUCKET_NAME}
RDS_INSTANCE_NAME=${RDS_INSTANCE_NAME}
ELASTICACHE_CLUSTER_NAME=${ELASTICACHE_CLUSTER_NAME}
EOF
    
    echo "💾 Configuration saved to: aws-deployment-config.env"
}

# Main deployment function
main() {
    echo "Starting infrastructure deployment..."
    
    create_iam_roles
    create_vpc
    create_s3_bucket
    create_rds
    create_elasticache
    create_ecs_cluster
    create_iot_core
    output_deployment_info
    
    echo ""
    echo "✅ Infrastructure deployment completed successfully!"
    echo "🕒 Please allow 5-10 minutes for RDS and ElastiCache to become available."
}

# Run main function
main
