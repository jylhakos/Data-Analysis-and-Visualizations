#!/bin/bash

# AWS Setup Script for Arcee Agent SageMaker Deployment
# This script creates the necessary IAM roles and configures AWS for deployment

set -e  # Exit on any error

echo "🔐 AWS Setup for Arcee Agent SageMaker Deployment"
echo "================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    log_error "AWS CLI not found. Please install AWS CLI first."
    echo "Run: sudo apt install awscli"
    exit 1
fi

log_success "AWS CLI found"

# Check if AWS is configured
if ! aws sts get-caller-identity >/dev/null 2>&1; then
    log_error "AWS credentials not configured."
    echo ""
    echo "Please run: aws configure"
    echo "You'll need:"
    echo "  - AWS Access Key ID"
    echo "  - AWS Secret Access Key"
    echo "  - Default region (recommend: us-east-1)"
    echo "  - Output format (recommend: json)"
    exit 1
fi

log_success "AWS credentials configured"

# Get AWS account info
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
REGION=$(aws configure get region)

echo ""
log_info "AWS Account ID: $ACCOUNT_ID"
log_info "User ARN: $USER_ARN"
log_info "Region: $REGION"
echo ""

# Step 1: Create SageMaker Execution Role
echo "📝 Step 1: Creating SageMaker Execution Role"
echo "============================================"

ROLE_NAME="ArceeAgentSageMakerExecutionRole"

# Create trust policy
cat > /tmp/sagemaker-trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

# Create the role
if aws iam create-role \
    --role-name $ROLE_NAME \
    --assume-role-policy-document file:///tmp/sagemaker-trust-policy.json \
    --description "Execution role for Arcee Agent SageMaker deployment" >/dev/null 2>&1; then
    log_success "Created IAM role: $ROLE_NAME"
else
    log_warning "IAM role $ROLE_NAME already exists"
fi

# Step 2: Attach AWS Managed Policies
echo ""
echo "🔗 Step 2: Attaching AWS Managed Policies"
echo "========================================="

POLICIES=(
    "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    "arn:aws:iam::aws:policy/AmazonS3FullAccess"
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess"
    "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
)

for policy in "${POLICIES[@]}"; do
    if aws iam attach-role-policy --role-name $ROLE_NAME --policy-arn $policy 2>/dev/null; then
        log_success "Attached policy: $(basename $policy)"
    else
        log_warning "Policy already attached: $(basename $policy)"
    fi
done

# Step 3: Create Custom Policy
echo ""
echo "📋 Step 3: Creating Custom Policy"
echo "================================="

CUSTOM_POLICY_NAME="ArceeAgentSageMakerPolicy"

cat > /tmp/sagemaker-custom-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:*",
                "iam:PassRole",
                "ecr:*",
                "logs:*",
                "s3:*",
                "lambda:InvokeFunction"
            ],
            "Resource": "*"
        }
    ]
}
EOF

# Create custom policy
if aws iam create-policy \
    --policy-name $CUSTOM_POLICY_NAME \
    --policy-document file:///tmp/sagemaker-custom-policy.json \
    --description "Custom policy for Arcee Agent SageMaker operations" >/dev/null 2>&1; then
    log_success "Created custom policy: $CUSTOM_POLICY_NAME"
else
    log_warning "Custom policy $CUSTOM_POLICY_NAME already exists"
fi

# Attach custom policy to role
if aws iam attach-role-policy \
    --role-name $ROLE_NAME \
    --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${CUSTOM_POLICY_NAME}" 2>/dev/null; then
    log_success "Attached custom policy to role"
else
    log_warning "Custom policy already attached to role"
fi

# Step 4: Get Role ARN
echo ""
echo "🔍 Step 4: Getting Role ARN"
echo "==========================="

ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)
log_success "Role ARN: $ROLE_ARN"

# Step 5: Create ECR Repository
echo ""
echo "🐳 Step 5: Creating ECR Repository"
echo "=================================="

ECR_REPO_NAME="arcee-agent"

if aws ecr create-repository --repository-name $ECR_REPO_NAME >/dev/null 2>&1; then
    log_success "Created ECR repository: $ECR_REPO_NAME"
else
    log_warning "ECR repository $ECR_REPO_NAME already exists"
fi

ECR_URI=$(aws ecr describe-repositories --repository-names $ECR_REPO_NAME --query 'repositories[0].repositoryUri' --output text)
log_success "ECR URI: $ECR_URI"

# Step 6: Create S3 Bucket for Models
echo ""
echo "🪣 Step 6: Creating S3 Bucket"
echo "============================="

S3_BUCKET_NAME="arcee-agent-models-${ACCOUNT_ID}-${REGION}"

if aws s3 mb "s3://${S3_BUCKET_NAME}" >/dev/null 2>&1; then
    log_success "Created S3 bucket: $S3_BUCKET_NAME"
else
    log_warning "S3 bucket $S3_BUCKET_NAME already exists"
fi

# Step 7: Save Configuration
echo ""
echo "💾 Step 7: Saving Configuration"
echo "==============================="

# Create .env file
cat > .env << EOF
# AWS Configuration for Arcee Agent
AWS_ACCOUNT_ID=$ACCOUNT_ID
AWS_REGION=$REGION
SAGEMAKER_ROLE_ARN=$ROLE_ARN
ECR_REPOSITORY_URI=$ECR_URI
S3_BUCKET_NAME=$S3_BUCKET_NAME

# SageMaker Configuration
SAGEMAKER_INSTANCE_TYPE=ml.m5.large
SAGEMAKER_INSTANCE_COUNT=1
EOF

log_success "Configuration saved to .env file"

# Add to bashrc if not already there
if ! grep -q "SAGEMAKER_ROLE_ARN" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Arcee Agent AWS Configuration" >> ~/.bashrc
    echo "export SAGEMAKER_ROLE_ARN='$ROLE_ARN'" >> ~/.bashrc
    echo "export ECR_REPOSITORY_URI='$ECR_URI'" >> ~/.bashrc
    echo "export S3_BUCKET_NAME='$S3_BUCKET_NAME'" >> ~/.bashrc
    log_success "Environment variables added to ~/.bashrc"
fi

# Step 8: Verify Setup
echo ""
echo "🧪 Step 8: Verifying Setup"
echo "=========================="

# Test SageMaker access
if aws sagemaker list-endpoints >/dev/null 2>&1; then
    log_success "SageMaker access verified"
else
    log_error "SageMaker access failed"
fi

# Test ECR access
if aws ecr get-login-token >/dev/null 2>&1; then
    log_success "ECR access verified"
else
    log_error "ECR access failed"
fi

# Test S3 access
if aws s3 ls "s3://${S3_BUCKET_NAME}" >/dev/null 2>&1; then
    log_success "S3 access verified"
else
    log_error "S3 access failed"
fi

# Cleanup temp files
rm -f /tmp/sagemaker-trust-policy.json /tmp/sagemaker-custom-policy.json

echo ""
echo "🎉 AWS Setup Complete!"
echo "======================"
echo ""
echo "📋 Summary:"
echo "  ✅ IAM Role: $ROLE_NAME"
echo "  ✅ Role ARN: $ROLE_ARN"
echo "  ✅ ECR Repository: $ECR_URI"
echo "  ✅ S3 Bucket: $S3_BUCKET_NAME"
echo ""
echo "🚀 Next Steps:"
echo "  1. Source environment: source ~/.bashrc"
echo "  2. Test project: python test_arcee_agent.py"
echo "  3. Build Docker: docker build -t arcee-agent-api ."
echo "  4. Deploy to SageMaker: python scripts/sagemaker_deployment.py"
echo ""
echo "💡 Configuration saved in .env file"
echo "🔧 Environment variables added to ~/.bashrc"
echo ""
echo "Your Arcee Agent project is now ready for AWS SageMaker deployment! 🚀"
