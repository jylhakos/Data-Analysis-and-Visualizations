# AWS credentials and IAM roles reference

## Overview

This document provides a quick reference for setting up AWS credentials and IAM roles required for deploying the Arcee Agent to Amazon SageMaker.

## Setup (3 Steps)

### Step 1: Configure AWS credentials
```bash
# Get your AWS Access Key ID and Secret Access Key from AWS Console
# IAM → Users → [Your User] → Security credentials → Create access key

aws configure
# AWS Access Key ID [None]: AKIA...
# AWS Secret Access Key [None]: xxxx...
# Default region name [None]: us-east-1
# Default output format [None]: json
```

### Step 2: Create IAM roles (Automated)
```bash
# Run the automated setup script
./scripts/setup_aws.sh

# This creates:
# - SageMaker execution role
# - S3 bucket for model artifacts
# - ECR repository for Docker images
# - All necessary IAM policies
```

### Step 3: Validate
```bash
# Run validation script
./validate_aws_setup.sh

# Load environment variables
source ~/.bashrc
```

## Required IAM roles and permissions

### Primary role: ArceeAgentSageMakerExecutionRole

**Trust policy**: Allows `sagemaker.amazonaws.com` to assume the role

**Attached policies**:
- `AmazonSageMakerFullAccess` - Complete SageMaker access
- `AmazonS3FullAccess` - S3 bucket access for model artifacts
- `AmazonEC2ContainerRegistryFullAccess` - ECR access for Docker images
- `CloudWatchLogsFullAccess` - Logging and monitoring
- `ArceeAgentSageMakerEnhancedPolicy` - Custom policy for advanced operations

## Verification commands

```bash
# Test AWS credentials
aws sts get-caller-identity

# Check SageMaker access
aws sagemaker list-endpoints

# Verify IAM role
aws iam get-role --role-name ArceeAgentSageMakerExecutionRole

# Test S3 access
aws s3 ls

# Test ECR access
aws ecr describe-repositories
```

## AWS resources

| Resource Type | Name/Pattern | Purpose |
|---------------|--------------|---------|
| IAM Role | ArceeAgentSageMakerExecutionRole | SageMaker execution |
| IAM Policy | ArceeAgentSageMakerEnhancedPolicy | Custom permissions |
| S3 Bucket | arcee-agent-{account-id} | Model artifacts storage |
| ECR Repository | arcee-agent | Docker image storage |

## Environment variables

After setup, these variables will be available:

```bash
export AWS_DEFAULT_REGION=us-east-1
export SAGEMAKER_ROLE_ARN=arn:aws:iam::{account}:role/ArceeAgentSageMakerExecutionRole
export S3_BUCKET_NAME=arcee-agent-{account-id}
export ECR_REPOSITORY_URI={account}.dkr.ecr.us-east-1.amazonaws.com/arcee-agent
```

## Troubleshooting

### Issues:

1. **Permission Denied**: Check user IAM policies
2. **Role Already Exists**: Use existing role or delete and recreate
3. **Region Issues**: Ensure us-east-1 is configured
4. **S3 Bucket Name Conflicts**: Bucket names must be globally unique

### Fixes:

```bash
# Reset AWS configuration
aws configure

# Re-run setup
./scripts/setup_aws.sh

# Validate again
./validate_aws_setup.sh
```

## Cost estimates

| Component | Estimated Cost |
|-----------|----------------|
| IAM Roles | Free |
| S3 Storage | ~$0.02-0.05/month |
| ECR Storage | ~$0.10/month |
| SageMaker Training | ~$2-10/training job |
| SageMaker Inference | ~$50-150/month |

## Next steps

After successful AWS setup:

1. Build Docker image: `docker build -t arcee-agent-api .`
2. Deploy to SageMaker: `python scripts/sagemaker_deployment.py`
3. Test deployment: `curl http://localhost:8000/health`

## Automated setup and validation scripts

- `scripts/setup_aws.sh` - Automated setup script
- `validate_aws_setup.sh` - Validation script
