# Installation of Docker, AWS CLI, and Amazon AWS configuration

## Dependencies installation

### 1. Docker installation

```bash
# Update package list
sudo apt update

# Install Docker
sudo apt install -y docker.io docker-compose

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (allows running docker without sudo)
sudo usermod -aG docker $USER

# Test Docker installation
sudo docker run hello-world

# Verify Docker version
docker --version
```

**Note**: After adding user to docker group, you may need to log out and log back in or restart the terminal for the changes to take effect.

### 2. AWS CLI v2 installation

```bash
# Method 1: Download and install official AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Method 2: Install via apt (alternative)
sudo apt install -y awscli

# Verify installation
aws --version

# Should show: aws-cli/2.x.x Python/3.x.x Linux/x.x.x
```

### 3. AWS credentials configuration

```bash
# Configure AWS credentials interactively
aws configure

# You will be prompted for:
# AWS Access Key ID [None]: YOUR_ACCESS_KEY_ID
# AWS Secret Access Key [None]: YOUR_SECRET_ACCESS_KEY
# Default region name [None]: us-east-1 (or your preferred region)
# Default output format [None]: json
```

#### Alternative: Configure AWS credentials manually

```bash
# Create AWS credentials directory
mkdir -p ~/.aws

# Create credentials file
cat << EOF > ~/.aws/credentials
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
EOF

# Create config file
cat << EOF > ~/.aws/config
[default]
region = us-east-1
output = json
EOF

# Set proper permissions
chmod 600 ~/.aws/credentials
chmod 600 ~/.aws/config
```

### 4. Verify AWS configuration

```bash
# Test AWS credentials
aws sts get-caller-identity

# Should return your AWS account details:
# {
#     "UserId": "AIDXXXXXXXXXXXXXXXXX",
#     "Account": "123456789012",
#     "Arn": "arn:aws:iam::123456789012:user/your-username"
# }

# List S3 buckets (if you have any)
aws s3 ls

# Check AWS regions
aws ec2 describe-regions --output table
```

## Required AWS IAM permissions for SageMaker

Your AWS user/role needs the following permissions:

### 1. Create IAM role for SageMaker

```bash
# Create SageMaker execution role
aws iam create-role \
    --role-name SageMakerExecutionRole \
    --assume-role-policy-document '{
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
    }'

# Attach SageMaker execution policy
aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

# Attach S3 access policy
aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess

# Attach ECR access policy
aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess
```

### 2. Required user permissions

Your AWS user needs these managed policies:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`
- `AmazonEC2ContainerRegistryFullAccess`
- `IAMFullAccess` (to create roles)

## Environment variables (Optional)

```bash
# Add to ~/.bashrc or ~/.profile
export AWS_DEFAULT_REGION=us-east-1
export AWS_PROFILE=default

# SageMaker specific
export SAGEMAKER_ROLE_ARN=arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole
```

## Troubleshooting

### Docker issues

```bash
# If "permission denied" error
sudo usermod -aG docker $USER
newgrp docker  # Apply group changes immediately

# If Docker daemon not running
sudo systemctl start docker
sudo systemctl status docker
```

### AWS CLI issues

```bash
# If aws command not found
export PATH="/usr/local/bin:$PATH"

# If credentials issues
aws configure list
aws sts get-caller-identity

# If region issues
aws configure set region us-east-1
```

### SageMaker issues

```bash
# Check SageMaker availability in your region
aws sagemaker list-endpoints

# Verify IAM role
aws iam get-role --role-name SageMakerExecutionRole
```

## Verification script

```bash
#!/bin/bash
echo "=== System Verification ==="

echo "1. Docker:"
docker --version && echo "✅ Docker installed" || echo "❌ Docker not found"

echo "2. AWS CLI:"
aws --version && echo "✅ AWS CLI installed" || echo "❌ AWS CLI not found"

echo "3. AWS Credentials:"
aws sts get-caller-identity >/dev/null 2>&1 && echo "✅ AWS credentials configured" || echo "❌ AWS credentials not configured"

echo "4. Docker Service:"
sudo systemctl is-active docker >/dev/null 2>&1 && echo "✅ Docker service running" || echo "❌ Docker service not running"

echo "5. SageMaker Access:"
aws sagemaker list-endpoints >/dev/null 2>&1 && echo "✅ SageMaker accessible" || echo "❌ SageMaker not accessible"
```

## Steps after dependencies installation

1. **Test the Arcee Agent project**:
   ```bash
   cd "/path/to/Arcee Agent"
   python test_arcee_agent.py
   ```

2. **Build Docker image**:
   ```bash
   docker build -t arcee-agent-api .
   ```

3. **Test local Docker deployment**:
   ```bash
   docker run -p 8000:8000 arcee-agent-api
   ```

4. **Deploy to SageMaker**:
   ```bash
   python scripts/sagemaker_deployment.py
   ```

This completes the installation of all required system dependencies for the Arcee Agent AWS SageMaker deployment.
