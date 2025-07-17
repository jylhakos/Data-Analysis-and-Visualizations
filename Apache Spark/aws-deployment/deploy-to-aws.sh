#!/bin/bash

set -e

DEPLOYMENT_TYPE=${1:-"emr"}
AWS_REGION=${2:-"us-east-1"}
PROJECT_NAME="airtraffic-analysis"

echo "🚀 Deploying AirTraffic Analysis to AWS using $DEPLOYMENT_TYPE"

# Check prerequisites
check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        echo "❌ AWS CLI not found. Please install it first."
        echo "Run: curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip' && unzip awscliv2.zip && sudo ./aws/install"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo "❌ AWS credentials not configured. Run 'aws configure'"
        exit 1
    fi
    
    # Check Terraform for EMR deployment
    if [ "$DEPLOYMENT_TYPE" = "emr" ] && ! command -v terraform &> /dev/null; then
        echo "❌ Terraform not found. Please install it first."
        echo "Visit: https://www.terraform.io/downloads.html"
        exit 1
    fi
    
    echo "✅ Prerequisites check passed"
}

# Generate SSH key pair if it doesn't exist
setup_ssh_key() {
    if [ ! -f ~/.ssh/id_rsa ]; then
        echo "🔑 Generating SSH key pair..."
        ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
    fi
}

# Deploy using EMR
deploy_emr() {
    echo "📦 Deploying to Amazon EMR..."
    
    # Setup SSH key
    setup_ssh_key
    
    cd terraform
    terraform init
    terraform plan -var="aws_region=$AWS_REGION" -var="project_name=$PROJECT_NAME"
    terraform apply -auto-approve -var="aws_region=$AWS_REGION" -var="project_name=$PROJECT_NAME"
    
    # Get outputs
    BUCKET_NAME=$(terraform output -raw s3_bucket_name)
    EMR_DNS=$(terraform output -raw emr_master_public_dns)
    
    echo "📤 Uploading data and notebooks..."
    cd ..
    aws s3 cp 2008.csv s3://$BUCKET_NAME/data/ 2>/dev/null || echo "⚠️  2008.csv not found"
    aws s3 cp 2008_sample.csv s3://$BUCKET_NAME/data/ 2>/dev/null || echo "⚠️  2008_sample.csv not found"
    aws s3 cp carriers.csv s3://$BUCKET_NAME/data/
    aws s3 cp airports.csv s3://$BUCKET_NAME/data/
    aws s3 cp AirTrafficProcessor.ipynb s3://$BUCKET_NAME/notebooks/
    aws s3 cp verify_pyspark_mllib.py s3://$BUCKET_NAME/scripts/
    aws s3 cp quick_verify.py s3://$BUCKET_NAME/scripts/
    
    echo "✅ EMR deployment complete!"
    echo "🌐 JupyterHub URL: https://$EMR_DNS:9443"
    echo "📊 S3 Bucket: $BUCKET_NAME"
    echo "🔑 SSH Access: ssh -i ~/.ssh/id_rsa hadoop@$EMR_DNS"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Wait 10-15 minutes for EMR cluster to be ready"
    echo "2. Access JupyterHub at the URL above"
    echo "3. Login with username: jovyan, password: jupyter"
    echo "4. Navigate to /home/jovyan/notebooks/"
    echo "5. Upload or access AirTrafficProcessor.ipynb"
}

# Deploy using SageMaker
deploy_sagemaker() {
    echo "📦 Deploying to Amazon SageMaker..."
    
    # Get AWS account ID
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ROLE_NAME="SageMakerExecutionRole-AirTraffic"
    ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/$ROLE_NAME"
    
    # Create SageMaker execution role if it doesn't exist
    echo "🔐 Setting up IAM role..."
    
    # Create trust policy
    cat > sagemaker-trust-policy.json << EOF
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
    
    aws iam create-role --role-name $ROLE_NAME \
        --assume-role-policy-document file://sagemaker-trust-policy.json || true
    
    aws iam attach-role-policy --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess || true
    
    aws iam attach-role-policy --role-name $ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess || true
    
    # Create notebook instance
    echo "🚀 Creating SageMaker notebook instance..."
    aws sagemaker create-notebook-instance \
        --notebook-instance-name airtraffic-analysis \
        --instance-type ml.t3.medium \
        --role-arn $ROLE_ARN \
        --volume-size-in-gb 20 || echo "⚠️  Notebook instance may already exist"
    
    # Create S3 bucket for data
    BUCKET_NAME="$PROJECT_NAME-sagemaker-$(date +%s)"
    aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION || true
    
    # Upload data
    echo "📤 Uploading data to S3..."
    aws s3 cp 2008.csv s3://$BUCKET_NAME/data/ 2>/dev/null || echo "⚠️  2008.csv not found"
    aws s3 cp 2008_sample.csv s3://$BUCKET_NAME/data/ 2>/dev/null || echo "⚠️  2008_sample.csv not found"
    aws s3 cp carriers.csv s3://$BUCKET_NAME/data/
    aws s3 cp airports.csv s3://$BUCKET_NAME/data/
    aws s3 cp AirTrafficProcessor.ipynb s3://$BUCKET_NAME/notebooks/
    
    echo "✅ SageMaker deployment initiated!"
    echo "🌐 Check AWS Console: https://console.aws.amazon.com/sagemaker/"
    echo "📊 S3 Bucket: $BUCKET_NAME"
    echo "⏳ Notebook instance will be ready in 5-10 minutes"
    
    # Clean up temporary files
    rm -f sagemaker-trust-policy.json
}

# Deploy using EC2
deploy_ec2() {
    echo "📦 Deploying to Amazon EC2..."
    
    # Get default VPC and subnet
    VPC_ID=$(aws ec2 describe-vpcs --filters "Name=is-default,Values=true" --query "Vpcs[0].VpcId" --output text)
    SUBNET_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$VPC_ID" --query "Subnets[0].SubnetId" --output text)
    
    # Create security group
    echo "🔐 Creating security group..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name airtraffic-jupyter-sg \
        --description "Security group for AirTraffic Jupyter notebook" \
        --vpc-id $VPC_ID \
        --query 'GroupId' --output text) || \
        SG_ID=$(aws ec2 describe-security-groups \
            --filters "Name=group-name,Values=airtraffic-jupyter-sg" \
            --query "SecurityGroups[0].GroupId" --output text)
    
    # Add security group rules
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0 || true
    
    aws ec2 authorize-security-group-ingress \
        --group-id $SG_ID \
        --protocol tcp \
        --port 8888 \
        --cidr 0.0.0.0/0 || true
    
    # Create key pair
    setup_ssh_key
    aws ec2 import-key-pair \
        --key-name $PROJECT_NAME-key \
        --public-key-material fileb://~/.ssh/id_rsa.pub || true
    
    # Create user data script
    cat > user-data.sh << 'EOF'
#!/bin/bash
yum update -y
yum install -y python3 python3-pip git java-11-amazon-corretto

# Create jupyter user
useradd -m jupyter
echo "jupyter ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Switch to jupyter user
su - jupyter << 'JUPYTER_SETUP'

# Install Python packages
pip3 install --user jupyter jupyterlab pyspark pandas numpy matplotlib boto3

# Create Jupyter config
mkdir -p ~/.jupyter
cat > ~/.jupyter/jupyter_notebook_config.py << 'JUPYTER_CONFIG'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.token = 'airtraffic-secure-token-2024'
c.NotebookApp.allow_root = True
c.NotebookApp.notebook_dir = '/home/jupyter/notebooks'
JUPYTER_CONFIG

# Create notebooks directory
mkdir -p /home/jupyter/notebooks

# Download notebook from S3 if bucket is specified
# This would be configured after deployment

# Start Jupyter
nohup /home/jupyter/.local/bin/jupyter lab > /home/jupyter/jupyter.log 2>&1 &

JUPYTER_SETUP

EOF
    
    # Launch instance
    echo "🚀 Launching EC2 instance..."
    INSTANCE_ID=$(aws ec2 run-instances \
        --image-id ami-0c02fb55956c7d316 \
        --instance-type t3.large \
        --key-name $PROJECT_NAME-key \
        --security-group-ids $SG_ID \
        --subnet-id $SUBNET_ID \
        --user-data file://user-data.sh \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=AirTraffic-Jupyter}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
    
    # Wait for instance to be running
    echo "⏳ Waiting for instance to be running..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID
    
    # Get public IP
    PUBLIC_IP=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    
    # Create S3 bucket and upload data
    BUCKET_NAME="$PROJECT_NAME-ec2-$(date +%s)"
    aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION || true
    
    echo "📤 Uploading data to S3..."
    aws s3 cp 2008.csv s3://$BUCKET_NAME/data/ 2>/dev/null || echo "⚠️  2008.csv not found"
    aws s3 cp 2008_sample.csv s3://$BUCKET_NAME/data/ 2>/dev/null || echo "⚠️  2008_sample.csv not found"
    aws s3 cp carriers.csv s3://$BUCKET_NAME/data/
    aws s3 cp airports.csv s3://$BUCKET_NAME/data/
    aws s3 cp AirTrafficProcessor.ipynb s3://$BUCKET_NAME/notebooks/
    
    echo "✅ EC2 deployment complete!"
    echo "🖥️  Instance ID: $INSTANCE_ID"
    echo "🌐 Public IP: $PUBLIC_IP"
    echo "🔗 Jupyter URL: http://$PUBLIC_IP:8888/?token=airtraffic-secure-token-2024"
    echo "🔑 SSH Access: ssh -i ~/.ssh/id_rsa ec2-user@$PUBLIC_IP"
    echo "📊 S3 Bucket: $BUCKET_NAME"
    echo "⏳ Wait 5-10 minutes for setup to complete"
    
    # Clean up temporary files
    rm -f user-data.sh
}

# Display help
show_help() {
    echo "Usage: $0 [DEPLOYMENT_TYPE] [AWS_REGION]"
    echo ""
    echo "DEPLOYMENT_TYPE:"
    echo "  emr       - Deploy on Amazon EMR (recommended for production)"
    echo "  sagemaker - Deploy on Amazon SageMaker (recommended for ML development)"
    echo "  ec2       - Deploy on self-managed EC2 instance"
    echo ""
    echo "AWS_REGION:"
    echo "  Default: us-east-1"
    echo "  Examples: us-west-2, eu-west-1, ap-southeast-1"
    echo ""
    echo "Examples:"
    echo "  $0 emr us-east-1"
    echo "  $0 sagemaker us-west-2"
    echo "  $0 ec2"
}

# Main execution
main() {
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        show_help
        exit 0
    fi
    
    check_prerequisites
    
    case $DEPLOYMENT_TYPE in
        "emr")
            deploy_emr
            ;;
        "sagemaker")
            deploy_sagemaker
            ;;
        "ec2")
            deploy_ec2
            ;;
        *)
            echo "❌ Unknown deployment type: $DEPLOYMENT_TYPE"
            show_help
            exit 1
            ;;
    esac
    
    echo ""
    echo "🎉 Deployment script completed!"
    echo "📚 For detailed instructions, see: aws-deployment/AMAZON-AWS-DEPLOYMENT.md"
}

main "$@"
