#!/bin/bash

# Amazon Bedrock BERT fine-tuning setup script
# This script automates the setup process for fine-tuning BERT models with Amazon Bedrock

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DEFAULT_REGION="us-east-1"
DEFAULT_ENVIRONMENT="dev"
DEFAULT_PROJECT_NAME="bert-fine-tuning"
DEFAULT_BUDGET=100

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check AWS CLI configuration
check_aws_cli() {
    print_status "Checking AWS CLI configuration..."
    
    if ! command_exists aws; then
        print_error "AWS CLI is not installed. Please install it first."
        echo "Installation guide: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        exit 1
    fi
    
    # Check if AWS is configured
    if ! aws sts get-caller-identity >/dev/null 2>&1; then
        print_error "AWS CLI is not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    CURRENT_REGION=$(aws configure get region || echo $DEFAULT_REGION)
    
    print_success "AWS CLI configured for account: $ACCOUNT_ID in region: $CURRENT_REGION"
}

# Function to setup Python environment
setup_python_environment() {
    print_status "Setting up Python virtual environment..."
    
    # Check if Python 3 is available
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.8 or later."
        exit 1
    fi
    
    # Create virtual environment
    if [ ! -d "bedrock_env" ]; then
        python3 -m venv bedrock_env
        print_success "Created virtual environment: bedrock_env"
    else
        print_warning "Virtual environment 'bedrock_env' already exists"
    fi
    
    # Activate virtual environment
    source bedrock_env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements-bedrock.txt
    
    print_success "Python environment setup complete"
}

# Function to setup Terraform
setup_terraform() {
    print_status "Checking Terraform installation..."
    
    if ! command_exists terraform; then
        print_warning "Terraform not found. Installing Terraform..."
        
        # Download and install Terraform (Linux)
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
            echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
            sudo apt update && sudo apt install terraform
        else
            print_error "Please install Terraform manually: https://learn.hashicorp.com/tutorials/terraform/install-cli"
            exit 1
        fi
    fi
    
    terraform version
    print_success "Terraform is available"
}

# Function to deploy infrastructure
deploy_infrastructure() {
    print_status "Deploying AWS infrastructure with Terraform..."
    
    cd infrastructure
    
    # Initialize Terraform
    terraform init
    
    # Create terraform.tfvars if it doesn't exist
    if [ ! -f "terraform.tfvars" ]; then
        cat > terraform.tfvars << EOF
aws_region = "$AWS_REGION"
environment = "$ENVIRONMENT"
project_name = "$PROJECT_NAME"
max_budget_amount = $BUDGET
notification_email = "$EMAIL"
EOF
        print_success "Created terraform.tfvars with default values"
    fi
    
    # Plan deployment
    terraform plan
    
    # Ask for confirmation
    read -p "Do you want to apply these changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        terraform apply -auto-approve
        print_success "Infrastructure deployed successfully"
        
        # Extract outputs
        S3_BUCKET=$(terraform output -raw s3_bucket_name)
        ROLE_ARN=$(terraform output -raw bedrock_execution_role_arn)
        
        print_success "S3 Bucket: $S3_BUCKET"
        print_success "Execution Role ARN: $ROLE_ARN"
        
        # Save configuration
        mkdir -p ../config
        cat > ../config/bedrock_config.json << EOF
{
    "s3_bucket": "$S3_BUCKET",
    "execution_role_arn": "$ROLE_ARN",
    "region": "$AWS_REGION",
    "environment": "$ENVIRONMENT"
}
EOF
        print_success "Configuration saved to config/bedrock_config.json"
    else
        print_warning "Infrastructure deployment cancelled"
    fi
    
    cd ..
}

# Function to test setup
test_setup() {
    print_status "Testing Amazon Bedrock setup..."
    
    # Activate virtual environment
    source bedrock_env/bin/activate
    
    # Test AWS connectivity
    aws bedrock list-foundation-models --region $AWS_REGION > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_success "Amazon Bedrock access confirmed"
    else
        print_warning "Amazon Bedrock access may be limited. Check your permissions."
    fi
    
    # Test Python script
    python src/bedrock_bert_fine_tuning.py --validate-only
    if [ $? -eq 0 ]; then
        print_success "Python integration test passed"
    else
        print_warning "Python integration test failed. Check your configuration."
    fi
}

# Function to create sample configuration
create_sample_config() {
    print_status "Creating sample configuration files..."
    
    mkdir -p config data
    
    # Create sample training data
    cat > data/sample_training_data.json << 'EOF'
[
    {"text": "I love this product, it's amazing!", "label": 1},
    {"text": "This is the worst service I've ever experienced.", "label": 0},
    {"text": "The quality is excellent and delivery was fast.", "label": 1},
    {"text": "Terrible customer support, very disappointing.", "label": 0},
    {"text": "Great value for money, highly recommended!", "label": 1},
    {"text": "The product broke after one day of use.", "label": 0},
    {"text": "Outstanding customer service and fast shipping.", "label": 1},
    {"text": "Poor quality materials and overpriced.", "label": 0}
]
EOF
    
    # Create environment configuration
    cat > .env.example << EOF
# AWS Configuration
AWS_REGION=$AWS_REGION
AWS_PROFILE=default

# Bedrock Configuration
BEDROCK_REGION=$AWS_REGION
S3_BUCKET=your-bedrock-bucket
EXECUTION_ROLE_ARN=your-role-arn

# Training Configuration
MAX_TRAINING_COST=100
TRAINING_TIMEOUT_HOURS=24

# Monitoring
ENABLE_CLOUDWATCH=true
LOG_LEVEL=INFO
EOF
    
    print_success "Sample configuration files created"
}

# Function to display usage instructions
show_usage_instructions() {
    echo
    echo "==============================================="
    echo "SETUP COMPLETE! Next Steps:"
    echo "==============================================="
    echo
    echo "1. Activate the Python environment:"
    echo "   source bedrock_env/bin/activate"
    echo
    echo "2. Update configuration files:"
    echo "   - Edit config/bedrock_config.json with your specific settings"
    echo "   - Copy .env.example to .env and update values"
    echo
    echo "3. Test the setup:"
    echo "   python src/bedrock_bert_fine_tuning.py"
    echo
    echo "4. Start fine-tuning:"
    echo "   python src/bedrock_bert_fine_tuning.py --train"
    echo
    echo "5. Monitor costs in AWS Console:"
    echo "   - Go to AWS Billing Dashboard"
    echo "   - Check Budget alerts"
    echo "   - Monitor Bedrock usage"
    echo
    echo "==============================================="
    echo "Useful Commands:"
    echo "==============================================="
    echo "- View infrastructure: cd infrastructure && terraform show"
    echo "- Update infrastructure: cd infrastructure && terraform apply"
    echo "- Destroy infrastructure: cd infrastructure && terraform destroy"
    echo "- View logs: aws logs describe-log-groups --log-group-name-prefix '/aws/bedrock'"
    echo "- Check costs: aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-12-31 --granularity MONTHLY --metrics BlendedCost"
    echo
}

# Main setup function
main() {
    echo "=============================================="
    echo "Amazon Bedrock BERT Fine-tuning Setup Script"
    echo "=============================================="
    echo
    
    # Get user inputs
    read -p "AWS Region [$DEFAULT_REGION]: " AWS_REGION
    AWS_REGION=${AWS_REGION:-$DEFAULT_REGION}
    
    read -p "Environment [$DEFAULT_ENVIRONMENT]: " ENVIRONMENT
    ENVIRONMENT=${ENVIRONMENT:-$DEFAULT_ENVIRONMENT}
    
    read -p "Project Name [$DEFAULT_PROJECT_NAME]: " PROJECT_NAME
    PROJECT_NAME=${PROJECT_NAME:-$DEFAULT_PROJECT_NAME}
    
    read -p "Budget Limit (USD) [$DEFAULT_BUDGET]: " BUDGET
    BUDGET=${BUDGET:-$DEFAULT_BUDGET}
    
    read -p "Notification Email: " EMAIL
    if [ -z "$EMAIL" ]; then
        print_error "Email is required for notifications"
        exit 1
    fi
    
    echo
    print_status "Starting setup with the following configuration:"
    echo "  AWS Region: $AWS_REGION"
    echo "  Environment: $ENVIRONMENT"
    echo "  Project Name: $PROJECT_NAME"
    echo "  Budget: \$$BUDGET"
    echo "  Email: $EMAIL"
    echo
    
    # Setup steps
    check_aws_cli
    setup_python_environment
    setup_terraform
    create_sample_config
    deploy_infrastructure
    test_setup
    show_usage_instructions
    
    print_success "Setup completed successfully!"
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
