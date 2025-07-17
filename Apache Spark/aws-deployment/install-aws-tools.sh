#!/bin/bash

# Install AWS deployment tools on Debian/Ubuntu Linux

echo "🔧 Installing AWS Deployment Tools for AirTraffic Analysis"
echo "=========================================================="

# Update package manager
echo "📦 Updating package manager..."
sudo apt update

# Install prerequisites
echo "🔧 Installing prerequisites..."
sudo apt install -y curl unzip python3 python3-pip git software-properties-common gnupg

# Install AWS CLI v2
echo "☁️  Installing AWS CLI v2..."
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
    echo "✅ AWS CLI v2 installed"
else
    echo "✅ AWS CLI already installed: $(aws --version)"
fi

# Install Terraform
echo "🏗️  Installing Terraform..."
if ! command -v terraform &> /dev/null; then
    wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
    sudo apt update
    sudo apt install -y terraform
    echo "✅ Terraform installed: $(terraform version | head -1)"
else
    echo "✅ Terraform already installed: $(terraform version | head -1)"
fi

# Install Node.js and npm (for AWS CDK)
echo "📦 Installing Node.js and npm..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt install -y nodejs
    echo "✅ Node.js installed: $(node --version)"
else
    echo "✅ Node.js already installed: $(node --version)"
fi

# Install AWS CDK
echo "☁️  Installing AWS CDK..."
if ! command -v cdk &> /dev/null; then
    sudo npm install -g aws-cdk
    echo "✅ AWS CDK installed: $(cdk --version)"
else
    echo "✅ AWS CDK already installed: $(cdk --version)"
fi

# Install Python packages for AWS
echo "🐍 Installing Python packages for AWS..."
pip3 install --user boto3 awscli-plugin-endpoint

# Install Docker (for containerized deployments)
echo "🐳 Installing Docker..."
if ! command -v docker &> /dev/null; then
    sudo apt install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker $USER
    echo "✅ Docker installed. You may need to log out and back in for group changes."
else
    echo "✅ Docker already installed: $(docker --version)"
fi

# Install kubectl (for EKS deployments)
echo "☸️  Installing kubectl..."
if ! command -v kubectl &> /dev/null; then
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    chmod +x kubectl
    sudo mv kubectl /usr/local/bin/
    echo "✅ kubectl installed: $(kubectl version --client --short)"
else
    echo "✅ kubectl already installed: $(kubectl version --client --short)"
fi

# Install eksctl (for EKS cluster management)
echo "🔧 Installing eksctl..."
if ! command -v eksctl &> /dev/null; then
    curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
    sudo mv /tmp/eksctl /usr/local/bin
    echo "✅ eksctl installed: $(eksctl version)"
else
    echo "✅ eksctl already installed: $(eksctl version)"
fi

# Install AWS Session Manager plugin
echo "🔐 Installing AWS Session Manager plugin..."
if ! command -v session-manager-plugin &> /dev/null; then
    curl "https://s3.amazonaws.com/session-manager-downloads/plugin/latest/ubuntu_64bit/session-manager-plugin.deb" -o "session-manager-plugin.deb"
    sudo dpkg -i session-manager-plugin.deb
    rm session-manager-plugin.deb
    echo "✅ Session Manager plugin installed"
else
    echo "✅ Session Manager plugin already installed"
fi

# Create AWS credentials directory
echo "📁 Setting up AWS configuration..."
mkdir -p ~/.aws

# Create helper scripts
echo "📜 Creating helper scripts..."

# Create AWS configure helper
cat > ~/aws-configure-helper.sh << 'EOF'
#!/bin/bash
echo "🔐 AWS Configuration Helper"
echo "=========================="
echo "You'll need the following information:"
echo "1. AWS Access Key ID"
echo "2. AWS Secret Access Key"
echo "3. Default region (e.g., us-east-1)"
echo "4. Default output format (json recommended)"
echo ""
echo "If you don't have these, create them in AWS IAM:"
echo "https://console.aws.amazon.com/iam/home#/users"
echo ""
read -p "Press Enter to start AWS configuration..."
aws configure
echo "✅ AWS configuration complete!"
echo "Test with: aws sts get-caller-identity"
EOF

chmod +x ~/aws-configure-helper.sh

# Create deployment verification script
cat > ~/verify-aws-deployment-tools.sh << 'EOF'
#!/bin/bash
echo "🧪 Verifying AWS Deployment Tools"
echo "================================="

tools=(
    "aws:AWS CLI"
    "terraform:Terraform"
    "node:Node.js"
    "cdk:AWS CDK"
    "docker:Docker"
    "kubectl:Kubernetes CLI"
    "eksctl:EKS CLI"
)

for tool_info in "${tools[@]}"; do
    tool=$(echo $tool_info | cut -d: -f1)
    name=$(echo $tool_info | cut -d: -f2)
    
    if command -v $tool &> /dev/null; then
        version=$($tool --version 2>&1 | head -1)
        echo "✅ $name: $version"
    else
        echo "❌ $name: Not installed"
    fi
done

echo ""
echo "🔐 AWS Configuration Status:"
if aws sts get-caller-identity &> /dev/null; then
    aws sts get-caller-identity
    echo "✅ AWS credentials configured"
else
    echo "❌ AWS credentials not configured"
    echo "Run: ~/aws-configure-helper.sh"
fi

echo ""
echo "🐳 Docker Status:"
if docker ps &> /dev/null; then
    echo "✅ Docker daemon running"
else
    echo "⚠️  Docker daemon not running or user not in docker group"
    echo "Try: sudo systemctl start docker"
    echo "Or log out and back in if recently added to docker group"
fi
EOF

chmod +x ~/verify-aws-deployment-tools.sh

# Create quick deployment script
cat > ~/quick-deploy-airtraffic.sh << 'EOF'
#!/bin/bash
echo "🚀 Quick Deploy AirTraffic Analysis to AWS"
echo "=========================================="

# Check if in correct directory
if [ ! -f "AirTrafficProcessor.ipynb" ]; then
    echo "❌ Please run this script from the AirTraffic project directory"
    echo "Expected to find: AirTrafficProcessor.ipynb"
    exit 1
fi

echo "Select deployment option:"
echo "1. Amazon EMR (recommended for production)"
echo "2. Amazon SageMaker (recommended for ML development)"
echo "3. Self-managed EC2"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        ./aws-deployment/deploy-to-aws.sh emr
        ;;
    2)
        ./aws-deployment/deploy-to-aws.sh sagemaker
        ;;
    3)
        ./aws-deployment/deploy-to-aws.sh ec2
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac
EOF

chmod +x ~/quick-deploy-airtraffic.sh

echo ""
echo "🎉 Installation Complete!"
echo "========================"
echo ""
echo "📋 Next Steps:"
echo "1. Configure AWS credentials: ~/aws-configure-helper.sh"
echo "2. Verify installation: ~/verify-aws-deployment-tools.sh"
echo "3. Deploy to AWS: ~/quick-deploy-airtraffic.sh"
echo ""
echo "📚 Tools Installed:"
echo "✅ AWS CLI v2"
echo "✅ Terraform"
echo "✅ Node.js & npm"
echo "✅ AWS CDK"
echo "✅ Docker"
echo "✅ kubectl"
echo "✅ eksctl"
echo "✅ AWS Session Manager plugin"
echo ""
echo "🔧 Helper scripts created in home directory:"
echo "• ~/aws-configure-helper.sh - AWS credential setup"
echo "• ~/verify-aws-deployment-tools.sh - Verify installation"
echo "• ~/quick-deploy-airtraffic.sh - Quick deployment"
echo ""
echo "⚠️  Important Notes:"
echo "• You may need to log out and back in for Docker group changes"
echo "• Configure AWS credentials before deploying"
echo "• Check AWS service limits and quotas"
echo "• Review pricing for chosen deployment options"
