#!/bin/bash

# AWS Tools Installation Script for Linux
# This script installs all necessary AWS tools for the ETL platform

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Linux
check_os() {
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_error "This script is designed for Linux systems only."
        exit 1
    fi
    print_status "Operating System: Linux ✓"
}

# Install AWS CLI v2
install_aws_cli() {
    print_status "Installing AWS CLI v2..."
    
    if command -v aws &> /dev/null; then
        AWS_VERSION=$(aws --version 2>&1 | cut -d/ -f2 | cut -d' ' -f1)
        print_status "AWS CLI is already installed (version: $AWS_VERSION)"
        
        # Check if it's version 2
        if [[ $AWS_VERSION == 2.* ]]; then
            print_status "AWS CLI v2 is already installed ✓"
            return
        else
            print_warning "AWS CLI v1 detected. Upgrading to v2..."
        fi
    fi
    
    # Download and install AWS CLI v2
    cd /tmp
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip -q awscliv2.zip
    sudo ./aws/install --update
    
    # Verify installation
    if command -v aws &> /dev/null; then
        AWS_VERSION=$(aws --version 2>&1 | cut -d/ -f2 | cut -d' ' -f1)
        print_status "AWS CLI v2 installed successfully (version: $AWS_VERSION) ✓"
    else
        print_error "AWS CLI installation failed"
        exit 1
    fi
    
    # Cleanup
    rm -rf awscliv2.zip aws/
    cd - > /dev/null
}

# Install Docker
install_docker() {
    print_status "Installing Docker..."
    
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | sed 's/,//')
        print_status "Docker is already installed (version: $DOCKER_VERSION) ✓"
        return
    fi
    
    # Install Docker using the official script
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    # Start and enable Docker service
    sudo systemctl start docker
    sudo systemctl enable docker
    
    # Verify installation
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | sed 's/,//')
        print_status "Docker installed successfully (version: $DOCKER_VERSION) ✓"
        print_warning "Please log out and log back in for Docker group changes to take effect"
    else
        print_error "Docker installation failed"
        exit 1
    fi
    
    # Cleanup
    rm -f get-docker.sh
}

# Install Docker Compose
install_docker_compose() {
    print_status "Installing Docker Compose..."
    
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | sed 's/,//')
        print_status "Docker Compose is already installed (version: $COMPOSE_VERSION) ✓"
        return
    fi
    
    # Get latest release version
    COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
    
    # Download and install
    sudo curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Verify installation
    if command -v docker-compose &> /dev/null; then
        COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | sed 's/,//')
        print_status "Docker Compose installed successfully (version: $COMPOSE_VERSION) ✓"
    else
        print_error "Docker Compose installation failed"
        exit 1
    fi
}

# Install Terraform
install_terraform() {
    print_status "Installing Terraform..."
    
    if command -v terraform &> /dev/null; then
        TERRAFORM_VERSION=$(terraform --version | head -1 | cut -d' ' -f2)
        print_status "Terraform is already installed (version: $TERRAFORM_VERSION) ✓"
        return
    fi
    
    # Get latest release version
    TERRAFORM_VERSION=$(curl -s https://api.github.com/repos/hashicorp/terraform/releases/latest | grep 'tag_name' | cut -d\" -f4 | sed 's/v//')
    
    # Download and install
    cd /tmp
    wget "https://releases.hashicorp.com/terraform/${TERRAFORM_VERSION}/terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
    unzip -q "terraform_${TERRAFORM_VERSION}_linux_amd64.zip"
    sudo mv terraform /usr/local/bin/
    
    # Verify installation
    if command -v terraform &> /dev/null; then
        TERRAFORM_VERSION=$(terraform --version | head -1 | cut -d' ' -f2)
        print_status "Terraform installed successfully (version: $TERRAFORM_VERSION) ✓"
    else
        print_error "Terraform installation failed"
        exit 1
    fi
    
    # Cleanup
    rm -f terraform_*_linux_amd64.zip
    cd - > /dev/null
}

# Install kubectl
install_kubectl() {
    print_status "Installing kubectl..."
    
    if command -v kubectl &> /dev/null; then
        KUBECTL_VERSION=$(kubectl version --client --short 2>/dev/null | cut -d' ' -f3)
        print_status "kubectl is already installed (version: $KUBECTL_VERSION) ✓"
        return
    fi
    
    # Download and install kubectl
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
    
    # Verify installation
    if command -v kubectl &> /dev/null; then
        KUBECTL_VERSION=$(kubectl version --client --short 2>/dev/null | cut -d' ' -f3)
        print_status "kubectl installed successfully (version: $KUBECTL_VERSION) ✓"
    else
        print_error "kubectl installation failed"
        exit 1
    fi
    
    # Cleanup
    rm -f kubectl
}

# Install AWS SAM CLI
install_sam_cli() {
    print_status "Installing AWS SAM CLI..."
    
    if command -v sam &> /dev/null; then
        SAM_VERSION=$(sam --version | cut -d' ' -f4)
        print_status "AWS SAM CLI is already installed (version: $SAM_VERSION) ✓"
        return
    fi
    
    # Download and install SAM CLI
    cd /tmp
    wget https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-x86_64.zip
    unzip -q aws-sam-cli-linux-x86_64.zip -d sam-installation
    sudo ./sam-installation/install
    
    # Verify installation
    if command -v sam &> /dev/null; then
        SAM_VERSION=$(sam --version | cut -d' ' -f4)
        print_status "AWS SAM CLI installed successfully (version: $SAM_VERSION) ✓"
    else
        print_error "AWS SAM CLI installation failed"
        exit 1
    fi
    
    # Cleanup
    rm -rf aws-sam-cli-linux-x86_64.zip sam-installation/
    cd - > /dev/null
}

# Install eksctl
install_eksctl() {
    print_status "Installing eksctl..."
    
    if command -v eksctl &> /dev/null; then
        EKSCTL_VERSION=$(eksctl version)
        print_status "eksctl is already installed (version: $EKSCTL_VERSION) ✓"
        return
    fi
    
    # Download and install eksctl
    curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
    sudo mv /tmp/eksctl /usr/local/bin
    
    # Verify installation
    if command -v eksctl &> /dev/null; then
        EKSCTL_VERSION=$(eksctl version)
        print_status "eksctl installed successfully (version: $EKSCTL_VERSION) ✓"
    else
        print_error "eksctl installation failed"
        exit 1
    fi
}

# Install Java 17 (required for Spring Boot)
install_java() {
    print_status "Installing Java 17..."
    
    if command -v java &> /dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | head -1 | cut -d'"' -f2)
        print_status "Java is already installed (version: $JAVA_VERSION)"
        
        # Check if it's Java 17 or higher
        JAVA_MAJOR=$(echo $JAVA_VERSION | cut -d'.' -f1)
        if [ "$JAVA_MAJOR" -ge 17 ]; then
            print_status "Java 17+ is already installed ✓"
            return
        else
            print_warning "Java version is less than 17. Installing Java 17..."
        fi
    fi
    
    # Install Java 17
    sudo apt-get update
    sudo apt-get install -y openjdk-17-jdk
    
    # Set JAVA_HOME
    JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::")
    echo "export JAVA_HOME=$JAVA_HOME" >> ~/.bashrc
    
    # Verify installation
    if command -v java &> /dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | head -1 | cut -d'"' -f2)
        print_status "Java 17 installed successfully (version: $JAVA_VERSION) ✓"
    else
        print_error "Java installation failed"
        exit 1
    fi
}

# Install Maven
install_maven() {
    print_status "Installing Maven..."
    
    if command -v mvn &> /dev/null; then
        MAVEN_VERSION=$(mvn -version | head -1 | cut -d' ' -f3)
        print_status "Maven is already installed (version: $MAVEN_VERSION) ✓"
        return
    fi
    
    # Install Maven
    sudo apt-get update
    sudo apt-get install -y maven
    
    # Verify installation
    if command -v mvn &> /dev/null; then
        MAVEN_VERSION=$(mvn -version | head -1 | cut -d' ' -f3)
        print_status "Maven installed successfully (version: $MAVEN_VERSION) ✓"
    else
        print_error "Maven installation failed"
        exit 1
    fi
}

# Install Node.js (for React dashboard)
install_nodejs() {
    print_status "Installing Node.js..."
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Node.js is already installed (version: $NODE_VERSION) ✓"
        return
    fi
    
    # Install Node.js using NodeSource repository
    curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
    sudo apt-get install -y nodejs
    
    # Verify installation
    if command -v node &> /dev/null && command -v npm &> /dev/null; then
        NODE_VERSION=$(node --version)
        NPM_VERSION=$(npm --version)
        print_status "Node.js installed successfully (version: $NODE_VERSION, npm: $NPM_VERSION) ✓"
    else
        print_error "Node.js installation failed"
        exit 1
    fi
}

# Configure AWS CLI
configure_aws() {
    print_status "AWS CLI Configuration"
    print_warning "Please configure your AWS credentials:"
    
    echo "You can get your AWS credentials from:"
    echo "1. AWS Console > IAM > Users > Your User > Security credentials"
    echo "2. Or from your organization's AWS administrator"
    echo ""
    
    read -p "Do you want to configure AWS CLI now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        aws configure
        print_status "AWS CLI configured successfully ✓"
    else
        print_warning "AWS CLI configuration skipped. You can run 'aws configure' later."
    fi
}

# Display installation summary
show_summary() {
    print_status "Installation Summary"
    echo "================================"
    
    echo "AWS Tools:"
    command -v aws >/dev/null && echo "  ✓ AWS CLI: $(aws --version 2>&1 | cut -d/ -f2 | cut -d' ' -f1)" || echo "  ✗ AWS CLI: Not installed"
    command -v sam >/dev/null && echo "  ✓ AWS SAM CLI: $(sam --version | cut -d' ' -f4)" || echo "  ✗ AWS SAM CLI: Not installed"
    
    echo ""
    echo "Container Tools:"
    command -v docker >/dev/null && echo "  ✓ Docker: $(docker --version | cut -d' ' -f3 | sed 's/,//')" || echo "  ✗ Docker: Not installed"
    command -v docker-compose >/dev/null && echo "  ✓ Docker Compose: $(docker-compose --version | cut -d' ' -f3 | sed 's/,//')" || echo "  ✗ Docker Compose: Not installed"
    
    echo ""
    echo "Infrastructure Tools:"
    command -v terraform >/dev/null && echo "  ✓ Terraform: $(terraform --version | head -1 | cut -d' ' -f2)" || echo "  ✗ Terraform: Not installed"
    command -v kubectl >/dev/null && echo "  ✓ kubectl: $(kubectl version --client --short 2>/dev/null | cut -d' ' -f3)" || echo "  ✗ kubectl: Not installed"
    command -v eksctl >/dev/null && echo "  ✓ eksctl: $(eksctl version)" || echo "  ✗ eksctl: Not installed"
    
    echo ""
    echo "Development Tools:"
    command -v java >/dev/null && echo "  ✓ Java: $(java -version 2>&1 | head -1 | cut -d'"' -f2)" || echo "  ✗ Java: Not installed"
    command -v mvn >/dev/null && echo "  ✓ Maven: $(mvn -version | head -1 | cut -d' ' -f3)" || echo "  ✗ Maven: Not installed"
    command -v node >/dev/null && echo "  ✓ Node.js: $(node --version)" || echo "  ✗ Node.js: Not installed"
    command -v npm >/dev/null && echo "  ✓ npm: $(npm --version)" || echo "  ✗ npm: Not installed"
    
    echo ""
    echo "================================"
    print_status "Installation completed!"
    echo ""
    print_warning "Important notes:"
    echo "1. If Docker was installed, please log out and log back in"
    echo "2. Run 'aws configure' to set up your AWS credentials"
    echo "3. Source your ~/.bashrc to update environment variables: source ~/.bashrc"
}

# Main installation function
main() {
    print_status "Starting AWS ETL Platform tools installation..."
    
    check_os
    
    # Update package list
    sudo apt-get update
    
    # Install essential tools
    install_java
    install_maven
    install_nodejs
    install_aws_cli
    install_docker
    install_docker_compose
    install_terraform
    install_kubectl
    install_sam_cli
    install_eksctl
    
    # Configure AWS
    configure_aws
    
    # Show summary
    show_summary
}

# Run main function
main "$@"
