# AWS tools and local environment setup

This document provides comprehensive guidance on setting up the local development environment and installing necessary AWS tools for deploying the Weather ETL platform to Amazon AWS.

## Prerequisites

- **Operating system**: Linux (Ubuntu 20.04+ recommended)
- **Hardware**: Minimum 8GB RAM, 20GB free disk space
- **Network**: Stable internet connection for downloading tools and AWS services

## AWS tools for local development

### 1. AWS CLI v2 (Required)

The AWS Command Line Interface is essential for managing AWS resources and deploying applications.

**Installation:**
```bash
# Download and install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Verify installation
aws --version
```

**Configuration:**
```bash
# Configure AWS credentials
aws configure

# Configure additional profiles (optional)
aws configure --profile production
```

**Configuration:**
- AWS Access Key ID
- AWS Secret Access Key
- Default region (e.g., us-east-1)
- Default output format (JSON recommended)

### 2. Docker & Docker Compose (Required)

Docker is essential for containerizing microservices and local development.

**Installation:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

### 3. Terraform (Highly recommended)

Infrastructure as Code tool for managing AWS resources.

**Installation:**
```bash
# Add HashiCorp GPG key
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -

# Add HashiCorp repository
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"

# Install Terraform
sudo apt-get update && sudo apt-get install terraform

# Verify installation
terraform --version
```

### 4. AWS SAM CLI (Recommended)

For serverless application development and testing.

**Installation:**
```bash
# Download AWS SAM CLI
wget https://github.com/aws/aws-sam-cli/releases/latest/download/aws-sam-cli-linux-x86_64.zip
unzip aws-sam-cli-linux-x86_64.zip -d sam-installation
sudo ./sam-installation/install

# Verify installation
sam --version
```

### 5. kubectl (Recommended for EKS)

Kubernetes command-line tool for container orchestration.

**Installation:**
```bash
# Download kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Install kubectl
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Verify installation
kubectl version --client
```

### 6. The eksctl (Optional for EKS)

Command-line tool for creating and managing EKS clusters.

**Installation:**
```bash
# Download and install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Verify installation
eksctl version
```

## Development environment setup

### 1. Java Development Kit (JDK) 17

**Installation:**
```bash
# Install OpenJDK 17
sudo apt update
sudo apt install openjdk-17-jdk

# Set JAVA_HOME
echo 'export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc

# Verify installation
java -version
javac -version
```

### 2. Apache Maven

**Installation:**
```bash
# Install Maven
sudo apt install maven

# Verify installation
mvn -version
```

### 3. Node.js and npm (for React Dashboard)

**Installation:**
```bash
# Install Node.js 18.x
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version
npm --version
```

### 4. Git

**Installation:**
```bash
# Install Git
sudo apt install git

# Configure Git
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify installation
git --version
```

## Infrastructure as Code (IaC)

### Why use Infrastructure as Code?

1. **Version Control**: Track infrastructure changes over time
2. **Reproducibility**: Consistent deployments across environments
3. **Automation**: Automated provisioning and updates
4. **Documentation**: Infrastructure becomes self-documenting
5. **Cost Management**: Better resource tracking and optimization
6. **Disaster Recovery**: Quick environment recreation

### Terraform vs CloudFormation

#### Terraform (Recommended)

**Advantages:**
- Multi-cloud support (AWS, Azure, GCP)
- Rich ecosystem of providers and modules
- More readable HCL (HashiCorp Configuration Language) syntax
- Better state management with remote backends
- Excellent plan/apply workflow
- Strong community support

**Use Cases:**
- Multi-cloud deployments
- Complex infrastructure with multiple providers
- Teams familiar with HashiCorp tools
- Need for advanced state management

**Example Terraform structure:**
```
infrastructure/terraform/
├── main.tf              # Main resources
├── variables.tf         # Input variables
├── outputs.tf          # Output values
├── networking.tf       # VPC, subnets, security groups
├── iam.tf             # IAM roles and policies
├── storage.tf         # S3 buckets
├── compute.tf         # ECS, EC2 instances
├── data.tf            # Glue, Athena resources
├── monitoring.tf      # CloudWatch, X-Ray
└── environments/
    ├── dev.tfvars
    ├── staging.tfvars
    └── prod.tfvars
```

#### CloudFormation (Alternative)

**Advantages:**
- Native AWS integration
- No additional tools required
- Tight integration with AWS services
- AWS Support included
- Built-in rollback capabilities

**Use Cases:**
- AWS-only deployments
- Teams preferring native AWS tools
- Organizations with AWS-first strategy
- Simple to moderate complexity infrastructure

**Example CloudFormation structure:**
```
infrastructure/cloudformation/
├── master-stack.yaml
├── networking-stack.yaml
├── iam-stack.yaml
├── storage-stack.yaml
├── compute-stack.yaml
├── data-stack.yaml
└── monitoring-stack.yaml
```

### Recommendation: Terraform

For this ETL platform, we recommend **Terraform** for the following reasons:

1. **State management**: Remote state with locking prevents conflicts
2. **Modular design**: Reusable modules for different environments
3. **Ecosystem**: Many pre-built modules available
4. **Planning**: Preview changes before applying
5. **Flexibility**: Can integrate with other cloud providers if needed

## AWS services in the platform

### Core Data services

1. **Amazon S3**: Data lake storage for raw and processed data
2. **AWS Glue**: ETL job orchestration and data catalog
3. **Amazon Athena**: Serverless query engine for data analysis
4. **Amazon MSK**: Managed Apache Kafka for streaming data

### Compute services

1. **Amazon ECS**: Container orchestration for microservices
2. **AWS Fargate**: Serverless compute for containers
3. **Amazon Kinesis Data Analytics**: Apache Flink stream processing

### Networking & security

1. **Amazon VPC**: Virtual private cloud with public/private subnets
2. **AWS IAM**: Identity and access management
3. **AWS Secrets Manager**: Secure storage for credentials
4. **AWS Parameter Store**: Configuration management

### Monitoring & logging

1. **Amazon CloudWatch**: Metrics, logs, and alarms
2. **AWS X-Ray**: Distributed tracing
3. **CloudWatch Logs**: Centralized logging

### Deployment

1. **Amazon ECR**: Container registry
2. **AWS CodePipeline**: CI/CD pipeline (optional)
3. **Amazon CloudFormation**: Alternative IaC (optional)

## Local development workflow

### 1. Initial setup

```bash
# Clone the repository
git clone <repository-url>
cd weather-etl-platform

# Make scripts executable
chmod +x scripts/*.sh

# Install AWS tools (automated script)
./scripts/install-aws-tools.sh

# Configure AWS credentials
aws configure
```

### 2. Local development with Docker

```bash
# Start local development environment
docker-compose up -d

# Build and test microservices
mvn clean package

# Build Docker images
docker-compose build

# View logs
docker-compose logs -f data-ingestion-service
```

### 3. Deploy to AWS

```bash
# Initialize Terraform
cd infrastructure/terraform
terraform init

# Plan deployment
terraform plan -var-file="environments/dev.tfvars"

# Deploy infrastructure
terraform apply -var-file="environments/dev.tfvars"

# Deploy applications
./scripts/deploy-to-aws.sh
```

## Cost optimization

### 1. Development environment

- Use **AWS Free Tier** resources where possible
- **Stop/Start services** when not in use
- Use **Spot Instances** for non-critical workloads
- **Monitor costs** with AWS Cost Explorer

### 2. Production environment

- Use **Reserved Instances** for predictable workloads
- Implement **Auto Scaling** for variable workloads
- Use **S3 Intelligent Tiering** for cost-effective storage
- Set up **billing alerts** for cost monitoring

### 3. Resource tagging

Implement consistent tagging strategy:
```hcl
default_tags {
  tags = {
    Project     = "weather-etl"
    Environment = var.environment
    Owner       = "data-team"
    ManagedBy   = "terraform"
    CostCenter  = "engineering"
  }
}
```

## Security

### 1. AWS account security

- Enable **AWS CloudTrail** for audit logging
- Use **AWS Config** for compliance monitoring
- Enable **GuardDuty** for threat detection
- Implement **multi-factor authentication** (MFA)

### 2. Application security

- Use **least privilege** IAM policies
- Store secrets in **AWS Secrets Manager**
- Encrypt data **at REST and in transit**
- Implement **VPC security groups** properly

### 3. Development security

- Never commit **AWS credentials** to version control
- Use **IAM roles** instead of long-term credentials
- Regularly **rotate access keys**
- Use **AWS CLI profiles** for different environments

## Troubleshooting issues

### AWS CLI issues

```bash
# Check AWS credentials
aws sts get-caller-identity

# Debug AWS CLI calls
aws s3 ls --debug

# Use different profile
aws s3 ls --profile production
```

### Docker issues

```bash
# Check Docker daemon
sudo systemctl status docker

# View Docker logs
docker logs <container-name>

# Clean up Docker resources
docker system prune -a
```

### Terraform issues

```bash
# Initialize Terraform
terraform init

# Validate configuration
terraform validate

# Check state
terraform state list

# Refresh state
terraform refresh
```

## Monitoring and observability

### Local development

1. **Docker Compose Logs**: `docker-compose logs -f`
2. **Application Metrics**: Access via actuator endpoints
3. **Local Kafka UI**: Monitor message flow
4. **MinIO Console**: S3-compatible storage interface

### AWS production

1. **CloudWatch Dashboards**: Custom metrics visualization
2. **X-Ray Tracing**: Distributed request tracing
3. **CloudWatch Alarms**: Automated alerting
4. **Cost and Usage Reports**: Cost optimization insights

## Next

1. **Install required tools**: Run the installation script
2. **Configure AWS credentials**: Set up AWS CLI access
3. **Clone repository**: Get the project code
4. **Testing**: Start with Docker Compose
5. **Infrastructure deployment**: Use Terraform for AWS resources
6. **Application deployment**: Deploy microservices to ECS
7. **Monitoring setup**: Configure CloudWatch and alarms

## References

- [AWS CLI User Guide](https://docs.aws.amazon.com/cli/latest/userguide/)
- [Terraform AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Docker Documentation](https://docs.docker.com/)
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [Spring Boot on AWS](https://spring.io/guides/gs/spring-boot-docker/)

For immediate assistance, run the automated installation script:
```bash
./scripts/install-aws-tools.sh
```
