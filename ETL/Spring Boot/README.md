# AWS ETL microservices

A microservices architecture for processing weather station data using Spring Boot, deployed on AWS with real-time ETL capabilities.

## Architecture

This solution implements a scalable ETL pipeline processing temperature measurements from 5 weather stations every hour within a 100km radius.

### Components

- **Data Ingestion Service**: MQTT message broker integration
- **ETL Processing Service**: AWS Glue for batch ETL operations
- **Stream Processing Service**: Apache Flink for real-time data processing
- **Query Service**: Amazon Athena with Apache Iceberg for schema evolution
- **Dashboard API**: REST API for frontend integration
- **React Dashboard**: Real-time visualization web application

### AWS services

- **AWS Glue**: ETL job orchestration
- **Amazon Athena**: Serverless query engine
- **Apache Iceberg**: Table format with schema evolution
- **Apache Flink**: Stream processing
- **Amazon MSK**: Managed Kafka for streaming
- **Amazon S3**: Data lake storage
- **Amazon ECS/EKS**: Container orchestration
- **Amazon API Gateway**: API management
- **AWS CloudFormation/Terraform**: Infrastructure as Code

![alt text](https://github.com/jylhakos/Data-Analysis-and-Visualizations/blob/main/ETL/Spring%20Boot/DATA-FLOW.png?raw=true)

## Prerequisites

### Local Development Tools

1. **Java Development Kit (JDK) 17+**
2. **Maven 3.8+**
3. **Docker & Docker Compose**
4. **AWS CLI v2**
5. **Terraform** (optional, for IaC)
6. **Node.js 18+** (for React dashboard)

### AWS CLI Installation & Configuration

```bash
# Install AWS CLI v2 on Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
```

### Required AWS Tools

1. **AWS CLI v2**: Command-line interface
2. **AWS CDK** (optional): Cloud Development Kit
3. **AWS SAM CLI**: Serverless Application Model
4. **Docker**: For containerization
5. **kubectl**: Kubernetes command-line tool

## Quick Start

1. **Clone and Setup**:
   ```bash
   cd /path/to/project
   mvn clean install
   ```

2. **Build Docker Images**:
   ```bash
   docker-compose build
   ```

3. **Deploy Infrastructure**:
   ```bash
   cd infrastructure/terraform
   terraform init
   terraform plan
   terraform apply
   ```

4. **Deploy Microservices**:
   ```bash
   ./scripts/deploy-to-aws.sh
   ```

5. **Start React Dashboard**:
   ```bash
   cd frontend/dashboard
   npm install
   npm start
   ```

## Essential IAM Roles

### 1. ETL Processing Role
- **Purpose**: Execute AWS Glue jobs
- **Permissions**: 
  - `AWSGlueServiceRole`
  - S3 read/write access
  - CloudWatch logs access

### 2. Stream Processing Role
- **Purpose**: Apache Flink job execution
- **Permissions**:
  - Kinesis Data Analytics access
  - MSK cluster access
  - S3 read/write access

### 3. Query Service Role
- **Purpose**: Amazon Athena query execution
- **Permissions**:
  - Athena query execution
  - S3 data lake access
  - Glue catalog access

### 4. Application Runtime Role
- **Purpose**: ECS/EKS task execution
- **Permissions**:
  - ECR image pull
  - CloudWatch logs
  - Parameter Store access

## Infrastructure as Code (IaC)

### Benefits of Using IaC

1. **Version Control**: Track infrastructure changes
2. **Reproducibility**: Consistent deployments across environments
3. **Automation**: Automated provisioning and updates
4. **Cost Management**: Better resource tracking and optimization

### Terraform vs CloudFormation

**Recommended: Terraform**
- Multi-cloud support
- Better state management
- Rich ecosystem and modules
- More readable HCL syntax

**CloudFormation Alternative**
- Native AWS integration
- No additional tools required
- Tight integration with AWS services

## Project Structure

```
├── microservices/
│   ├── data-ingestion-service/
│   ├── etl-processing-service/
│   ├── stream-processing-service/
│   ├── query-service/
│   └── dashboard-api-service/
├── infrastructure/
│   ├── terraform/
│   └── cloudformation/
├── frontend/
│   └── dashboard/
├── docker/
├── scripts/
└── docs/
```

## Development Workflow

1. **Local Development**: Use Docker Compose for local testing
2. **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
3. **Monitoring**: CloudWatch, X-Ray for observability
4. **Security**: IAM least privilege, VPC isolation

## Next Steps

1. Configure AWS credentials
2. Review and customize Terraform configurations
3. Deploy infrastructure components
4. Build and deploy microservices
5. Configure monitoring and alerting

For detailed implementation guides, see the `docs/` directory.
