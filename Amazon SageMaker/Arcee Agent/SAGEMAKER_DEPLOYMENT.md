# Amazon SageMaker deployment

## Implementation

This document summarizes the complete AWS SageMaker deployment solution for the Arcee Agent function calling system.

## Features

### 1. Core infrastructure
- **SageMaker inference client** (`sagemaker_inference.py`)
  - Real-time and batch inference support
  - Function calling integration
  - Error handling and metrics
  - Endpoint validation and monitoring

- **FastAPI production server** (`api_server.py`)
  - RESTful API endpoints
  - Health checks and monitoring
  - Integration with SageMaker endpoints
  - CORS and security middleware

### 2. Deployment automation
- **Deployment script** (`deploy_to_sagemaker.sh`)
  - One-command deployment pipeline
  - IAM role setup automation
  - Docker image building and pushing
  - Training job management
  - Endpoint deployment
  - Cost optimization

- **IAM setup script** (`scripts/setup_iam_roles.sh`)
  - Automated role and policy creation
  - Least-privilege permissions
  - SageMaker execution roles

### 3. Training and fine-tuning
- **SageMaker training script** (`scripts/sagemaker_training.py`)
  - LoRA fine-tuning support
  - Quantization for efficiency
  - Hyperparameter configuration
  - Cost-optimized instance selection

- **Training Docker image** (`docker/training/Dockerfile`)
  - HuggingFace Transformers base
  - GPU optimization
  - Fine-tuning dependencies

### 4. Model deployment
- **Deployment script** (`scripts/sagemaker_deployment.py`)
  - Model creation and versioning
  - Endpoint configuration
  - Auto-scaling support
  - Blue-green deployments

- **Inference Docker image** (`docker/inference/Dockerfile`)
  - Production-ready inference
  - SageMaker protocol compliance
  - Performance optimization

- **Inference handler** (`docker/inference/inference.py`)
  - SageMaker inference protocol
  - Function calling support
  - Model loading and caching
  - Health checks

### 5. Monitoring and cost management
- **Monitoring** (`scripts/monitor_deployment.py`)
  - Real-time metrics collection
  - Cost analysis and alerts
  - Performance monitoring
  - Resource cleanup automation

- **Cost optimization**
  - Automatic resource cleanup
  - Budget alerts and thresholds
  - Instance type recommendations
  - Auto-scaling configuration

### 6. Production
- **Docker Compose** (`docker-compose.yaml`)
  - Multi-service orchestration
  - Development and production configs
  - Monitoring stack integration
  - Redis caching support

- **Production Dockerfile** (`Dockerfile`)
  - Security best practices
  - Non-root user execution
  - Health checks
  - Multi-stage builds

### 7. Configuration management
- **Environment configuration** (`.env.example`)
  - All deployment parameters
  - Cost management settings
  - Security configurations
  - Development/production modes

## Start commands

### Deployment
```bash
# Set your S3 bucket
export S3_BUCKET="your-arcee-agent-bucket"

# Run complete deployment pipeline
./deploy_to_sagemaker.sh --all

# Start API service
./deploy_to_sagemaker.sh --start-api
```

### Step-by-step
```bash
# 1. Setup IAM roles
./deploy_to_sagemaker.sh --setup-iam

# 2. Upload dataset
./deploy_to_sagemaker.sh --upload-data

# 3. Build Docker images
./deploy_to_sagemaker.sh --build-images

# 4. Train model
./deploy_to_sagemaker.sh --train-model

# 5. Deploy endpoint
./deploy_to_sagemaker.sh --deploy-endpoint

# 6. Start API
./deploy_to_sagemaker.sh --start-api
```

### Monitoring and management
```bash
# Daily monitoring report
python scripts/monitor_deployment.py --daily-report

# Cost analysis
python scripts/monitor_deployment.py --cost-analysis --endpoint-name arcee-agent-endpoint

# Resource cleanup
python scripts/monitor_deployment.py --cleanup --dry-run
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client App    │───▶│   FastAPI API    │───▶│  SageMaker      │
│   (CURL/REST)   │    │   (Port 8000)    │    │  Endpoint       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   CloudWatch     │    │  Fine-tuned     │
                       │   Monitoring     │    │  Arcee Agent    │
                       └──────────────────┘    └─────────────────┘
```

## Cost estimation

### Training (One-time)
- **Instance**: ml.g4dn.xlarge (~$0.53/hour)
- **Duration**: ~3 hours for fine-tuning
- **Cost**: ~$1.60 per training run

### Inference (Monthly)
- **Instance**: ml.m5.large (~$0.115/hour)
- **24/7 Operation**: ~$83/month
- **With Auto-scaling**: ~$25-50/month

### Storage
- **Dataset Storage**: ~$0.10/month
- **Model Artifacts**: ~$0.50/month

**Total estimated monthly cost: $50-150**


## Testing

### API testing
```bash
# Health check
curl http://localhost:8000/health

# Function calling test
curl -X POST "http://localhost:8000/function-call" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather?", "tools": [...]}'
```

### SageMaker testing
```bash
# Direct endpoint test
python sagemaker_inference.py \
  --endpoint-name arcee-agent-endpoint \
  --test-query "What is the weather like?"
```

### Load testing
```bash
# Use included scripts
python test_api.py --load-test --concurrent-requests 10
```

## Documentation

- **README.md**: Complete setup and usage guide
- **SAGEMAKER_DEPLOYMENT.md**: Detailed deployment guide
- **API Documentation**: Available at http://localhost:8000/docs
