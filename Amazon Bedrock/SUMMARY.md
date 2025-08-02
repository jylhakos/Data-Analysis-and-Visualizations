# Amazon Bedrock BERT fine-tuning

This project demonstrates how to fine-tune BERT models using **Amazon Bedrock** - AWS's fully managed foundation model service. The solution combines local development capabilities with cloud-scale managed infrastructure for enterprise-grade machine learning workflows.

## 📁 Project structure

```
Amazon Bedrock/
├── README.md                                    # Main project documentation
└── fine-tuning/
    ├── README.md                               # Fine-tuning specific documentation
    ├── setup_bedrock.sh                       # 🚀 Automated setup script
    ├── bedrock_example.py                      # 🎯 Complete workflow example
    ├── cost_monitor.py                         # 💰 Cost monitoring and alerts
    │
    ├── src/                                   # Source code
    │   ├── bert_fine_tuning.py               # Original BERT fine-tuning
    │   ├── bedrock_bert_fine_tuning.py       # 🆕 Bedrock integration
    │   ├── minimal_bert.py                   # Simplified version
    │   └── test_environment.py               # Environment testing
    │
    ├── infrastructure/                        # 🏗️ Infrastructure as Code
    │   ├── main.tf                          # Terraform configuration
    │   ├── variables.tf                     # Terraform variables
    │   ├── bedrock-cloudformation.yaml      # CloudFormation template
    │   └── templates/
    │       └── aws_config.tpl               # AWS configuration template
    │
    ├── requirements-bedrock.txt              # 🆕 Bedrock dependencies
    ├── requirements.txt                      # Core dependencies
    ├── requirements-api.txt                  # API dependencies
    │
    ├── api.py                               # FastAPI backend
    ├── examples.py                          # Usage examples
    ├── test_setup.py                        # Setup verification
    ├── test_model.py                        # Model testing
    └── test_api.sh                          # API testing
```

## Features

### **BERT fine-tuning with Amazon Bedrock**
- **Managed Infrastructure**: No need to provision or manage compute resources
- **Scalable Training**: Automatic scaling based on workload
- **Cost Optimization**: Pay-per-use pricing model
- **Enterprise Security**: Built-in encryption and compliance

### **Infrastructure as Code**
- **Terraform Support**: Complete AWS resource provisioning
- **CloudFormation Alternative**: For teams preferring CloudFormation
- **Automated Setup**: One-command deployment script
- **Resource Management**: Automatic cleanup and cost control

### **Cost management**
- **Budget Controls**: Automatic spending limits and alerts
- **Cost Monitoring**: Real-time usage tracking
- **Optimization Recommendations**: AI-driven cost optimization
- **Billing Alerts**: Email notifications for cost thresholds

### 🔐 **Security & compliance**
- **IAM Integration**: Role-based access control
- **Data Encryption**: Encryption at rest and in transit
- **Audit Logging**: Complete audit trails
- **Compliance Ready**: Enterprise security standards

## Start

### Option 1: Automated setup (Recommended)
```bash
# Navigate to project directory
cd "Amazon Bedrock/fine-tuning"

# Run automated setup
chmod +x setup_bedrock.sh
./setup_bedrock.sh

# Follow interactive prompts for configuration
```

### Option 2: Manual setup
```bash
# 1. Setup AWS CLI
aws configure

# 2. Create Python environment
python3 -m venv bedrock_env
source bedrock_env/bin/activate
pip install -r requirements-bedrock.txt

# 3. Deploy infrastructure
cd infrastructure
terraform init && terraform apply

# 4. Run fine-tuning
python bedrock_example.py --mode all
```

## **Components**

### **Local development layer**
- Python scripts for data preparation and model training
- Integration with existing BERT fine-tuning code
- Development and testing utilities

### **AWS Cloud layer**
- **Amazon Bedrock**: Managed fine-tuning service
- **Amazon S3**: Training data and model artifact storage
- **IAM**: Security and access management
- **CloudWatch**: Monitoring and logging

### **Infrastructure layer**
- **Terraform/CloudFormation**: Infrastructure provisioning
- **Cost Controls**: Budgets and billing alerts
- **Monitoring**: Performance and cost tracking

### **DevOps layer**
- **CI/CD Integration**: Automated deployment pipelines
- **Cost Optimization**: Automated resource management
- **Quality Assurance**: Testing and validation workflows

## **Use Cases**

### **Sentiment Analysis**
Fine-tune BERT for customer feedback classification
```python
# Example: Customer review sentiment analysis
texts = ["Great product!", "Poor quality"]
labels = [1, 0]  # positive, negative
```

### **Document Classification**
Categorize documents by type or content
```python
# Example: Document type classification
texts = ["Legal contract text", "Marketing email content"]
labels = [0, 1]  # legal, marketing
```

### **Content Moderation**
Automatically detect inappropriate content
```python
# Example: Content safety classification
texts = ["Family-friendly content", "Inappropriate content"]
labels = [1, 0]  # safe, unsafe
```

## **Best Practices**

### **Data preparation**
- Use high-quality, domain-specific training data
- Ensure balanced label distributions
- Validate data format before uploading to S3

### **Cost optimization**
- Set realistic budget limits
- Monitor costs daily during training
- Use spot instances when available
- Clean up unused resources regularly

### **Security**
- Use IAM roles with minimal required permissions
- Enable CloudTrail for audit logging
- Encrypt sensitive data at rest and in transit
- Regularly rotate access keys

### **Performance**
- Start with small datasets for initial experiments
- Use appropriate hyperparameters for your domain
- Monitor training metrics for early stopping
- Validate model performance on held-out test data

## 🔧 **Configuration options**

### **Terraform variables**
```hcl
# infrastructure/terraform.tfvars
aws_region = "us-east-1"
environment = "dev"
project_name = "bert-fine-tuning"
max_budget_amount = 100
notification_email = "your-email@example.com"
```

### **Python configuration**
```python
# Bedrock configuration
config = BedrockConfig(
    region_name='us-east-1',
    s3_bucket='your-training-bucket',
    base_model_id='amazon.titan-text-express-v1',
    max_training_cost=100.0
)
```

## **Monitoring & analytics**

### **Cost monitoring**
```bash
# Check current costs
python cost_monitor.py --check-current

# Generate monthly report
python cost_monitor.py --report monthly

# Set cost alerts
python cost_monitor.py --set-alert 50.0 --email your-email@example.com
```

### **Training metrics**
- Training loss and accuracy
- Validation performance
- Resource utilization
- Cost per training epoch

### **Performance evaluation**
- Model accuracy on test data
- Inference latency
- Cost per prediction
- Scalability metrics

**Ready to get started?** Run `./setup_bedrock.sh` and follow the interactive setup process!

## **Troubleshooting**

### **Issues**
1. **Amazon AWS permissions**: Ensure proper IAM roles and policies
2. **Data Format**: Verify training data is in correct JSONL format
3. **Cost Overruns**: Set up billing alerts and monitor regularly
4. **Training Failures**: Check CloudWatch logs for error details

---
