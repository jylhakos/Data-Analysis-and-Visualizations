#!/bin/bash

# Databricks BERT Fine-tuning Environment Setup Script
# This script sets up the local development environment for working with Databricks

set -e  # Exit on any error

echo "🚀 Setting up Databricks BERT Fine-tuning Development Environment"
echo "=================================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
    echo "----------------------------------------"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux systems"
    exit 1
fi

# Step 1: Check and install system dependencies
print_header "1. Installing System Dependencies"

# Update package list
print_status "Updating package list..."
sudo apt update

# Install required system packages
print_status "Installing required system packages..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    wget \
    unzip \
    git \
    jq \
    build-essential

# Step 2: Install AWS CLI
print_header "2. Installing AWS CLI"

if command -v aws &> /dev/null; then
    print_status "AWS CLI is already installed ($(aws --version))"
else
    print_status "Installing AWS CLI v2..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
    print_status "AWS CLI installed successfully"
fi

# Step 3: Install Terraform (optional)
print_header "3. Installing Terraform (Optional)"

if command -v terraform &> /dev/null; then
    print_status "Terraform is already installed ($(terraform version | head -n1))"
else
    print_status "Installing Terraform..."
    wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
    sudo apt update && sudo apt install -y terraform
    print_status "Terraform installed successfully"
fi

# Step 4: Create Python Virtual Environment
print_header "4. Setting up Python Virtual Environment"

VENV_NAME="databricks-bert-env"
VENV_PATH="$HOME/$VENV_NAME"

if [ -d "$VENV_PATH" ]; then
    print_warning "Virtual environment already exists at $VENV_PATH"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
        print_status "Removed existing virtual environment"
    else
        print_status "Using existing virtual environment"
    fi
fi

if [ ! -d "$VENV_PATH" ]; then
    print_status "Creating Python virtual environment..."
    python3 -m venv "$VENV_PATH"
    print_status "Virtual environment created at $VENV_PATH"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Step 5: Install Python Dependencies
print_header "5. Installing Python Dependencies"

print_status "Installing core ML and Databricks packages..."

# Core packages
pip install \
    torch>=2.0.0 \
    transformers>=4.20.0 \
    scikit-learn>=1.0.0 \
    numpy>=1.21.0 \
    pandas>=1.3.0 \
    mlflow>=2.0.0 \
    accelerate \
    datasets

# Databricks packages
pip install \
    databricks-cli \
    databricks-sdk \
    databricks-connect

# AWS packages
pip install \
    boto3 \
    awscli

# Additional ML packages
pip install \
    matplotlib \
    seaborn \
    jupyter \
    notebook \
    ipykernel

print_status "Python dependencies installed successfully"

# Step 6: Setup Jupyter Kernel
print_header "6. Setting up Jupyter Kernel"
python -m ipykernel install --user --name=$VENV_NAME --display-name="Databricks BERT Environment"
print_status "Jupyter kernel installed"

# Step 7: Create project structure
print_header "7. Creating Project Structure"

PROJECT_DIR="$HOME/databricks-bert-project"
mkdir -p "$PROJECT_DIR"/{data,notebooks,scripts,models,logs,configs}

print_status "Created project structure at $PROJECT_DIR"

# Step 8: Create configuration templates
print_header "8. Creating Configuration Templates"

# AWS Config template
cat > "$PROJECT_DIR/configs/aws-config-template.sh" << 'EOF'
#!/bin/bash
# AWS Configuration Template
# Copy this file to aws-config.sh and fill in your values

export AWS_PROFILE="your-profile-name"
export AWS_REGION="us-west-2"
export AWS_ACCOUNT_ID="your-account-id"
export S3_BUCKET="your-training-bucket"

# Databricks Configuration
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="your-personal-access-token"

# Optional: Set these if using specific credentials
# export AWS_ACCESS_KEY_ID="your-access-key"
# export AWS_SECRET_ACCESS_KEY="your-secret-key"
EOF

# Databricks CLI config template
cat > "$PROJECT_DIR/configs/databricks-config-template.json" << 'EOF'
{
  "profiles": {
    "DEFAULT": {
      "host": "https://your-workspace.cloud.databricks.com",
      "token": "your-personal-access-token"
    }
  }
}
EOF

# Training configuration template
cat > "$PROJECT_DIR/configs/training-config.yaml" << 'EOF'
# BERT Fine-tuning Configuration
model:
  model_name: "bert-base-uncased"
  num_labels: 2
  max_length: 128
  
training:
  num_epochs: 3
  batch_size: 16
  learning_rate: 2e-5
  warmup_steps: 500
  weight_decay: 0.01
  
databricks:
  cluster_name: "bert-training-cluster"
  node_type_id: "g4dn.2xlarge"
  min_workers: 1
  max_workers: 4
  spark_version: "13.3.x-ml-scala2.12"
  
data:
  train_path: "s3://your-bucket/data/train.jsonl"
  test_path: "s3://your-bucket/data/test.jsonl"
  
mlflow:
  experiment_name: "/Users/your-email@company.com/bert-fine-tuning"
  
output:
  model_path: "s3://your-bucket/models/"
  log_path: "s3://your-bucket/logs/"
EOF

print_status "Configuration templates created"

# Step 9: Create helper scripts
print_header "9. Creating Helper Scripts"

# AWS deployment script
cat > "$PROJECT_DIR/scripts/deploy-aws-infrastructure.sh" << 'EOF'
#!/bin/bash
# Deploy AWS Infrastructure for Databricks

set -e

echo "🚀 Deploying AWS Infrastructure for Databricks..."

# Load configuration
source ../configs/aws-config.sh

# Deploy using CloudFormation
echo "Deploying CloudFormation stack..."
aws cloudformation deploy \
    --template-file ../infrastructure/cloudformation/databricks-setup.yaml \
    --stack-name databricks-bert-infrastructure \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides \
        BucketName=$S3_BUCKET \
    --region $AWS_REGION

echo "✅ Infrastructure deployed successfully!"

# Get outputs
echo "📋 Stack Outputs:"
aws cloudformation describe-stacks \
    --stack-name databricks-bert-infrastructure \
    --query 'Stacks[0].Outputs' \
    --region $AWS_REGION \
    --output table
EOF

# Databricks setup script
cat > "$PROJECT_DIR/scripts/setup-databricks.py" << 'EOF'
#!/usr/bin/env python3
"""
Databricks Workspace Setup Script
This script sets up the Databricks workspace for BERT fine-tuning
"""

import os
import json
from databricks_sdk import WorkspaceClient
from databricks_sdk.service.compute import CreateCluster, ClusterSpec
from databricks_sdk.service.ml import CreateExperiment

def setup_databricks_workspace():
    """Setup Databricks workspace for BERT fine-tuning"""
    
    # Initialize Databricks client
    w = WorkspaceClient()
    
    print("🔧 Setting up Databricks workspace...")
    
    # Create cluster for BERT training
    cluster_config = CreateCluster(
        cluster_name="bert-training-cluster",
        spark_version="13.3.x-ml-scala2.12",
        node_type_id="g4dn.2xlarge",
        driver_node_type_id="g4dn.2xlarge",
        min_workers=1,
        max_workers=4,
        autotermination_minutes=120,
        enable_elastic_disk=True,
        init_scripts=[],
        spark_conf={
            "spark.databricks.cluster.profile": "ml",
            "spark.databricks.delta.preview.enabled": "true"
        }
    )
    
    try:
        cluster = w.clusters.create_and_wait(cluster_config)
        print(f"✅ Cluster created: {cluster.cluster_id}")
    except Exception as e:
        print(f"❌ Failed to create cluster: {e}")
    
    # Create MLflow experiment
    try:
        experiment = w.experiments.create_experiment(
            name="/Users/{}/bert-fine-tuning".format(w.current_user.me().user_name),
            artifact_location="s3://your-bucket/mlflow-artifacts/"
        )
        print(f"✅ MLflow experiment created: {experiment.experiment_id}")
    except Exception as e:
        print(f"❌ Failed to create experiment: {e}")

if __name__ == "__main__":
    setup_databricks_workspace()
EOF

# Data upload script
cat > "$PROJECT_DIR/scripts/upload-sample-data.sh" << 'EOF'
#!/bin/bash
# Upload sample training data to S3

set -e

echo "📤 Uploading sample data to S3..."

# Load configuration
source ../configs/aws-config.sh

# Create sample training data
mkdir -p ../data/sample

# Create sample JSON Lines file for text classification
cat > ../data/sample/train.jsonl << 'SAMPLE_DATA'
{"text": "I love this product! It's amazing and works perfectly.", "label": 1}
{"text": "This is terrible. Complete waste of money.", "label": 0}
{"text": "Great quality and fast shipping. Highly recommend!", "label": 1}
{"text": "Poor quality control. Item arrived damaged.", "label": 0}
{"text": "Excellent customer service and product quality.", "label": 1}
{"text": "Not what I expected. Very disappointed.", "label": 0}
SAMPLE_DATA

cat > ../data/sample/test.jsonl << 'SAMPLE_DATA'
{"text": "Outstanding product! Exceeded my expectations.", "label": 1}
{"text": "Horrible experience. Would not recommend.", "label": 0}
{"text": "Good value for money. Works as described.", "label": 1}
{"text": "Defective item. Poor customer support.", "label": 0}
SAMPLE_DATA

# Upload to S3
aws s3 cp ../data/sample/train.jsonl s3://$S3_BUCKET/data/train.jsonl
aws s3 cp ../data/sample/test.jsonl s3://$S3_BUCKET/data/test.jsonl

echo "✅ Sample data uploaded successfully!"
EOF

# Make scripts executable
chmod +x "$PROJECT_DIR/scripts/"*.sh
chmod +x "$PROJECT_DIR/scripts/"*.py

print_status "Helper scripts created and made executable"

# Step 10: Create sample notebook
print_header "10. Creating Sample Jupyter Notebook"

cat > "$PROJECT_DIR/notebooks/bert-fine-tuning-databricks.ipynb" << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# BERT Fine-tuning on Databricks\n",
    "\n",
    "This notebook demonstrates how to fine-tune a BERT model for text classification using Databricks and MLflow."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install required packages (run this in Databricks)\n",
    "# %pip install transformers torch accelerate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import mlflow\n",
    "import mlflow.pytorch\n",
    "import torch\n",
    "from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer\n",
    "from torch.utils.data import Dataset\n",
    "import pandas as pd\n",
    "import json\n",
    "\n",
    "# Configure MLflow\n",
    "mlflow.set_registry_uri(\"databricks-uc\")\n",
    "\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")\n",
    "print(f\"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Custom Dataset class\n",
    "class TextClassificationDataset(Dataset):\n",
    "    def __init__(self, texts, labels, tokenizer, max_length=128):\n",
    "        self.texts = texts\n",
    "        self.labels = labels\n",
    "        self.tokenizer = tokenizer\n",
    "        self.max_length = max_length\n",
    "    \n",
    "    def __len__(self):\n",
    "        return len(self.texts)\n",
    "    \n",
    "    def __getitem__(self, idx):\n",
    "        text = str(self.texts[idx])\n",
    "        label = self.labels[idx]\n",
    "        \n",
    "        encoding = self.tokenizer(\n",
    "            text,\n",
    "            truncation=True,\n",
    "            padding='max_length',\n",
    "            max_length=self.max_length,\n",
    "            return_tensors='pt'\n",
    "        )\n",
    "        \n",
    "        return {\n",
    "            'input_ids': encoding['input_ids'].flatten(),\n",
    "            'attention_mask': encoding['attention_mask'].flatten(),\n",
    "            'labels': torch.tensor(label, dtype=torch.long)\n",
    "        }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data from S3 (update with your bucket name)\n",
    "def load_jsonl(file_path):\n",
    "    data = []\n",
    "    with open(file_path, 'r') as f:\n",
    "        for line in f:\n",
    "            data.append(json.loads(line))\n",
    "    return data\n",
    "\n",
    "# For Databricks, use dbutils to copy from S3\n",
    "# dbutils.fs.cp(\"s3://your-bucket/data/train.jsonl\", \"/tmp/train.jsonl\")\n",
    "# dbutils.fs.cp(\"s3://your-bucket/data/test.jsonl\", \"/tmp/test.jsonl\")\n",
    "\n",
    "# train_data = load_jsonl(\"/tmp/train.jsonl\")\n",
    "# test_data = load_jsonl(\"/tmp/test.jsonl\")\n",
    "\n",
    "# For local testing, use sample data\n",
    "train_data = [\n",
    "    {\"text\": \"I love this product!\", \"label\": 1},\n",
    "    {\"text\": \"This is terrible.\", \"label\": 0},\n",
    "    {\"text\": \"Great quality!\", \"label\": 1},\n",
    "    {\"text\": \"Poor quality.\", \"label\": 0}\n",
    "]\n",
    "\n",
    "test_data = [\n",
    "    {\"text\": \"Outstanding product!\", \"label\": 1},\n",
    "    {\"text\": \"Horrible experience.\", \"label\": 0}\n",
    "]\n",
    "\n",
    "print(f\"Training samples: {len(train_data)}\")\n",
    "print(f\"Test samples: {len(test_data)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize tokenizer and model\n",
    "model_name = 'bert-base-uncased'\n",
    "tokenizer = BertTokenizer.from_pretrained(model_name)\n",
    "model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)\n",
    "\n",
    "# Prepare datasets\n",
    "train_texts = [item['text'] for item in train_data]\n",
    "train_labels = [item['label'] for item in train_data]\n",
    "test_texts = [item['text'] for item in test_data]\n",
    "test_labels = [item['label'] for item in test_data]\n",
    "\n",
    "train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)\n",
    "test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training configuration\n",
    "training_args = TrainingArguments(\n",
    "    output_dir='./results',\n",
    "    num_train_epochs=2,\n",
    "    per_device_train_batch_size=8,\n",
    "    per_device_eval_batch_size=8,\n",
    "    warmup_steps=100,\n",
    "    weight_decay=0.01,\n",
    "    logging_dir='./logs',\n",
    "    logging_steps=10,\n",
    "    evaluation_strategy=\"epoch\",\n",
    "    save_strategy=\"epoch\",\n",
    "    load_best_model_at_end=True,\n",
    "    report_to=\"none\"  # Disable wandb\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Start MLflow run\n",
    "with mlflow.start_run(run_name=\"bert_fine_tuning\") as run:\n",
    "    # Log parameters\n",
    "    mlflow.log_param(\"model_name\", model_name)\n",
    "    mlflow.log_param(\"num_epochs\", training_args.num_train_epochs)\n",
    "    mlflow.log_param(\"batch_size\", training_args.per_device_train_batch_size)\n",
    "    mlflow.log_param(\"learning_rate\", training_args.learning_rate)\n",
    "    \n",
    "    # Initialize trainer\n",
    "    trainer = Trainer(\n",
    "        model=model,\n",
    "        args=training_args,\n",
    "        train_dataset=train_dataset,\n",
    "        eval_dataset=test_dataset,\n",
    "    )\n",
    "    \n",
    "    # Train the model\n",
    "    print(\"Starting training...\")\n",
    "    trainer.train()\n",
    "    \n",
    "    # Evaluate the model\n",
    "    eval_results = trainer.evaluate()\n",
    "    \n",
    "    # Log metrics\n",
    "    for key, value in eval_results.items():\n",
    "        mlflow.log_metric(key, value)\n",
    "    \n",
    "    # Log model\n",
    "    mlflow.pytorch.log_model(model, \"bert_model\")\n",
    "    \n",
    "    print(f\"Training completed! Run ID: {run.info.run_id}\")\n",
    "    print(f\"Evaluation results: {eval_results}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Databricks BERT Environment",
   "language": "python",
   "name": "databricks-bert-env"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

print_status "Sample Jupyter notebook created"

# Step 11: Create README for the project
print_header "11. Creating Project README"

cat > "$PROJECT_DIR/README.md" << 'EOF'
# Databricks BERT Fine-tuning Project

This project provides a complete setup for fine-tuning BERT models using Databricks and AWS services.

## Project Structure

```
databricks-bert-project/
├── configs/                 # Configuration files
│   ├── aws-config-template.sh
│   ├── databricks-config-template.json
│   └── training-config.yaml
├── data/                    # Training data
│   └── sample/
├── notebooks/               # Jupyter notebooks
│   └── bert-fine-tuning-databricks.ipynb
├── scripts/                 # Helper scripts
│   ├── deploy-aws-infrastructure.sh
│   ├── setup-databricks.py
│   └── upload-sample-data.sh
├── models/                  # Saved models
├── logs/                    # Training logs
└── README.md
```

## Quick Start

1. **Configure AWS and Databricks**:
   ```bash
   cd configs/
   cp aws-config-template.sh aws-config.sh
   # Edit aws-config.sh with your AWS and Databricks details
   ```

2. **Deploy AWS Infrastructure**:
   ```bash
   cd scripts/
   ./deploy-aws-infrastructure.sh
   ```

3. **Upload Sample Data**:
   ```bash
   ./upload-sample-data.sh
   ```

4. **Setup Databricks Workspace**:
   ```bash
   python setup-databricks.py
   ```

5. **Run the Notebook**:
   Open `notebooks/bert-fine-tuning-databricks.ipynb` in Databricks or Jupyter

## Requirements

- Python 3.8+
- AWS CLI configured
- Databricks account and workspace
- GPU-enabled Databricks cluster (recommended)

## Environment Variables

Make sure to set these in your `configs/aws-config.sh`:

- `AWS_PROFILE` or `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `S3_BUCKET`
- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`

## GPU Recommendations

For optimal performance, use these Databricks cluster configurations:

- **Small datasets**: `g4dn.xlarge` or `g4dn.2xlarge`
- **Medium datasets**: `p3.2xlarge`
- **Large datasets**: `p3.8xlarge` or `p3.16xlarge`

## Troubleshooting

1. **AWS Permissions**: Ensure your AWS credentials have the necessary permissions for S3, EC2, and IAM
2. **Databricks Token**: Make sure your personal access token has the required workspace permissions
3. **GPU Memory**: If you encounter CUDA out-of-memory errors, reduce the batch size or use gradient accumulation

## Next Steps

1. Replace sample data with your actual training dataset
2. Adjust model hyperparameters in the configuration files
3. Set up automated training pipelines using Databricks jobs
4. Implement model deployment using MLflow model serving

EOF

print_status "Project README created"

# Step 12: Final setup and verification
print_header "12. Final Setup and Verification"

# Create activation script
cat > "$PROJECT_DIR/activate-env.sh" << EOF
#!/bin/bash
# Activation script for Databricks BERT environment

echo "🚀 Activating Databricks BERT Environment"
echo "========================================"

# Activate virtual environment
source $VENV_PATH/bin/activate

# Load AWS configuration if it exists
if [ -f "./configs/aws-config.sh" ]; then
    source ./configs/aws-config.sh
    echo "✅ AWS configuration loaded"
else
    echo "⚠️  AWS configuration not found. Copy configs/aws-config-template.sh to configs/aws-config.sh and configure it."
fi

# Set project directory
cd $PROJECT_DIR
export PROJECT_ROOT=$PROJECT_DIR

echo "📁 Project directory: $PROJECT_DIR"
echo "🐍 Python environment: $VENV_NAME"
echo "💡 To deactivate, run: deactivate"
echo ""
echo "🎯 Next steps:"
echo "   1. Configure AWS credentials: aws configure"
echo "   2. Configure Databricks CLI: databricks configure --token"
echo "   3. Edit configs/aws-config.sh with your settings"
echo "   4. Run scripts/deploy-aws-infrastructure.sh"

# Change to project directory
cd $PROJECT_DIR
EOF

chmod +x "$PROJECT_DIR/activate-env.sh"

# Add to .bashrc for easy access
echo "" >> ~/.bashrc
echo "# Databricks BERT Environment" >> ~/.bashrc
echo "alias activate-databricks='source $PROJECT_DIR/activate-env.sh'" >> ~/.bashrc

print_status "Environment activation script created"

# Final verification
print_header "13. Verification and Summary"

print_status "Verifying installation..."

# Check Python packages
source "$VENV_PATH/bin/activate"

echo "Python packages:"
pip list | grep -E "(torch|transformers|databricks|mlflow|boto3)" || true

# Check CLI tools
echo ""
echo "CLI tools:"
aws --version 2>/dev/null && echo "✅ AWS CLI installed" || echo "❌ AWS CLI not found"
terraform --version 2>/dev/null && echo "✅ Terraform installed" || echo "❌ Terraform not found"

print_header "🎉 Setup Complete!"

echo ""
echo "📋 Summary:"
echo "==========="
echo "✅ System dependencies installed"
echo "✅ AWS CLI installed"
echo "✅ Terraform installed"
echo "✅ Python virtual environment created: $VENV_PATH"
echo "✅ Python packages installed"
echo "✅ Project structure created: $PROJECT_DIR"
echo "✅ Configuration templates created"
echo "✅ Helper scripts created"
echo "✅ Sample notebook created"
echo ""
echo "🚀 Next Steps:"
echo "=============="
echo "1. Configure AWS credentials:"
echo "   aws configure"
echo ""
echo "2. Configure Databricks CLI:"
echo "   databricks configure --token"
echo ""
echo "3. Activate the environment:"
echo "   source $PROJECT_DIR/activate-env.sh"
echo "   # Or use the alias: activate-databricks"
echo ""
echo "4. Edit configuration files:"
echo "   nano $PROJECT_DIR/configs/aws-config.sh"
echo ""
echo "5. Deploy AWS infrastructure:"
echo "   cd $PROJECT_DIR/scripts && ./deploy-aws-infrastructure.sh"
echo ""
echo "6. Upload sample data:"
echo "   cd $PROJECT_DIR/scripts && ./upload-sample-data.sh"
echo ""
echo "7. Open and run the Jupyter notebook:"
echo "   jupyter notebook $PROJECT_DIR/notebooks/bert-fine-tuning-databricks.ipynb"
echo ""
echo "📚 For more information, check:"
echo "   - Project README: $PROJECT_DIR/README.md"
echo "   - AWS Infrastructure: $PROJECT_DIR/infrastructure/"
echo "   - Configuration templates: $PROJECT_DIR/configs/"
echo ""
print_status "Happy fine-tuning! 🤖🚀"
