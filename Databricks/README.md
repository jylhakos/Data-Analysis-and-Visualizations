# Explore how to use Databricks

![alt text](https://github.com/jylhakos/Data-Analysis-and-Visualizations/blob/main/Databricks/An_overview_of_generative_AI_capabilities_with_Databricks_and_AWS.png?raw=true)

_Figure: An overview of generative AI capabilities with_ **Amazon AWS and Databricks**

## Use case: Fine-tuning a BERT model

### What is Amazon Databricks?

Amazon Databricks is a unified data analytics platform that combines data engineering, data science, and machine learning on a single collaborative platform. Built on Apache Spark, it provides:

- **Collaborative workspace**: Interactive notebooks supporting Python, R, Scala, and SQL
- **Unified data platform**: Integration with AWS services (S3, RDS, Redshift, etc.)
- **MLflow integration**: Complete ML lifecycle management (tracking, projects, models, registry)
- **Auto-scaling compute**: Elastic clusters that scale based on workload demands
- **Unity catalog**: Centralized governance for data and AI assets

### Amazon Databricks vs Amazon Bedrock

| Feature | Amazon Databricks | Amazon Bedrock |
|---------|-------------------|----------------|
| **Purpose** | Data analytics and custom ML model development | Pre-trained foundation models as a service |
| **Use Case** | Custom BERT fine-tuning, data processing, MLOps | Quick deployment of LLMs without training |
| **Control** | Full control over model architecture and training | Limited customization, pre-built models |
| **Cost** | Pay for compute resources and storage | Pay per API call/token usage |
| **Time to Deploy** | Longer (custom development) | Immediate (API-based) |
| **Best for BERT Fine-tuning** | ✅ Full control, custom datasets | ❌ Limited BERT fine-tuning options |

**Choose Databricks when**: You need custom BERT fine-tuning, have specific domain data, require full MLOps pipeline control, or need data processing capabilities.

**Choose Bedrock when**: You need quick deployment, standard use cases, minimal customization, or want to avoid infrastructure management.

### Amazon AWS resources required for BERT fine-tuning

#### Amazon AWS services
1. **Amazon Databricks Workspace**
   - Multi-node clusters with GPU instances (p3.2xlarge, p3.8xlarge, or g4dn instances)
   - Databricks Runtime for Machine Learning (DBR ML)

2. **Amazon S3**
   - Store training datasets, model artifacts, and logs
   - Versioned model storage and backup

3. **Amazon IAM**
   - Databricks service roles and policies
   - Cross-account access for S3 and other services

4. **Amazon VPC**
   - Secure network isolation for Databricks clusters
   - Private subnets for compute resources

#### Recommended instance types for BERT Fine-tuning
- **Small datasets (< 1GB)**: `g4dn.xlarge` or `g4dn.2xlarge`
- **Medium datasets (1-10GB)**: `p3.2xlarge` or `g4dn.4xlarge`
- **Large datasets (> 10GB)**: `p3.8xlarge` or `p3.16xlarge`

### Databricks BERT fine-tuning architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS Cloud Environment                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐                    │
│  │   Amazon S3     │    │   Unity Catalog  │                    │
│  │                 │    │                  │                    │
│  │ • Training Data │    │ • Model Registry │                    │
│  │ • Model Artifacts│   │ • Data Governance│                    │
│  │ • Logs & Metrics│    │ • Access Control │                    │
│  └─────────────────┘    └──────────────────┘                    │
│           │                       │                             │
│           ▼                       ▼                             │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │            Amazon Databricks Workspace                      ││
│  │                                                             ││
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ ││
│  │  │   Data Prep     │  │  BERT Training  │  │   MLflow     │ ││
│  │  │                 │  │                 │  │              │ ││
│  │  │ • Load Data     │  │ • Model Loading │  │ • Experiment │ ││
│  │  │ • Tokenization  │  │ • Fine-tuning   │  │   Tracking   │ ││
│  │  │ • Preprocessing │  │ • Quantization  │  │ • Model      │ ││
│  │  │                 │  │ • Evaluation    │  │   Versioning │ ││
│  │  └─────────────────┘  └─────────────────┘  └──────────────┘ ││
│  │                                                             ││
│  │  ┌─────────────────────────────────────────────────────────┐││
│  │  │              GPU Cluster (p3/g4dn instances)            │││
│  │  │                                                         │││
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │││
│  │  │  │   Worker    │ │   Worker    │ │   Worker    │        │││
│  │  │  │    Node     │ │    Node     │ │    Node     │        │││
│  │  │  │             │ │             │ │             │        │││
│  │  │  │ • PyTorch   │ │ • PyTorch   │ │ • PyTorch   │        │││
│  │  │  │ • CUDA      │ │ • CUDA      │ │ • CUDA      │        │││
│  │  │  │ • BERT      │ │ • BERT      │ │ • BERT      │        │││
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘        │││
│  │  └─────────────────────────────────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  ┌─────────────────┐    ┌──────────────────┐                    │
│  │   Amazon IAM    │    │   Amazon VPC     │                    │
│  │                 │    │                  │                    │
│  │ • Service Roles │    │ • Network Security│                   │
│  │ • Access Policies│   │ • Private Subnets│                    │
│  │ • Cross-account │    │ • Security Groups│                    │
│  └─────────────────┘    └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### BERT fine-tuning pipeline on Databricks

#### Phase 1: Environment setup and data preparation

1. **Databricks Workspace configuration**
   ```python
   # Configure MLflow for Unity Catalog
   import mlflow
   mlflow.set_registry_uri("databricks-uc")
   
   # Set catalog and schema
   CATALOG_NAME = "ml_models"
   SCHEMA_NAME = "bert_experiments"
   ```

2. **Data loading and preprocessing**
   ```python
   # Load training data from S3
   training_data = spark.read.format("json").load("s3://your-bucket/bert-training-data/")
   
   # Convert to Pandas for tokenization
   df = training_data.toPandas()
   
   # Tokenize using BERT tokenizer
   from transformers import BertTokenizer
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   ```

#### Phase 2: Model fine-tuning

1. **BERT model loading and configuration**
   ```python
   from transformers import BertForSequenceClassification
   
   # Load pre-trained BERT model
   model = BertForSequenceClassification.from_pretrained(
       'bert-base-uncased', 
       num_labels=2  # Adjust based on your task
   )
   
   # Enable quantization for memory efficiency
   from torch.quantization import quantize_dynamic
   quantized_model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
   ```

2. **Training loop with MLflow tracking**
   ```python
   with mlflow.start_run(run_name="bert_fine_tuning") as run:
       # Training configuration
       optimizer = AdamW(model.parameters(), lr=2e-5)
       
       # Training loop
       for epoch in range(num_epochs):
           model.train()
           total_loss = 0
           
           for batch in train_dataloader:
               outputs = model(**batch)
               loss = outputs.loss
               loss.backward()
               optimizer.step()
               optimizer.zero_grad()
               
               total_loss += loss.item()
           
           # Log metrics
           mlflow.log_metric("epoch_loss", total_loss/len(train_dataloader), step=epoch)
   ```

#### Phase 3: Model evaluation and registration

1. **Model evaluation**
   ```python
   # Evaluate on test set
   model.eval()
   predictions = []
   true_labels = []
   
   with torch.no_grad():
       for batch in test_dataloader:
           outputs = model(**batch)
           predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
           true_labels.extend(batch['labels'].cpu().numpy())
   
   # Calculate metrics
   from sklearn.metrics import accuracy_score, classification_report
   accuracy = accuracy_score(true_labels, predictions)
   report = classification_report(true_labels, predictions)
   
   # Log evaluation metrics
   mlflow.log_metric("test_accuracy", accuracy)
   mlflow.log_text(report, "classification_report.txt")
   ```

2. **Model registration**
   ```python
   # Register model to Unity Catalog
   model_uri = f"runs:{run.info.run_id}/model"
   mlflow.register_model(
       model_uri,
       f"{CATALOG_NAME}.{SCHEMA_NAME}.bert_sentiment_classifier"
   )
   ```

### Step-by-Step implementation

#### Step 1: AWS Infrastructure setup

1. **Create Databricks Workspace**
   ```bash
   # Using AWS CLI to create necessary IAM roles
   aws iam create-role --role-name DatabricksRole \
     --assume-role-policy-document file://databricks-trust-policy.json
   
   # Attach required policies
   aws iam attach-role-policy --role-name DatabricksRole \
     --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
   ```

2. **Setup S3 Buckets**
   ```bash
   # Create S3 bucket for data and models
   aws s3 mb s3://your-bert-training-bucket
   aws s3 mb s3://your-bert-models-bucket
   
   # Upload training data
   aws s3 cp ./training_data/ s3://your-bert-training-bucket/data/ --recursive
   ```

#### Step 2: Databricks environment configuration

1. **Create Compute Cluster**
   - Instance type: `g4dn.2xlarge` (GPU-enabled)
   - Databricks Runtime: `13.3 LTS ML (includes Apache Spark 3.4.1, GPU, Scala 2.12)`
   - Auto-scaling: 1-4 workers

2. **Install required libraries**
   ```python
   %pip install torch>=2.0.0 transformers>=4.20.0 accelerate
   ```

#### Step 3: Data preparation

1. **Load and explore data**
   ```python
   # Read data from S3
   df = spark.read.format("json").option("multiline", "true").load("s3://your-bert-training-bucket/data/")
   df.show(5)
   
   # Convert to Pandas for processing
   pandas_df = df.toPandas()
   print(f"Dataset size: {len(pandas_df)} samples")
   ```

2. **Text preprocessing and tokenization**
   ```python
   from transformers import BertTokenizer
   
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   
   def tokenize_texts(texts, labels, max_length=128):
       inputs = tokenizer(
           texts,
           padding=True,
           truncation=True,
           max_length=max_length,
           return_tensors="pt"
       )
       inputs['labels'] = torch.tensor(labels)
       return inputs
   
   # Apply tokenization
   train_inputs = tokenize_texts(train_texts, train_labels)
   test_inputs = tokenize_texts(test_texts, test_labels)
   ```

#### Step 4: Model fine-tuning

1. **Model configuration**
   ```python
   from transformers import BertForSequenceClassification, TrainingArguments, Trainer
   
   model = BertForSequenceClassification.from_pretrained(
       'bert-base-uncased',
       num_labels=2,
       output_attentions=False,
       output_hidden_states=False,
   )
   
   # Move to GPU if available
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model.to(device)
   ```

2. **Training configuration**
   ```python
   training_args = TrainingArguments(
       output_dir='./results',
       num_train_epochs=3,
       per_device_train_batch_size=16,
       per_device_eval_batch_size=64,
       warmup_steps=500,
       weight_decay=0.01,
       logging_dir='./logs',
       logging_steps=10,
       evaluation_strategy="epoch",
       save_strategy="epoch",
       load_best_model_at_end=True,
   )
   ```

#### Step 5: Model training and monitoring

1. **Training with MLflow**
   ```python
   import mlflow
   import mlflow.pytorch
   
   with mlflow.start_run(run_name="bert_fine_tuning_experiment"):
       trainer = Trainer(
           model=model,
           args=training_args,
           train_dataset=train_dataset,
           eval_dataset=eval_dataset,
       )
       
       # Start training
       trainer.train()
       
       # Log model
       mlflow.pytorch.log_model(model, "bert_model")
       
       # Log metrics
       eval_results = trainer.evaluate()
       for key, value in eval_results.items():
           mlflow.log_metric(key, value)
   ```

#### Step 6: Model evaluation and deployment

1. **Evaluation**
   ```python
   from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
   
   # Get predictions
   predictions = trainer.predict(test_dataset)
   y_pred = np.argmax(predictions.predictions, axis=1)
   y_true = predictions.label_ids
   
   # Calculate metrics
   accuracy = accuracy_score(y_true, y_pred)
   precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
   
   print(f"Accuracy: {accuracy:.4f}")
   print(f"Precision: {precision:.4f}")
   print(f"Recall: {recall:.4f}")
   print(f"F1-Score: {f1:.4f}")
   
   # Log evaluation metrics
   mlflow.log_metric("test_accuracy", accuracy)
   mlflow.log_metric("test_precision", precision)
   mlflow.log_metric("test_recall", recall)
   mlflow.log_metric("test_f1", f1)
   ```

2. **Model registration and deployment**
   ```python
   # Register model to Unity Catalog
   model_version = mlflow.register_model(
       f"runs:/{mlflow.active_run().info.run_id}/bert_model",
       f"{CATALOG_NAME}.{SCHEMA_NAME}.bert_classifier"
   )
   
   # Transition to production
   from mlflow.tracking import MlflowClient
   client = MlflowClient()
   client.transition_model_version_stage(
       name=f"{CATALOG_NAME}.{SCHEMA_NAME}.bert_classifier",
       version=model_version.version,
       stage="Production"
   )
   ```

### Local development environment setup

#### Prerequisites

1. **AWS CLI configuration**
   ```bash
   # Install AWS CLI
   curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
   unzip awscliv2.zip
   sudo ./aws/install
   
   # Configure AWS credentials
   aws configure
   ```

2. **Python Virtual Environment**
   ```bash
   # Create virtual environment
   python3 -m venv databricks-bert-env
   source databricks-bert-env/bin/activate
   
   # Install required packages
   pip install -r requirements.txt
   pip install databricks-cli databricks-sdk boto3
   ```

3. **Databricks CLI setup**
   ```bash
   # Configure Databricks CLI
   databricks configure --token
   # Enter your Databricks workspace URL and personal access token
   ```

### Integration with Amazon AWS services

Databricks seamlessly integrates with AWS services:

1. **Amazon S3**: Direct access to data lakes and model storage
2. **AWS IAM**: Fine-grained access control and security
3. **Amazon VPC**: Network isolation and security
4. **AWS CloudFormation**: Infrastructure as code for Databricks deployment
5. **Amazon CloudWatch**: Monitoring and logging integration
6. **AWS Glue**: Data catalog integration with Unity Catalog

### Performance optimization

1. **Use GPU-optimized instances** for faster training
2. **Implement gradient accumulation** for larger effective batch sizes
3. **Use mixed precision training** to reduce memory usage
4. **Cache datasets** in Databricks File System (DBFS) for faster access
5. **Leverage Delta Lake** for versioned datasets and faster I/O

This comprehensive guide provides the foundation for fine-tuning BERT models using Databricks on AWS, offering both flexibility and scalability for production ML workflows.

## Start

### Prerequisites
- AWS Account with appropriate permissions
- Databricks workspace (trial or production)
- Basic knowledge of Python and machine learning

### 1. Local environment setup
```bash
# Clone or navigate to this repository
cd Databricks/

# Run the automated setup script
./scripts/setup-local-environment.sh

# This script will:
# - Install AWS CLI, Terraform, and Python dependencies
# - Create a virtual environment with all required packages
# - Set up project structure and configuration templates
# - Create helper scripts for deployment
```

### 2. AWS infrastructure deployment
```bash
# Option A: Using CloudFormation
aws cloudformation deploy \
    --template-file infrastructure/cloudformation/databricks-setup.yaml \
    --stack-name databricks-bert-infrastructure \
    --capabilities CAPABILITY_IAM

# Option B: Using Terraform
cd infrastructure/terraform/
terraform init
terraform plan
terraform apply
```

### 3. Configure Databricks
```bash
# Configure Databricks CLI
databricks configure --token

# Edit configuration files
nano fine-tuning/config/training-config.yaml

# Upload sample data
./scripts/upload-sample-data.sh
```

### 4. Run BERT fine-tuning

#### Option A: Using Databricks Notebooks (Recommended)
1. Upload `fine-tuning/notebooks/bert_fine_tuning_databricks.py` to your Databricks workspace
2. Create a GPU-enabled cluster (g4dn.2xlarge or p3.2xlarge)
3. Run the notebook step by step

#### Option B: Using Python scripts
```bash
# Local testing
cd fine-tuning/
python databricks_bert_fine_tuning.py --config config/training-config.yaml --local

# Deploy to Databricks
python deploy_training_job.py --config config/training-config.yaml --cluster-id <your-cluster-id>
```

## 📁 Project Structure

```
Databricks/
├── README.md                                    # This comprehensive guide
├── fine-tuning/                                 # Core fine-tuning code
│   ├── src/
│   │   ├── bert_fine_tuning.py                 # Original BERT training script
│   │   ├── minimal_bert.py                     # Simplified BERT example
│   │   └── test_environment.py                 # Environment testing
│   ├── databricks_bert_fine_tuning.py          # Main Databricks training script
│   ├── deploy_training_job.py                  # Job deployment automation
│   ├── config/
│   │   └── training-config.yaml                # Training configuration template
│   ├── notebooks/
│   │   └── bert_fine_tuning_databricks.py      # Databricks notebook
│   ├── requirements.txt                        # Basic requirements
│   ├── requirements-databricks.txt             # Databricks-specific requirements
│   └── ...                                     # Other training files
├── infrastructure/                              # Infrastructure as Code
│   ├── cloudformation/
│   │   └── databricks-setup.yaml              # CloudFormation template
│   └── terraform/
│       └── main.tf                             # Terraform configuration
├── scripts/
│   └── setup-local-environment.sh             # Automated setup script
└── docs/
    └── databricks-vs-bedrock-comparison.md    # Detailed platform comparison
```

## 🔧 Configuration

### Amazon AWS resources required
- **S3 Buckets**: For data storage and model artifacts
- **IAM Roles**: Cross-account access for Databricks
- **VPC**: Network isolation (optional)
- **EC2 Instances**: GPU instances for training (g4dn, p3 families)

### Recommended instance types
- **Development**: g4dn.xlarge ($0.50/hour)
- **Small datasets**: g4dn.2xlarge ($1.00/hour)
- **Medium datasets**: p3.2xlarge ($3.00/hour)
- **Large datasets**: p3.8xlarge ($12.00/hour)

## Use Cases

1. **Text Classification**
   - Sentiment analysis
   - Document categorization
   - Spam detection

2. **Named Entity Recognition**
   - Custom entity extraction
   - Domain-specific NER

3. **Question Answering**
   - Domain-specific QA systems
   - Knowledge base queries

4. **Text Similarity**
   - Document matching
   - Content recommendation

## Model evaluation

The pipeline includes comprehensive evaluation metrics:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Per-class performance
- **F1-Score**: Balanced performance metric
- **Confusion Matrix**: Detailed error analysis
- **Classification Report**: Complete performance breakdown

## Deployment options

### 1. MLflow Model Serving
```python
# Register model to Unity Catalog
mlflow.register_model(model_uri, "catalog.schema.model_name")

# Deploy using MLflow
mlflow.deploy("model_name", "production")
```

### 2. Databricks model serving
- Serverless inference endpoints
- Auto-scaling based on demand
- Built-in monitoring and logging

### 3. Custom REST API
- Use provided API scripts in fine-tuning directory
- Docker deployment with nginx
- Custom scaling and monitoring

### 4. Batch inference
- Process large datasets using Spark
- Schedule regular inference jobs
- Output to data lakes or databases

## Cost optimization

### Training costs
- Use Spot instances for non-critical training (up to 70% savings)
- Implement auto-termination (120 minutes idle)
- Choose appropriate instance sizes based on data volume

### Storage costs
- Use S3 Intelligent Tiering
- Implement data lifecycle policies
- Compress training data when possible

### Inference costs
- Use quantized models for deployment
- Implement caching for frequent requests
- Consider batch inference for bulk processing

## Monitoring and logging

### MLflow integration
- Experiment tracking with parameters and metrics
- Model versioning and comparison
- Artifact storage and retrieval

### Databricks monitoring
- Cluster utilization metrics
- Job execution logs
- Performance dashboards

### AWS CloudWatch
- Infrastructure monitoring
- Cost tracking and alerts
- Custom metrics and alarms

## Amazon Spot instances with Databricks

Amazon **Spot Instances** provide significant cost savings by utilizing spare AWS compute capacity at up to 90% discount compared to On-Demand prices. Databricks seamlessly integrates with Spot Instances to optimize costs for big data analytics and machine learning workloads.

### How Databricks utilizes Spot instances?

**Hybrid Cluster Configuration**: Databricks allows you to configure clusters with a mix of On-Demand and Spot Instances, or use entirely Spot Instances for cost-sensitive workloads.

**Worker Node Optimization**: Spot Instances are typically used for worker nodes in Databricks clusters, while keeping driver nodes on On-Demand instances for stability.

### Cost savings

- **Up to 90% Cost Reduction**: Spot Instances can reduce compute costs dramatically
- **Large-Scale Workloads**: Particularly effective for batch processing, ETL jobs, and training large ML models
- **Non-Critical Workloads**: Ideal for development, testing, and fault-tolerant applications

### Configuration in Databricks

#### 1. **Cluster configuration**
```yaml
# Example cluster configuration with Spot Instances
cluster_name: "bert-training-spot"
driver_node_type_id: "i3.xlarge"          # On-Demand for stability
node_type_id: "g4dn.xlarge"               # Spot Instance type
min_workers: 1
max_workers: 8
spot_bid_price_percent: 50                # Bid 50% of On-Demand price
enable_elastic_disk: true
```

#### 2. **Mixed instance policy**
```yaml
# Recommended configuration for production workloads
instance_configuration:
  driver:
    instance_type: "i3.xlarge"
    pricing: "ON_DEMAND"                  # Stable driver
  workers:
    instance_types: ["g4dn.xlarge", "g4dn.2xlarge"]
    pricing: "SPOT"                       # Cost-effective workers
    spot_bid_price_percent: 60
    fallback_to_on_demand: true          # Fallback if Spot unavailable
```

### Best practices for Spot instances

#### **1. Fault tolerant design**
- **Checkpointing**: Regularly save model checkpoints during training
- **Auto-Recovery**: Configure clusters to automatically replace interrupted instances
- **Multiple AZs**: Distribute across Availability Zones for better availability

#### **2. Workload**
```python
# Good for Spot Instances:
- Batch processing jobs
- ETL pipelines
- Model training (with checkpointing)
- Development and testing
- Data exploration

# Avoid for Spot Instances:
- Real-time inference
- Critical production services
- Jobs without checkpointing
- Time-sensitive workloads
```

#### **3. Spot instance management**
- **Diversification**: Use multiple instance types (g4dn.xlarge, g4dn.2xlarge, p3.2xlarge)
- **Bid Strategy**: Set competitive but reasonable bid prices (50-80% of On-Demand)
- **Monitoring**: Track Spot Instance interruptions and adjust strategy

### An example

#### **Terraform configuration**
```hcl
resource "databricks_cluster" "bert_training_spot" {
  cluster_name            = "bert-training-spot-cluster"
  spark_version          = data.databricks_spark_version.ml.id
  node_type_id           = "g4dn.xlarge"
  driver_node_type_id    = "i3.xlarge"
  num_workers            = 4
  
  aws_attributes {
    instance_profile_arn    = aws_iam_instance_profile.databricks_instance_profile.arn
    availability           = "SPOT_WITH_FALLBACK"
    spot_bid_price_percent = 60
    zone_id               = "auto"
  }
  
  library {
    pypi {
      package = "transformers==4.30.0"
    }
  }
}
```

#### **Cluster creation with Databricks CLI**
```bash
# Create Spot Instance cluster
databricks clusters create --json-file cluster-config-spot.json

# cluster-config-spot.json
{
  "cluster_name": "bert-spot-training",
  "spark_version": "13.3.x-gpu-ml-scala2.12",
  "node_type_id": "g4dn.xlarge",
  "driver_node_type_id": "i3.xlarge",
  "num_workers": 4,
  "aws_attributes": {
    "availability": "SPOT_WITH_FALLBACK",
    "spot_bid_price_percent": 60,
    "ebs_volume_type": "GENERAL_PURPOSE_SSD",
    "ebs_volume_count": 1,
    "ebs_volume_size": 100
  }
}
```

### Monitoring Spot instance usage

#### **Cost tracking**
- Monitor Spot Instance savings in AWS Cost Explorer
- Set up CloudWatch alarms for unexpected cost increases
- Use Databricks cluster usage metrics

#### **Interruption handling**
```python
# Python code to handle Spot Instance interruptions
import time
import boto3

def monitor_spot_interruption():
    """Monitor for Spot Instance interruption warnings"""
    try:
        response = requests.get(
            'http://169.254.169.254/latest/meta-data/spot/instance-action',
            timeout=2
        )
        if response.status_code == 200:
            print("Spot Instance interruption detected!")
            # Save checkpoint and gracefully shutdown
            save_model_checkpoint()
            return True
    except:
        pass
    return False

# Include in training loop
for epoch in range(num_epochs):
    if monitor_spot_interruption():
        break
    # Continue training...
```

### Cost optimization

#### **1. Right sizing**
- Start with smaller instances and scale based on actual usage
- Use Databricks autoscaling to optimize cluster size
- Monitor CPU/GPU utilization metrics

#### **2. Scheduling**
- Run non-urgent jobs during off-peak hours
- Use Databricks Jobs for automated scheduling
- Leverage different AZ pricing patterns

#### **3. Instance type selection**
```bash
# Cost-effective GPU instances for BERT training
g4dn.xlarge    # Good for development/small models
g4dn.2xlarge   # Balanced performance/cost
g4dn.4xlarge   # Large models with memory requirements
p3.2xlarge     # High-performance training (when available as Spot)
```

## Troubleshooting

### Issues

1. **CUDA Out of Memory**
   ```
   Solution: Reduce batch_size or enable gradient_accumulation_steps
   ```

2. **S3 Access Denied**
   ```
   Solution: Check IAM roles and bucket policies
   ```

3. **Databricks Cluster Startup Issues**
   ```
   Solution: Verify VPC configuration and security groups
   ```

4. **Model Registration Failures**
   ```
   Solution: Ensure Unity Catalog permissions are properly configured
   ```

### Performance optimization

1. **Slow training**
   - Enable mixed precision (fp16)
   - Use larger instances with more GPUs
   - Implement gradient accumulation

2. **High costs**
   - Use Spot instances
   - Implement auto-scaling
   - Optimize data loading

3. **Memory issues**
   - Reduce sequence length
   - Use gradient checkpointing
   - Implement data streaming

**Need Help?**
- Review the detailed comparison in `docs/databricks-vs-bedrock-comparison.md`

**Start** Run `./scripts/setup-local-environment.sh` to begin your BERT fine-tuning journey!

## Resources

### Documentation
- [Databricks ML Documentation](https://docs.databricks.com/machine-learning/)
- [MLflow Documentation](https://mlflow.org/docs/latest/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

### Tutorials
- [Databricks ML Quickstart](https://docs.databricks.com/getting-started/ml-quick-start.html)
- [BERT Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
- [MLflow Tutorial](https://mlflow.org/docs/latest/tutorials-and-examples/)

### Community
- [Databricks Community](https://community.databricks.com/)
- [Hugging Face Community](https://huggingface.co/community)
- [PyTorch Forums](https://discuss.pytorch.org/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

### References

[Start using Databricks Data Intelligence Platform with AWS Marketplace](https://aws.amazon.com/blogs/awsmarketplace/start-using-databricks-data-intelligence-platform-with-aws-marketplace/)

[Tutorial: Build your first machine learning model on Databricks](https://docs.databricks.com/aws/en/getting-started/ml-get-started)

[Connect to Amazon S3](https://docs.databricks.com/aws/en/connect/storage/amazon-s3)
