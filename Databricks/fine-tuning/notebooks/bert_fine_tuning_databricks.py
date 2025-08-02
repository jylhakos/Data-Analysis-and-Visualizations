# Databricks notebook source
# MAGIC %md
# MAGIC # BERT Fine-tuning on Databricks
# MAGIC 
# MAGIC This notebook demonstrates how to fine-tune a BERT model for text classification using:
# MAGIC - **Databricks Runtime ML** for the compute environment
# MAGIC - **MLflow** for experiment tracking and model management
# MAGIC - **Unity Catalog** for model registry
# MAGIC - **S3** for data storage
# MAGIC - **GPU acceleration** for faster training

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup and Package Installation

# COMMAND ----------

# Install required packages
%pip install transformers>=4.30.0 torch>=2.0.0 accelerate>=0.20.0 datasets

# COMMAND ----------

# Restart Python to ensure packages are properly loaded
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import Libraries and Configure MLflow

# COMMAND ----------

import mlflow
import mlflow.pytorch
import torch
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    AdamW
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, List, Tuple

# Configure MLflow for Unity Catalog
mlflow.set_registry_uri("databricks-uc")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Configuration Parameters

# COMMAND ----------

# Configuration - modify these parameters for your specific use case
CONFIG = {
    'model': {
        'model_name': 'bert-base-uncased',
        'num_labels': 2,
        'max_length': 128
    },
    'training': {
        'num_epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'warmup_steps': 500,
        'weight_decay': 0.01
    },
    'data': {
        'train_path': 's3://your-bucket/data/train.jsonl',  # Update with your S3 path
        'test_path': 's3://your-bucket/data/test.jsonl',    # Update with your S3 path
        'text_column': 'text',
        'label_column': 'label'
    },
    'mlflow': {
        'experiment_name': f'/Users/{spark.sql("SELECT current_user()").collect()[0][0]}/bert-fine-tuning',
        'run_name': 'bert-classification-databricks'
    },
    'unity_catalog': {
        'catalog_name': 'main',  # Update with your catalog
        'schema_name': 'default',  # Update with your schema
        'model_name': 'bert_sentiment_classifier'
    }
}

print("Configuration loaded:")
print(json.dumps(CONFIG, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Data Loading and Preprocessing

# COMMAND ----------

class TextClassificationDataset(Dataset):
    """Custom Dataset for text classification with BERT tokenization"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# COMMAND ----------

def load_data_from_s3(s3_path: str, use_sample_data: bool = True) -> pd.DataFrame:
    """Load data from S3 or use sample data for demo"""
    
    if use_sample_data:
        # Sample data for demonstration
        print("🔄 Using sample data for demonstration")
        
        sample_data = [
            {"text": "I absolutely love this product! It's amazing and works perfectly.", "label": 1},
            {"text": "This is terrible. Complete waste of money and time.", "label": 0},
            {"text": "Great quality and fast shipping. Highly recommend to everyone!", "label": 1},
            {"text": "Poor quality control. Item arrived damaged and unusable.", "label": 0},
            {"text": "Excellent customer service and outstanding product quality.", "label": 1},
            {"text": "Not what I expected. Very disappointed with this purchase.", "label": 0},
            {"text": "Outstanding value for the price point. Exceeded expectations.", "label": 1},
            {"text": "Defective item with poor build quality. Requesting refund.", "label": 0},
            {"text": "Perfect! Exactly what I was looking for. Five stars!", "label": 1},
            {"text": "Horrible experience. Would not recommend to anyone.", "label": 0}
        ]
        
        return pd.DataFrame(sample_data)
    
    else:
        # Load from S3
        print(f"📥 Loading data from {s3_path}")
        try:
            # Copy from S3 to local file system
            local_path = f"/tmp/{s3_path.split('/')[-1]}"
            dbutils.fs.cp(s3_path, f"file://{local_path}")
            
            # Read JSON Lines file
            data = []
            with open(local_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            
            return pd.DataFrame(data)
            
        except Exception as e:
            print(f"❌ Error loading from S3: {e}")
            print("Falling back to sample data")
            return load_data_from_s3("", use_sample_data=True)

# Load training and test data
train_df = load_data_from_s3(CONFIG['data']['train_path'], use_sample_data=True)
test_df = load_data_from_s3(CONFIG['data']['test_path'], use_sample_data=True)

print(f"📊 Training samples: {len(train_df)}")
print(f"📊 Test samples: {len(test_df)}")

# Display sample data
print("\nSample training data:")
display(train_df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model and Tokenizer Initialization

# COMMAND ----------

# Initialize tokenizer and model
model_name = CONFIG['model']['model_name']
num_labels = CONFIG['model']['num_labels']
max_length = CONFIG['model']['max_length']

print(f"🤖 Loading {model_name}...")

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False
)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(f"Model loaded and moved to {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Dataset Preparation

# COMMAND ----------

# Prepare datasets
text_col = CONFIG['data']['text_column']
label_col = CONFIG['data']['label_column']

train_texts = train_df[text_col].tolist()
train_labels = train_df[label_col].tolist()
test_texts = test_df[text_col].tolist()
test_labels = test_df[label_col].tolist()

# Create datasets
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, max_length)

print(f" Datasets created:")
print(f"   Training dataset: {len(train_dataset)} samples")
print(f"   Test dataset: {len(test_dataset)} samples")

# Show tokenization example
sample_text = train_texts[0]
sample_tokens = tokenizer(sample_text, truncation=True, padding='max_length', max_length=max_length)
print(f"\nTokenization example:")
print(f"Original text: {sample_text}")
print(f"Tokenized (first 10 tokens): {sample_tokens['input_ids'][:10]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Training Configuration and MLflow Setup

# COMMAND ----------

def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=CONFIG['training']['num_epochs'],
    per_device_train_batch_size=CONFIG['training']['batch_size'],
    per_device_eval_batch_size=CONFIG['training']['batch_size'],
    warmup_steps=CONFIG['training']['warmup_steps'],
    weight_decay=CONFIG['training']['weight_decay'],
    learning_rate=CONFIG['training']['learning_rate'],
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),  # Enable mixed precision on GPU
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="none"  # Disable wandb
)

print(" Training configuration ready")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Mixed precision: {training_args.fp16}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Model Training with MLflow Tracking

# COMMAND ----------

# Create or get MLflow experiment
experiment_name = CONFIG['mlflow']['experiment_name']
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"📊 Created new experiment: {experiment_name}")
    else:
        experiment_id = experiment.experiment_id
        print(f"📊 Using existing experiment: {experiment_name}")
except Exception as e:
    print(f"⚠️ Could not create/access experiment: {e}")
    experiment_name = None

# Start MLflow run
with mlflow.start_run(run_name=CONFIG['mlflow']['run_name'], experiment_id=experiment_id):
    print(" Starting BERT fine-tuning with MLflow tracking...")
    
    # Log parameters
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("num_labels", num_labels)
    mlflow.log_param("max_length", max_length)
    mlflow.log_param("num_epochs", training_args.num_train_epochs)
    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("learning_rate", training_args.learning_rate)
    mlflow.log_param("warmup_steps", training_args.warmup_steps)
    mlflow.log_param("weight_decay", training_args.weight_decay)
    mlflow.log_param("device", str(device))
    mlflow.log_param("fp16", training_args.fp16)
    mlflow.log_param("train_samples", len(train_dataset))
    mlflow.log_param("test_samples", len(test_dataset))
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Record training start time
    start_time = time.time()
    
    # Train the model
    print(" Training started...")
    training_result = trainer.train()
    
    # Calculate training time
    training_time = time.time() - start_time
    mlflow.log_metric("training_time_seconds", training_time)
    
    print(f" Training completed in {training_time:.2f} seconds")
    
    # Evaluate the model
    print(" Evaluating model...")
    eval_results = trainer.evaluate()
    
    # Log evaluation metrics
    for key, value in eval_results.items():
        mlflow.log_metric(key, value)
    
    print("Evaluation Results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Generate detailed predictions for analysis
    print(" Generating detailed predictions...")
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    # Create classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Log additional metrics
    mlflow.log_metric("test_accuracy", class_report['accuracy'])
    mlflow.log_metric("macro_avg_precision", class_report['macro avg']['precision'])
    mlflow.log_metric("macro_avg_recall", class_report['macro avg']['recall'])
    mlflow.log_metric("macro_avg_f1", class_report['macro avg']['f1-score'])
    
    # Save classification report as artifact
    report_text = classification_report(y_true, y_pred)
    with open("/tmp/classification_report.txt", "w") as f:
        f.write(report_text)
    mlflow.log_artifact("/tmp/classification_report.txt")
    
    # Log model to MLflow
    print(" Logging model to MLflow...")
    mlflow.pytorch.log_model(
        model, 
        "bert_model",
        conda_env={
            'channels': ['defaults', 'conda-forge', 'pytorch'],
            'dependencies': [
                'python=3.8',
                'pytorch>=2.0.0',
                'transformers>=4.30.0',
                {'pip': ['accelerate>=0.20.0']}
            ],
            'name': 'bert_env'
        }
    )
    
    # Register model to Unity Catalog
    try:
        catalog_name = CONFIG['unity_catalog']['catalog_name']
        schema_name = CONFIG['unity_catalog']['schema_name']
        model_name_uc = CONFIG['unity_catalog']['model_name']
        
        full_model_name = f"{catalog_name}.{schema_name}.{model_name_uc}"
        
        print(f"🏛️ Registering model to Unity Catalog: {full_model_name}")
        
        model_version = mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/bert_model",
            full_model_name
        )
        
        print(f" Model registered as {full_model_name} version {model_version.version}")
        mlflow.log_param("registered_model_name", full_model_name)
        mlflow.log_param("model_version", model_version.version)
        
    except Exception as e:
        print(f"⚠️ Could not register model to Unity Catalog: {e}")
    
    # Get the MLflow run info
    run_info = mlflow.active_run().info
    print(f"\n Training completed successfully!")
    print(f" MLflow Run ID: {run_info.run_id}")
    print(f" View in MLflow UI: {mlflow.get_tracking_uri()}/#/experiments/{run_info.experiment_id}/runs/{run_info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Model Testing and Inference Examples

# COMMAND ----------

# Test the trained model with custom examples
def predict_sentiment(text: str, model, tokenizer, device) -> Dict:
    """Predict sentiment for a given text"""
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    return {
        'text': text,
        'predicted_class': predicted_class,
        'predicted_label': 'Positive' if predicted_class == 1 else 'Negative',
        'confidence': confidence,
        'probabilities': {
            'Negative': predictions[0][0].item(),
            'Positive': predictions[0][1].item()
        }
    }

# Test examples
test_examples = [
    "This product is absolutely fantastic! I love it!",
    "Terrible quality. Completely disappointed.",
    "It's okay, nothing special but does the job.",
    "Best purchase I've made this year. Highly recommended!",
    "Poor customer service and low quality product."
]

print(" Testing model with example texts:")
print("=" * 60)

for i, text in enumerate(test_examples, 1):
    result = predict_sentiment(text, model, tokenizer, device)
    print(f"\nExample {i}:")
    print(f"Text: {result['text']}")
    print(f"Prediction: {result['predicted_label']} (confidence: {result['confidence']:.3f})")
    print(f"Probabilities: Negative={result['probabilities']['Negative']:.3f}, Positive={result['probabilities']['Positive']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Model Performance Analysis

# COMMAND ----------

# Create confusion matrix and performance visualizations
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Generate predictions for the entire test set
model.eval()
all_predictions = []
all_labels = []

test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.numpy())

# Create confusion matrix
cm = confusion_matrix(all_labels, all_predictions)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

# Save plot
plt.savefig('/tmp/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate and display detailed metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_predictions, average=None)

print("\n📊 Detailed Performance Metrics:")
print("=" * 50)
print(f"Overall Accuracy: {accuracy:.4f}")
print("\nPer-class metrics:")
for i, label in enumerate(['Negative', 'Positive']):
    print(f"{label}:")
    print(f"  Precision: {precision[i]:.4f}")
    print(f"  Recall: {recall[i]:.4f}")
    print(f"  F1-score: {f1[i]:.4f}")
    print(f"  Support: {support[i]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Model Deployment Preparation

# COMMAND ----------

# Prepare model for deployment by applying quantization
from torch.quantization import quantize_dynamic

print("⚡ Applying dynamic quantization for deployment...")

# Apply quantization
quantized_model = quantize_dynamic(
    model.cpu(),  # Move to CPU for quantization
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Calculate model sizes
def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    return (param_size + buffer_size) / 1024 / 1024

original_size = get_model_size(model)
quantized_size = get_model_size(quantized_model)

print(f" Model size comparison:")
print(f"  Original model: {original_size:.2f} MB")
print(f"  Quantized model: {quantized_size:.2f} MB")
print(f"  Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")

# Test quantized model performance
print("\n Testing quantized model...")
sample_text = "This is a great product!"
original_result = predict_sentiment(sample_text, model.to(device), tokenizer, device)
quantized_result = predict_sentiment(sample_text, quantized_model, tokenizer, torch.device('cpu'))

print(f"Original model prediction: {original_result['predicted_label']} ({original_result['confidence']:.3f})")
print(f"Quantized model prediction: {quantized_result['predicted_label']} ({quantized_result['confidence']:.3f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Summary and Next Steps

# COMMAND ----------

print(" BERT Fine-tuning Completed Successfully!")
print("=" * 60)
print(f"✅ Model: {CONFIG['model']['model_name']}")
print(f" Training samples: {len(train_dataset)}")
print(f" Test samples: {len(test_dataset)}")
print(f" Final accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
print(f" Training time: {training_time:.2f} seconds")
print(f" Model registered in Unity Catalog")

print("\n Next Steps:")
print("1.  Replace sample data with your actual dataset")
print("2. ⚙️ Tune hyperparameters for better performance")
print("3. Deploy model using MLflow Model Serving")
print("4. Set up monitoring and logging for production")
print("5. Schedule regular retraining with new data")

print(f"\n Resources:")
print(f"   MLflow Experiment: {experiment_name}")
if 'full_model_name' in locals():
    print(f"   Unity Catalog Model: {full_model_name}")
print(f"   Databricks Runtime: {spark.conf.get('spark.databricks.clusterUsageTags.sparkVersion')}")

print("\n Tips:")
print("   - Use larger GPU instances (p3.8xlarge) for bigger datasets")
print("   - Enable gradient accumulation for larger effective batch sizes")
print("   - Use mixed precision training (fp16) for faster training")
print("   - Monitor GPU memory usage and adjust batch size accordingly")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Additional Resources
# MAGIC
# MAGIC ### Documentation Links:
# MAGIC - [Databricks Runtime ML](https://docs.databricks.com/runtime/mlruntime.html)
# MAGIC - [MLflow on Databricks](https://docs.databricks.com/mlflow/index.html)
# MAGIC - [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html)
# MAGIC - [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
# MAGIC
# MAGIC ### Model Deployment Options:
# MAGIC 1. **MLflow Model Serving** - Built-in model serving in Databricks
# MAGIC 2. **Databricks Model Serving** - Serverless inference endpoints
# MAGIC 3. **Custom REST API** - Deploy using the provided API scripts
# MAGIC 4. **Batch Inference** - Process large datasets using Spark
# MAGIC
# MAGIC ### Performance Optimization:
# MAGIC - Use **GPU-optimized clusters** for training
# MAGIC - Enable **automatic scaling** for variable workloads
# MAGIC - Implement **data caching** for faster I/O
# MAGIC - Use **Delta Lake** for versioned datasets
# MAGIC - Apply **gradient checkpointing** for memory efficiency
