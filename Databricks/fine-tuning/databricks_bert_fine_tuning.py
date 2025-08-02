#!/usr/bin/env python3
"""
Databricks BERT fine-tuning workflow
====================================

This script demonstrates how to fine-tune a BERT model on Databricks
with MLflow tracking, quantization, and distributed training support.

Key Features:
- MLflow experiment tracking and model registration
- GPU optimization and memory management
- Quantization for deployment efficiency
- Integration with Unity Catalog
- S3 data loading and model storage

Usage:
    python databricks_bert_fine_tuning.py --config config.yaml
"""

import os
import json
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.quantization import quantize_dynamic
import transformers
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
    AdamW,
    get_scheduler
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
import pandas as pd

# MLflow imports (for Databricks)
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available. Tracking will be disabled.")

# Databricks utilities (only available in Databricks environment)
try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("PySpark not available. Running in local mode.")


class DatabricksConfig:
    """Configuration class for Databricks BERT fine-tuning"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self._validate_config()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from YAML file or use defaults"""
        default_config = {
            'model': {
                'model_name': 'bert-base-uncased',
                'num_labels': 2,
                'max_length': 128,
                'quantize': True
            },
            'training': {
                'num_epochs': 3,
                'batch_size': 16,
                'learning_rate': 2e-5,
                'warmup_steps': 500,
                'weight_decay': 0.01,
                'gradient_accumulation_steps': 1,
                'fp16': True
            },
            'data': {
                'train_path': 's3://your-bucket/data/train.jsonl',
                'test_path': 's3://your-bucket/data/test.jsonl',
                'text_column': 'text',
                'label_column': 'label'
            },
            'mlflow': {
                'experiment_name': '/Users/user@company.com/bert-fine-tuning',
                'run_name': 'bert-classification',
                'model_name': 'bert_sentiment_classifier'
            },
            'databricks': {
                'catalog_name': 'ml_models',
                'schema_name': 'bert_experiments'
            },
            'output': {
                'model_path': 's3://your-bucket/models/',
                'log_path': 's3://your-bucket/logs/'
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Merge user config with defaults
            self._deep_update(default_config, user_config)
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> None:
        """Recursively update nested dictionary"""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        required_keys = ['model', 'training', 'data']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration section: {key}")


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
        
        # Tokenize text
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


class DatabricksBertTrainer:
    """Main trainer class for BERT fine-tuning on Databricks"""
    
    def __init__(self, config: DatabricksConfig):
        self.config = config.config
        self.device = self._get_device()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        
    def _get_device(self) -> torch.device:
        """Get the best available device"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            self.logger.warning("CUDA not available. Using CPU.")
        
        return device
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_data(self) -> None:
        """Load training and test data"""
        self.logger.info("Loading training data...")
        
        if SPARK_AVAILABLE and self.config['data']['train_path'].startswith('s3://'):
            # Use Spark for loading large datasets from S3
            self._load_data_spark()
        else:
            # Local loading for testing
            self._load_data_local()
    
    def _load_data_spark(self) -> None:
        """Load data using Spark (for Databricks environment)"""
        spark = SparkSession.builder.appName("BERTDataLoading").getOrCreate()
        
        # Read training data
        train_df = spark.read.json(self.config['data']['train_path'])
        train_pd = train_df.toPandas()
        
        # Read test data
        test_df = spark.read.json(self.config['data']['test_path'])
        test_pd = test_df.toPandas()
        
        # Extract texts and labels
        text_col = self.config['data']['text_column']
        label_col = self.config['data']['label_column']
        
        self.train_texts = train_pd[text_col].tolist()
        self.train_labels = train_pd[label_col].tolist()
        self.test_texts = test_pd[text_col].tolist()
        self.test_labels = test_pd[label_col].tolist()
        
        self.logger.info(f"Loaded {len(self.train_texts)} training samples")
        self.logger.info(f"Loaded {len(self.test_texts)} test samples")
    
    def _load_data_local(self) -> None:
        """Load sample data for local testing"""
        # Sample data for demonstration
        self.train_texts = [
            "I love this product! It's amazing and works perfectly.",
            "This is terrible. Complete waste of money.",
            "Great quality and fast shipping. Highly recommend!",
            "Poor quality control. Item arrived damaged.",
            "Excellent customer service and product quality.",
            "Not what I expected. Very disappointed.",
            "Outstanding value for the price point.",
            "Defective item with poor build quality."
        ]
        
        self.train_labels = [1, 0, 1, 0, 1, 0, 1, 0]
        
        self.test_texts = [
            "Outstanding product! Exceeded my expectations.",
            "Horrible experience. Would not recommend."
        ]
        
        self.test_labels = [1, 0]
        
        self.logger.info("Using sample data for local testing")
    
    def prepare_model(self) -> None:
        """Initialize tokenizer and model"""
        self.logger.info("Loading BERT tokenizer and model...")
        
        model_name = self.config['model']['model_name']
        num_labels = self.config['model']['num_labels']
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Load model
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Move to device
        self.model.to(self.device)
        
        self.logger.info(f"Model loaded: {model_name}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def prepare_datasets(self) -> None:
        """Create PyTorch datasets"""
        self.logger.info("Preparing datasets...")
        
        max_length = self.config['model']['max_length']
        
        self.train_dataset = TextClassificationDataset(
            self.train_texts, self.train_labels, self.tokenizer, max_length
        )
        
        self.test_dataset = TextClassificationDataset(
            self.test_texts, self.test_labels, self.tokenizer, max_length
        )
        
        self.logger.info(f"Train dataset size: {len(self.train_dataset)}")
        self.logger.info(f"Test dataset size: {len(self.test_dataset)}")
    
    def train(self) -> Dict[str, float]:
        """Train the model with MLflow tracking"""
        self.logger.info("Starting BERT fine-tuning...")
        
        # Start MLflow run
        if MLFLOW_AVAILABLE:
            # Configure MLflow for Unity Catalog
            mlflow.set_registry_uri("databricks-uc")
            
            # Start experiment
            experiment_name = self.config['mlflow']['experiment_name']
            try:
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if experiment is None:
                    mlflow.create_experiment(experiment_name)
            except Exception as e:
                self.logger.warning(f"Could not create experiment: {e}")
        
        with mlflow.start_run(run_name=self.config['mlflow']['run_name']) if MLFLOW_AVAILABLE else self._dummy_context():
            # Log parameters
            if MLFLOW_AVAILABLE:
                self._log_parameters()
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir='./results',
                num_train_epochs=self.config['training']['num_epochs'],
                per_device_train_batch_size=self.config['training']['batch_size'],
                per_device_eval_batch_size=self.config['training']['batch_size'],
                warmup_steps=self.config['training']['warmup_steps'],
                weight_decay=self.config['training']['weight_decay'],
                logging_dir='./logs',
                logging_steps=10,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="eval_accuracy",
                greater_is_better=True,
                fp16=self.config['training'].get('fp16', False),
                gradient_accumulation_steps=self.config['training'].get('gradient_accumulation_steps', 1),
                report_to="none"  # Disable wandb
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.test_dataset,
                compute_metrics=self._compute_metrics,
            )
            
            # Train the model
            start_time = time.time()
            trainer.train()
            training_time = time.time() - start_time
            
            # Evaluate the model
            eval_results = trainer.evaluate()
            
            # Log metrics
            if MLFLOW_AVAILABLE:
                mlflow.log_metric("training_time", training_time)
                for key, value in eval_results.items():
                    mlflow.log_metric(key, value)
            
            # Quantize model if specified
            if self.config['model'].get('quantize', False):
                self.logger.info("Quantizing model...")
                quantized_model = self._quantize_model(self.model)
                
                if MLFLOW_AVAILABLE:
                    mlflow.pytorch.log_model(quantized_model, "quantized_model")
            
            # Log model
            if MLFLOW_AVAILABLE:
                mlflow.pytorch.log_model(self.model, "bert_model")
                
                # Register model to Unity Catalog
                self._register_model()
            
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            self.logger.info(f"Final evaluation results: {eval_results}")
            
            return eval_results
    
    def _log_parameters(self) -> None:
        """Log training parameters to MLflow"""
        mlflow.log_param("model_name", self.config['model']['model_name'])
        mlflow.log_param("num_labels", self.config['model']['num_labels'])
        mlflow.log_param("max_length", self.config['model']['max_length'])
        mlflow.log_param("num_epochs", self.config['training']['num_epochs'])
        mlflow.log_param("batch_size", self.config['training']['batch_size'])
        mlflow.log_param("learning_rate", self.config['training']['learning_rate'])
        mlflow.log_param("warmup_steps", self.config['training']['warmup_steps'])
        mlflow.log_param("weight_decay", self.config['training']['weight_decay'])
        mlflow.log_param("device", str(self.device))
        mlflow.log_param("quantize", self.config['model'].get('quantize', False))
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
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
    
    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to the model"""
        quantized_model = quantize_dynamic(
            model, 
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        
        # Calculate model sizes
        original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024 / 1024
        
        self.logger.info(f"Original model size: {original_size:.2f} MB")
        self.logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        self.logger.info(f"Size reduction: {(1 - quantized_size/original_size)*100:.1f}%")
        
        if MLFLOW_AVAILABLE:
            mlflow.log_metric("original_model_size_mb", original_size)
            mlflow.log_metric("quantized_model_size_mb", quantized_size)
            mlflow.log_metric("size_reduction_percent", (1 - quantized_size/original_size)*100)
        
        return quantized_model
    
    def _register_model(self) -> None:
        """Register model to Unity Catalog"""
        try:
            catalog_name = self.config['databricks']['catalog_name']
            schema_name = self.config['databricks']['schema_name']
            model_name = self.config['mlflow']['model_name']
            
            full_model_name = f"{catalog_name}.{schema_name}.{model_name}"
            
            # Register model
            model_version = mlflow.register_model(
                f"runs:/{mlflow.active_run().info.run_id}/bert_model",
                full_model_name
            )
            
            self.logger.info(f"Model registered: {full_model_name} (version {model_version.version})")
            
        except Exception as e:
            self.logger.warning(f"Could not register model to Unity Catalog: {e}")
    
    def _dummy_context(self):
        """Dummy context manager when MLflow is not available"""
        class DummyContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return DummyContext()
    
    def evaluate_model(self) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        self.logger.info("Running comprehensive model evaluation...")
        
        # Create DataLoader for evaluation
        eval_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )
        
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                pred_labels = torch.argmax(logits, dim=1)
                predictions.extend(pred_labels.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Generate classification report
        class_report = classification_report(true_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        self.logger.info(f"Evaluation Results: {results}")
        self.logger.info(f"Classification Report:\n{class_report}")
        
        return results


def main():
    """Main function to run BERT fine-tuning"""
    parser = argparse.ArgumentParser(description='BERT Fine-tuning on Databricks')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--local', action='store_true', help='Run in local mode with sample data')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = DatabricksConfig(args.config)
    
    # Initialize trainer
    trainer = DatabricksBertTrainer(config)
    
    try:
        # Load data
        trainer.load_data()
        
        # Prepare model and datasets
        trainer.prepare_model()
        trainer.prepare_datasets()
        
        # Train the model
        results = trainer.train()
        
        # Additional evaluation
        eval_results = trainer.evaluate_model()
        
        print(f"\n🎉 Training completed successfully!")
        print(f"Final Results: {results}")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
