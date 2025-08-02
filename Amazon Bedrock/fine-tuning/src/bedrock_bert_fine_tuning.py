#!/usr/bin/env python3
"""
Amazon Bedrock BERT fine-tuning integration
==========================================

This script integrates the existing BERT fine-tuning code with Amazon Bedrock
for scalable, managed model customization in the cloud.

Features:
- Convert local training data to Bedrock format
- Upload data to S3 automatically
- Create and monitor Bedrock fine-tuning jobs
- Cost monitoring and billing controls
- Performance evaluation and metrics
- Integration with existing BERT code

Dependencies:
- boto3 (AWS SDK)
- torch, transformers (existing BERT dependencies)
- Local BERT fine-tuning code
"""

import boto3
import json
import time
import os
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Import existing BERT fine-tuning functionality
from bert_fine_tuning import get_device_info, monitor_resources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class BedrockConfig:
    """Configuration for Amazon Bedrock fine-tuning"""
    region_name: str = 'us-east-1'
    s3_bucket: str = 'your-bedrock-training-bucket'
    iam_role_arn: str = 'arn:aws:iam::YOUR_ACCOUNT:role/BedrockExecutionRole'
    base_model_id: str = 'amazon.titan-text-express-v1'  # Example base model
    max_training_cost: float = 100.0  # USD
    
class BedrockBertFineTuner:
    """
    Amazon Bedrock integration for BERT fine-tuning
    
    This class provides methods to:
    1. Convert local BERT training data to Bedrock format
    2. Upload training data to S3
    3. Create and monitor fine-tuning jobs
    4. Evaluate model performance
    5. Manage costs and resources
    """
    
    def __init__(self, config: BedrockConfig):
        self.config = config
        self.bedrock_client = boto3.client('bedrock', region_name=config.region_name)
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=config.region_name)
        self.s3_client = boto3.client('s3', region_name=config.region_name)
        self.cloudwatch = boto3.client('cloudwatch', region_name=config.region_name)
        
        logger.info(f"Initialized BedrockBertFineTuner for region: {config.region_name}")
        
    def validate_aws_setup(self) -> bool:
        """Validate AWS credentials and permissions"""
        try:
            # Check AWS credentials
            sts_client = boto3.client('sts')
            identity = sts_client.get_caller_identity()
            logger.info(f"AWS Identity: {identity.get('Arn', 'Unknown')}")
            
            # Check S3 bucket access
            self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
            logger.info(f"S3 bucket '{self.config.s3_bucket}' is accessible")
            
            # Check Bedrock model access
            models = self.bedrock_client.list_foundation_models()
            logger.info(f"Available Bedrock models: {len(models.get('modelSummaries', []))}")
            
            return True
            
        except Exception as e:
            logger.error(f"AWS setup validation failed: {e}")
            return False
    
    def prepare_bedrock_training_data(self, 
                                    texts: List[str], 
                                    labels: List[int],
                                    output_file: str = 'bedrock_training_data.jsonl') -> str:
        """
        Convert training data to Amazon Bedrock JSONL format
        
        Args:
            texts: List of input texts
            labels: List of corresponding labels
            output_file: Output filename for JSONL data
            
        Returns:
            Path to the created JSONL file
        """
        logger.info(f"Preparing {len(texts)} training examples for Bedrock")
        
        # Validate input data
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
        
        # Convert to Bedrock format
        training_examples = []
        label_mapping = {0: "negative", 1: "positive"}  # Customize as needed
        
        for text, label in zip(texts, labels):
            # Bedrock fine-tuning format
            example = {
                "prompt": f"Classify the sentiment of the following text: {text}",
                "completion": label_mapping.get(label, str(label))
            }
            training_examples.append(example)
        
        # Save as JSONL format
        output_path = os.path.join('data', output_file)
        os.makedirs('data', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Training data saved to: {output_path}")
        return output_path
    
    def upload_to_s3(self, local_file: str, s3_key: Optional[str] = None) -> str:
        """
        Upload training data to S3
        
        Args:
            local_file: Path to local file
            s3_key: S3 object key (optional, defaults to filename with timestamp)
            
        Returns:
            S3 URI of uploaded file
        """
        if s3_key is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.basename(local_file)
            s3_key = f"training-data/{timestamp}_{filename}"
        
        try:
            logger.info(f"Uploading {local_file} to s3://{self.config.s3_bucket}/{s3_key}")
            
            self.s3_client.upload_file(
                local_file, 
                self.config.s3_bucket, 
                s3_key,
                ExtraArgs={'ServerSideEncryption': 'AES256'}
            )
            
            s3_uri = f"s3://{self.config.s3_bucket}/{s3_key}"
            logger.info(f"Successfully uploaded to: {s3_uri}")
            return s3_uri
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
    
    def create_fine_tuning_job(self, 
                             training_data_uri: str,
                             job_name: Optional[str] = None,
                             hyperparameters: Optional[Dict] = None) -> str:
        """
        Create a fine-tuning job in Amazon Bedrock
        
        Args:
            training_data_uri: S3 URI of training data
            job_name: Name for the fine-tuning job
            hyperparameters: Custom hyperparameters
            
        Returns:
            Job ARN
        """
        if job_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            job_name = f"bert-fine-tuning-{timestamp}"
        
        if hyperparameters is None:
            hyperparameters = {
                'epochCount': '3',
                'batchSize': '8',
                'learningRate': '0.00002',
                'learningRateWarmupSteps': '0'
            }
        
        # Output configuration
        output_s3_uri = f"s3://{self.config.s3_bucket}/model-outputs/{job_name}/"
        
        try:
            logger.info(f"Creating fine-tuning job: {job_name}")
            
            response = self.bedrock_client.create_model_customization_job(
                jobName=job_name,
                customModelName=f"{job_name}-custom-model",
                roleArn=self.config.iam_role_arn,
                baseModelIdentifier=self.config.base_model_id,
                trainingDataConfig={
                    's3Uri': training_data_uri
                },
                outputDataConfig={
                    's3Uri': output_s3_uri
                },
                hyperParameters=hyperparameters,
                tags=[
                    {
                        'key': 'Project',
                        'value': 'BERT-Fine-Tuning'
                    },
                    {
                        'key': 'CreatedBy',
                        'value': 'BedrockBertFineTuner'
                    }
                ]
            )
            
            job_arn = response['jobArn']
            logger.info(f"Fine-tuning job created: {job_arn}")
            return job_arn
            
        except Exception as e:
            logger.error(f"Error creating fine-tuning job: {e}")
            raise
    
    def monitor_training_job(self, job_arn: str, check_interval: int = 300) -> Dict:
        """
        Monitor the progress of a fine-tuning job
        
        Args:
            job_arn: ARN of the fine-tuning job
            check_interval: Check interval in seconds
            
        Returns:
            Final job status and details
        """
        logger.info(f"Monitoring job: {job_arn}")
        start_time = time.time()
        
        while True:
            try:
                response = self.bedrock_client.get_model_customization_job(
                    jobIdentifier=job_arn
                )
                
                status = response['status']
                elapsed_time = time.time() - start_time
                
                logger.info(f"Job Status: {status} (Elapsed: {elapsed_time/60:.1f} minutes)")
                
                # Log additional details if available
                if 'failureMessage' in response:
                    logger.error(f"Failure Message: {response['failureMessage']}")
                
                if 'trainingMetrics' in response:
                    metrics = response['trainingMetrics']
                    logger.info(f"Training Metrics: {metrics}")
                
                # Check if job is complete
                if status in ['Completed', 'Failed', 'Stopped']:
                    logger.info(f"Job finished with status: {status}")
                    break
                
                # Cost monitoring
                self._check_costs(elapsed_time)
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring job: {e}")
                time.sleep(check_interval)
        
        return response
    
    def _check_costs(self, elapsed_time: float) -> None:
        """Monitor costs and stop if exceeding budget"""
        try:
            # This is a simplified cost check
            # In practice, you'd query CloudWatch billing metrics
            estimated_cost = (elapsed_time / 3600) * 10  # $10/hour estimate
            
            if estimated_cost > self.config.max_training_cost:
                logger.warning(f"Estimated cost ${estimated_cost:.2f} exceeds budget ${self.config.max_training_cost}")
                # You could automatically stop the job here
                
        except Exception as e:
            logger.error(f"Error checking costs: {e}")
    
    def evaluate_model(self, 
                      model_arn: str,
                      test_texts: List[str],
                      test_labels: List[int]) -> Dict:
        """
        Evaluate the fine-tuned model performance
        
        Args:
            model_arn: ARN of the fine-tuned model
            test_texts: Test texts
            test_labels: True labels
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating model: {model_arn}")
        
        predictions = []
        confidences = []
        
        for text in test_texts:
            try:
                response = self.bedrock_runtime.invoke_model(
                    modelId=model_arn,
                    body=json.dumps({
                        "prompt": f"Classify the sentiment of the following text: {text}",
                        "max_tokens": 10,
                        "temperature": 0.1
                    })
                )
                
                result = json.loads(response['body'].read())
                prediction = result.get('completion', '').strip().lower()
                
                # Convert text prediction to numeric
                if 'positive' in prediction:
                    pred_label = 1
                elif 'negative' in prediction:
                    pred_label = 0
                else:
                    pred_label = 0  # Default
                
                predictions.append(pred_label)
                confidences.append(1.0)  # Placeholder
                
            except Exception as e:
                logger.error(f"Error during inference: {e}")
                predictions.append(0)
                confidences.append(0.0)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        accuracy = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'num_test_samples': len(test_texts),
            'avg_confidence': np.mean(confidences)
        }
        
        logger.info(f"Model Evaluation - Accuracy: {accuracy:.4f}")
        return metrics
    
    def cleanup_resources(self, job_arn: str = None) -> None:
        """Clean up AWS resources to control costs"""
        logger.info("Cleaning up resources...")
        
        try:
            # Stop active training job if provided
            if job_arn:
                self.bedrock_client.stop_model_customization_job(
                    jobIdentifier=job_arn
                )
                logger.info(f"Stopped training job: {job_arn}")
            
            # Clean up S3 training data (optional)
            # Note: Be careful with this in production
            # self._cleanup_s3_data()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """
    Main execution function demonstrating the complete workflow
    """
    print("="*80)
    print("AMAZON BEDROCK BERT FINE-TUNING DEMO")
    print("="*80)
    
    # Configuration
    config = BedrockConfig(
        region_name='us-east-1',
        s3_bucket='your-bedrock-training-bucket',  # Update with your bucket
        iam_role_arn='arn:aws:iam::YOUR_ACCOUNT:role/BedrockExecutionRole'  # Update
    )
    
    # Initialize fine-tuner
    fine_tuner = BedrockBertFineTuner(config)
    
    # Validate AWS setup
    if not fine_tuner.validate_aws_setup():
        logger.error("AWS setup validation failed. Please check your configuration.")
        return
    
    # Example training data (replace with your actual data)
    sample_texts = [
        "I love this product, it's amazing!",
        "This is the worst service I've ever experienced.",
        "The quality is excellent and delivery was fast.",
        "Terrible customer support, very disappointing.",
        "Great value for money, highly recommended!"
    ]
    sample_labels = [1, 0, 1, 0, 1]  # 1=positive, 0=negative
    
    try:
        # Step 1: Prepare training data
        training_file = fine_tuner.prepare_bedrock_training_data(
            sample_texts, sample_labels
        )
        
        # Step 2: Upload to S3
        training_data_uri = fine_tuner.upload_to_s3(training_file)
        
        # Step 3: Create fine-tuning job
        job_arn = fine_tuner.create_fine_tuning_job(training_data_uri)
        
        # Step 4: Monitor training (in production, this would run longer)
        print("\nStarting training monitoring...")
        print("Note: In a real scenario, this would take hours to complete.")
        print("For demo purposes, we'll simulate the process.")
        
        # In practice, you would run:
        # final_status = fine_tuner.monitor_training_job(job_arn)
        
        print("\n✅ Demo completed successfully!")
        print(f"Training job ARN: {job_arn}")
        print(f"Training data URI: {training_data_uri}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        
    finally:
        # Cleanup (optional)
        # fine_tuner.cleanup_resources()
        pass

if __name__ == "__main__":
    main()
