#!/usr/bin/env python3
"""
Complete Example: BERT fine-tuning with Amazon Bedrock
=====================================================

This example demonstrates the complete workflow for fine-tuning a BERT model
using Amazon Bedrock, from data preparation to model evaluation.

Usage:
    python bedrock_example.py --mode [prepare|train|evaluate|all]
    
Examples:
    python bedrock_example.py --mode all
    python bedrock_example.py --mode train --config config/bedrock_config.json
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from bedrock_bert_fine_tuning import BedrockBertFineTuner, BedrockConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_sample_data():
    """Load sample sentiment analysis data"""
    sample_data = [
        ("I absolutely love this product! Amazing quality and fast shipping.", 1),
        ("Terrible customer service, will never buy again.", 0),
        ("Great value for money, exactly as described.", 1),
        ("The product arrived damaged and the return process was difficult.", 0),
        ("Excellent build quality and works perfectly.", 1),
        ("Overpriced for what you get, disappointed with the purchase.", 0),
        ("Fast delivery and the product exceeded my expectations.", 1),
        ("Poor quality materials, broke after a few days.", 0),
        ("Outstanding customer support helped resolve my issue quickly.", 1),
        ("The product description was misleading, not what I expected.", 0),
        # Add more examples for better training
        ("This is the best purchase I've made this year!", 1),
        ("Complete waste of money, don't recommend.", 0),
        ("Good product, reasonable price, would buy again.", 1),
        ("Delivery was delayed and the packaging was poor.", 0),
        ("Perfect fit and excellent quality materials.", 1),
        ("Had to return due to defects, very frustrating experience.", 0),
        ("Impressed with the attention to detail and craftsmanship.", 1),
        ("The product stopped working after just one week.", 0),
        ("Great customer service and hassle-free returns.", 1),
        ("Not worth the price, quality is below average.", 0)
    ]
    
    texts, labels = zip(*sample_data)
    return list(texts), list(labels)

def prepare_data_mode(fine_tuner, texts, labels):
    """Prepare and upload training data to S3"""
    logger.info("Preparing training data for Amazon Bedrock...")
    
    # Split data for training and testing
    split_idx = int(len(texts) * 0.8)
    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    test_texts = texts[split_idx:]
    test_labels = labels[split_idx:]
    
    # Prepare training data
    training_file = fine_tuner.prepare_bedrock_training_data(
        train_texts, train_labels, 'bedrock_training_data.jsonl'
    )
    
    # Upload to S3
    training_data_uri = fine_tuner.upload_to_s3(training_file)
    
    # Save test data for later evaluation
    test_data = {
        'texts': test_texts,
        'labels': test_labels
    }
    
    os.makedirs('data', exist_ok=True)
    with open('data/test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    logger.info(f"Training data uploaded to: {training_data_uri}")
    logger.info(f"Test data saved to: data/test_data.json")
    
    return training_data_uri, test_data

def train_mode(fine_tuner, training_data_uri):
    """Start fine-tuning job in Amazon Bedrock"""
    logger.info("Starting Amazon Bedrock fine-tuning job...")
    
    # Custom hyperparameters for BERT fine-tuning
    hyperparameters = {
        'epochCount': '3',
        'batchSize': '8',
        'learningRate': '0.00002',
        'learningRateWarmupSteps': '100'
    }
    
    # Create fine-tuning job
    job_arn = fine_tuner.create_fine_tuning_job(
        training_data_uri=training_data_uri,
        job_name='bert-sentiment-analysis',
        hyperparameters=hyperparameters
    )
    
    logger.info(f"Fine-tuning job created: {job_arn}")
    
    # Save job information
    job_info = {
        'job_arn': job_arn,
        'training_data_uri': training_data_uri,
        'hyperparameters': hyperparameters,
        'status': 'started'
    }
    
    with open('config/training_job.json', 'w') as f:
        json.dump(job_info, f, indent=2)
    
    # Monitor training (in practice, this runs for hours)
    logger.info("Training job submitted. Monitor progress in AWS Console.")
    logger.info("In production, use fine_tuner.monitor_training_job(job_arn) to track progress.")
    
    return job_arn

def evaluate_mode(fine_tuner, model_arn=None):
    """Evaluate the fine-tuned model"""
    logger.info("Evaluating fine-tuned model...")
    
    # Load test data
    try:
        with open('data/test_data.json', 'r') as f:
            test_data = json.load(f)
        
        test_texts = test_data['texts']
        test_labels = test_data['labels']
        
    except FileNotFoundError:
        logger.warning("No test data found. Using sample data for evaluation.")
        test_texts, test_labels = load_sample_data()
        test_texts = test_texts[-5:]  # Use last 5 samples
        test_labels = test_labels[-5:]
    
    if model_arn is None:
        # In practice, you would get this from the completed training job
        logger.warning("No model ARN provided. Using placeholder for demonstration.")
        model_arn = "arn:aws:bedrock:us-east-1:123456789012:custom-model/your-model-id"
    
    # Evaluate model performance
    try:
        metrics = fine_tuner.evaluate_model(model_arn, test_texts, test_labels)
        
        logger.info("Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Samples: {metrics['num_test_samples']}")
        logger.info(f"Average Confidence: {metrics['avg_confidence']:.4f}")
        
        # Save evaluation results
        with open('config/evaluation_results.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.info("This is expected in demo mode without a real trained model.")

def load_config(config_path):
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return BedrockConfig(
            region_name=config_data.get('region', 'us-east-1'),
            s3_bucket=config_data.get('s3_bucket', 'your-bedrock-bucket'),
            iam_role_arn=config_data.get('execution_role_arn', 'your-role-arn'),
            base_model_id=config_data.get('base_model_id', 'amazon.titan-text-express-v1'),
            max_training_cost=config_data.get('max_training_cost', 100.0)
        )
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using default configuration.")
        return BedrockConfig()

def main():
    parser = argparse.ArgumentParser(description='BERT Fine-tuning with Amazon Bedrock')
    parser.add_argument('--mode', choices=['prepare', 'train', 'evaluate', 'all'], 
                       default='all', help='Operation mode')
    parser.add_argument('--config', default='config/bedrock_config.json',
                       help='Configuration file path')
    parser.add_argument('--training-data-uri', help='S3 URI of training data (for train mode)')
    parser.add_argument('--model-arn', help='ARN of trained model (for evaluate mode)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate setup without running training')
    
    args = parser.parse_args()
    
    print("="*80)
    print("AMAZON BEDROCK BERT FINE-TUNING EXAMPLE")
    print("="*80)
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize fine-tuner
    fine_tuner = BedrockBertFineTuner(config)
    
    # Validate setup
    if not fine_tuner.validate_aws_setup():
        logger.error("AWS setup validation failed. Please check your configuration.")
        if not args.validate_only:
            sys.exit(1)
        else:
            logger.info("Validation-only mode: continuing despite validation failure.")
    
    if args.validate_only:
        logger.info("Validation completed. Exiting.")
        return
    
    # Load sample data
    texts, labels = load_sample_data()
    
    training_data_uri = args.training_data_uri
    model_arn = args.model_arn
    
    try:
        if args.mode in ['prepare', 'all']:
            training_data_uri, test_data = prepare_data_mode(fine_tuner, texts, labels)
        
        if args.mode in ['train', 'all']:
            if not training_data_uri:
                logger.error("Training data URI is required for training mode.")
                sys.exit(1)
            model_arn = train_mode(fine_tuner, training_data_uri)
        
        if args.mode in ['evaluate', 'all']:
            evaluate_mode(fine_tuner, model_arn)
        
        print("\n" + "="*80)
        print("EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext Steps:")
        print("1. Monitor your training job in the AWS Console")
        print("2. Check S3 bucket for training data and model artifacts")
        print("3. Review CloudWatch logs for detailed training metrics")
        print("4. Set up billing alerts to monitor costs")
        print("5. Once training completes, test the model with new data")
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Example failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
