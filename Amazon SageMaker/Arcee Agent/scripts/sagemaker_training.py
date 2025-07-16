#!/usr/bin/env python3
"""
SageMaker Training Script for Arcee Agent Fine-tuning

This script handles the fine-tuning of Arcee Agent model on SageMaker
using the local dataset and LoRA configuration.
"""

import argparse
import boto3
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SageMakerTrainer:
    """
    SageMaker training job manager for Arcee Agent fine-tuning.
    """
    
    def __init__(self, region_name: str = "us-east-1"):
        """
        Initialize SageMaker trainer.
        
        Args:
            region_name: AWS region
        """
        self.region_name = region_name
        self.sagemaker = boto3.client("sagemaker", region_name=region_name)
        self.ecr = boto3.client("ecr", region_name=region_name)
        self.s3 = boto3.client("s3", region_name=region_name)
    
    def upload_dataset_to_s3(
        self,
        local_dataset_path: str,
        s3_bucket: str,
        s3_prefix: str = "arcee-agent/datasets/"
    ) -> str:
        """
        Upload local dataset to S3.
        
        Args:
            local_dataset_path: Path to local dataset directory
            s3_bucket: S3 bucket name
            s3_prefix: S3 prefix for dataset
            
        Returns:
            S3 URI of uploaded dataset
        """
        try:
            # Create bucket if it doesn't exist
            try:
                self.s3.head_bucket(Bucket=s3_bucket)
            except:
                logger.info(f"Creating S3 bucket: {s3_bucket}")
                if self.region_name == "us-east-1":
                    self.s3.create_bucket(Bucket=s3_bucket)
                else:
                    self.s3.create_bucket(
                        Bucket=s3_bucket,
                        CreateBucketConfiguration={"LocationConstraint": self.region_name}
                    )
            
            # Upload dataset files
            dataset_s3_uri = f"s3://{s3_bucket}/{s3_prefix}"
            
            for root, dirs, files in os.walk(local_dataset_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    relative_path = os.path.relpath(local_file, local_dataset_path)
                    s3_key = f"{s3_prefix}{relative_path}"
                    
                    logger.info(f"Uploading {local_file} to s3://{s3_bucket}/{s3_key}")
                    self.s3.upload_file(local_file, s3_bucket, s3_key)
            
            logger.info(f"Dataset uploaded to: {dataset_s3_uri}")
            return dataset_s3_uri
            
        except Exception as e:
            logger.error(f"Failed to upload dataset: {e}")
            raise
    
    def build_and_push_training_image(
        self,
        dockerfile_path: str = "./docker/training/Dockerfile",
        image_name: str = "arcee-agent-training",
        tag: str = "latest"
    ) -> str:
        """
        Build and push training Docker image to ECR.
        
        Args:
            dockerfile_path: Path to training Dockerfile
            image_name: Docker image name
            tag: Image tag
            
        Returns:
            ECR image URI
        """
        try:
            # Get AWS account ID
            sts = boto3.client("sts")
            account_id = sts.get_caller_identity()["Account"]
            
            # ECR repository URL
            ecr_repo = f"{account_id}.dkr.ecr.{self.region_name}.amazonaws.com/{image_name}"
            
            # Create ECR repository if it doesn't exist
            try:
                self.ecr.describe_repositories(repositoryNames=[image_name])
            except:
                logger.info(f"Creating ECR repository: {image_name}")
                self.ecr.create_repository(repositoryName=image_name)
            
            # Get ECR login token
            login_response = self.ecr.get_authorization_token()
            login_token = login_response["authorizationData"][0]["authorizationToken"]
            login_endpoint = login_response["authorizationData"][0]["proxyEndpoint"]
            
            # Build and push image (requires Docker to be installed)
            import subprocess
            
            # Login to ECR
            login_cmd = f"echo {login_token} | base64 -d | docker login --username AWS --password-stdin {login_endpoint}"
            subprocess.run(login_cmd, shell=True, check=True)
            
            # Build image
            build_cmd = f"docker build -f {dockerfile_path} -t {image_name}:{tag} ."
            subprocess.run(build_cmd, shell=True, check=True)
            
            # Tag and push image
            full_image_uri = f"{ecr_repo}:{tag}"
            tag_cmd = f"docker tag {image_name}:{tag} {full_image_uri}"
            subprocess.run(tag_cmd, shell=True, check=True)
            
            push_cmd = f"docker push {full_image_uri}"
            subprocess.run(push_cmd, shell=True, check=True)
            
            logger.info(f"Training image pushed to: {full_image_uri}")
            return full_image_uri
            
        except Exception as e:
            logger.error(f"Failed to build and push training image: {e}")
            raise
    
    def create_training_job(
        self,
        job_name: str,
        role_arn: str,
        training_image_uri: str,
        input_s3_uri: str,
        output_s3_uri: str,
        hyperparameters: Optional[Dict[str, str]] = None,
        instance_type: str = "ml.g4dn.xlarge",
        instance_count: int = 1,
        volume_size_gb: int = 30,
        max_runtime_hours: int = 24
    ) -> str:
        """
        Create SageMaker training job.
        
        Args:
            job_name: Training job name
            role_arn: SageMaker execution role ARN
            training_image_uri: ECR training image URI
            input_s3_uri: S3 URI for training data
            output_s3_uri: S3 URI for model artifacts
            hyperparameters: Training hyperparameters
            instance_type: EC2 instance type
            instance_count: Number of training instances
            volume_size_gb: EBS volume size
            max_runtime_hours: Maximum training time in hours
            
        Returns:
            Training job name
        """
        try:
            # Default hyperparameters
            if not hyperparameters:
                hyperparameters = {
                    "model_name": "arcee-ai/Arcee-Agent",
                    "dataset_name": "arcee-agent-function-calling",
                    "lora_r": "16",
                    "lora_alpha": "32",
                    "lora_dropout": "0.1",
                    "learning_rate": "2e-4",
                    "num_train_epochs": "3",
                    "per_device_train_batch_size": "1",
                    "gradient_accumulation_steps": "4",
                    "warmup_steps": "100",
                    "logging_steps": "10",
                    "save_steps": "500",
                    "max_seq_length": "2048",
                    "use_4bit": "true",
                    "bnb_4bit_compute_dtype": "float16",
                    "bnb_4bit_quant_type": "nf4"
                }
            
            # Create training job
            training_job_config = {
                "TrainingJobName": job_name,
                "RoleArn": role_arn,
                "AlgorithmSpecification": {
                    "TrainingImage": training_image_uri,
                    "TrainingInputMode": "File"
                },
                "InputDataConfig": [
                    {
                        "ChannelName": "training",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": input_s3_uri,
                                "S3DataDistributionType": "FullyReplicated"
                            }
                        },
                        "ContentType": "application/json",
                        "CompressionType": "None"
                    }
                ],
                "OutputDataConfig": {
                    "S3OutputPath": output_s3_uri
                },
                "ResourceConfig": {
                    "InstanceType": instance_type,
                    "InstanceCount": instance_count,
                    "VolumeSizeInGB": volume_size_gb
                },
                "StoppingCondition": {
                    "MaxRuntimeInSeconds": max_runtime_hours * 3600
                },
                "HyperParameters": hyperparameters
            }
            
            response = self.sagemaker.create_training_job(**training_job_config)
            logger.info(f"Created training job: {job_name}")
            
            return job_name
            
        except Exception as e:
            logger.error(f"Failed to create training job: {e}")
            raise
    
    def monitor_training_job(self, job_name: str) -> Dict[str, Any]:
        """
        Monitor training job progress.
        
        Args:
            job_name: Training job name
            
        Returns:
            Final training job status
        """
        try:
            logger.info(f"Monitoring training job: {job_name}")
            
            while True:
                response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
                status = response["TrainingJobStatus"]
                
                logger.info(f"Training job status: {status}")
                
                if status in ["Completed", "Failed", "Stopped"]:
                    break
                
                time.sleep(60)  # Check every minute
            
            # Get final details
            final_response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
            
            result = {
                "job_name": job_name,
                "status": final_response["TrainingJobStatus"],
                "creation_time": final_response["CreationTime"],
                "start_time": final_response.get("TrainingStartTime"),
                "end_time": final_response.get("TrainingEndTime"),
                "training_time_seconds": final_response.get("TrainingTimeInSeconds"),
                "billable_time_seconds": final_response.get("BillableTimeInSeconds"),
                "model_artifacts": final_response.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
                "failure_reason": final_response.get("FailureReason")
            }
            
            if result["status"] == "Completed":
                logger.info(f"Training completed successfully!")
                logger.info(f"Model artifacts: {result['model_artifacts']}")
            else:
                logger.error(f"Training failed: {result['failure_reason']}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to monitor training job: {e}")
            raise


def main():
    """
    Main function for SageMaker training.
    """
    parser = argparse.ArgumentParser(description="SageMaker training for Arcee Agent")
    
    parser.add_argument("--job-name", required=True, help="Training job name")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket for data and models")
    parser.add_argument("--dataset-path", default="./dataset", help="Local dataset path")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge", help="Training instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of training instances")
    parser.add_argument("--max-runtime-hours", type=int, default=24, help="Max training time (hours)")
    parser.add_argument("--build-image", action="store_true", help="Build and push training image")
    parser.add_argument("--monitor", action="store_true", help="Monitor training job progress")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = SageMakerTrainer(region_name=args.region)
    
    try:
        # Upload dataset to S3
        logger.info("Uploading dataset to S3...")
        input_s3_uri = trainer.upload_dataset_to_s3(
            args.dataset_path,
            args.s3_bucket,
            "arcee-agent/datasets/"
        )
        
        # Build and push training image if requested
        training_image_uri = None
        if args.build_image:
            logger.info("Building and pushing training image...")
            training_image_uri = trainer.build_and_push_training_image()
        else:
            # Use default training image (you'll need to provide this)
            account_id = boto3.client("sts").get_caller_identity()["Account"]
            training_image_uri = f"{account_id}.dkr.ecr.{args.region}.amazonaws.com/arcee-agent-training:latest"
        
        # Output S3 URI
        output_s3_uri = f"s3://{args.s3_bucket}/arcee-agent/models/"
        
        # Create training job
        logger.info("Creating training job...")
        job_name = trainer.create_training_job(
            job_name=args.job_name,
            role_arn=args.role_arn,
            training_image_uri=training_image_uri,
            input_s3_uri=input_s3_uri,
            output_s3_uri=output_s3_uri,
            instance_type=args.instance_type,
            instance_count=args.instance_count,
            max_runtime_hours=args.max_runtime_hours
        )
        
        # Monitor training job if requested
        if args.monitor:
            result = trainer.monitor_training_job(job_name)
            print(f"\nTraining job completed:")
            print(json.dumps(result, indent=2, default=str))
        else:
            print(f"Training job created: {job_name}")
            print(f"Monitor progress with: aws sagemaker describe-training-job --training-job-name {job_name}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
