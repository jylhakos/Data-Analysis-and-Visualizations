#!/usr/bin/env python3
"""
SageMaker Model Deployment Script for Arcee Agent

This script handles deploying the fine-tuned Arcee Agent model to 
SageMaker endpoints for real-time inference.
"""

import argparse
import boto3
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SageMakerDeployer:
    """
    SageMaker model deployment manager for Arcee Agent.
    """
    
    def __init__(self, region_name: str = "us-east-1"):
        """
        Initialize SageMaker deployer.
        
        Args:
            region_name: AWS region
        """
        self.region_name = region_name
        self.sagemaker = boto3.client("sagemaker", region_name=region_name)
        self.ecr = boto3.client("ecr", region_name=region_name)
    
    def create_model(
        self,
        model_name: str,
        role_arn: str,
        model_artifacts_s3_uri: str,
        inference_image_uri: Optional[str] = None,
        environment: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create SageMaker model.
        
        Args:
            model_name: Model name
            role_arn: SageMaker execution role ARN
            model_artifacts_s3_uri: S3 URI of model artifacts
            inference_image_uri: ECR inference image URI
            environment: Environment variables
            
        Returns:
            Model name
        """
        try:
            # Use default inference image if not provided
            if not inference_image_uri:
                account_id = boto3.client("sts").get_caller_identity()["Account"]
                inference_image_uri = f"{account_id}.dkr.ecr.{self.region_name}.amazonaws.com/arcee-agent-inference:latest"
            
            # Default environment variables
            if not environment:
                environment = {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
                    "MODEL_SERVER_TIMEOUT": "300",
                    "MODEL_SERVER_WORKERS": "1"
                }
            
            # Create model configuration
            model_config = {
                "ModelName": model_name,
                "ExecutionRoleArn": role_arn,
                "PrimaryContainer": {
                    "Image": inference_image_uri,
                    "ModelDataUrl": model_artifacts_s3_uri,
                    "Environment": environment
                }
            }
            
            response = self.sagemaker.create_model(**model_config)
            logger.info(f"Created model: {model_name}")
            
            return model_name
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
    
    def create_endpoint_config(
        self,
        config_name: str,
        model_name: str,
        instance_type: str = "ml.m5.large",
        initial_instance_count: int = 1,
        variant_name: str = "AllTraffic",
        initial_variant_weight: int = 1
    ) -> str:
        """
        Create SageMaker endpoint configuration.
        
        Args:
            config_name: Endpoint configuration name
            model_name: Model name
            instance_type: EC2 instance type
            initial_instance_count: Initial number of instances
            variant_name: Production variant name
            initial_variant_weight: Initial variant weight
            
        Returns:
            Endpoint configuration name
        """
        try:
            config = {
                "EndpointConfigName": config_name,
                "ProductionVariants": [
                    {
                        "VariantName": variant_name,
                        "ModelName": model_name,
                        "InitialInstanceCount": initial_instance_count,
                        "InstanceType": instance_type,
                        "InitialVariantWeight": initial_variant_weight
                    }
                ]
            }
            
            response = self.sagemaker.create_endpoint_config(**config)
            logger.info(f"Created endpoint config: {config_name}")
            
            return config_name
            
        except Exception as e:
            logger.error(f"Failed to create endpoint config: {e}")
            raise
    
    def create_endpoint(
        self,
        endpoint_name: str,
        config_name: str,
        tags: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Create SageMaker endpoint.
        
        Args:
            endpoint_name: Endpoint name
            config_name: Endpoint configuration name
            tags: Resource tags
            
        Returns:
            Endpoint name
        """
        try:
            endpoint_config = {
                "EndpointName": endpoint_name,
                "EndpointConfigName": config_name
            }
            
            if tags:
                endpoint_config["Tags"] = tags
            
            response = self.sagemaker.create_endpoint(**endpoint_config)
            logger.info(f"Created endpoint: {endpoint_name}")
            
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Failed to create endpoint: {e}")
            raise
    
    def wait_for_endpoint(self, endpoint_name: str, timeout_minutes: int = 30) -> Dict[str, Any]:
        """
        Wait for endpoint to be in service.
        
        Args:
            endpoint_name: Endpoint name
            timeout_minutes: Timeout in minutes
            
        Returns:
            Endpoint details
        """
        try:
            logger.info(f"Waiting for endpoint {endpoint_name} to be in service...")
            
            start_time = time.time()
            timeout_seconds = timeout_minutes * 60
            
            while True:
                response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
                status = response["EndpointStatus"]
                
                logger.info(f"Endpoint status: {status}")
                
                if status == "InService":
                    logger.info(f"Endpoint {endpoint_name} is now in service!")
                    return response
                elif status in ["Failed", "OutOfService"]:
                    failure_reason = response.get("FailureReason", "Unknown")
                    raise Exception(f"Endpoint deployment failed: {failure_reason}")
                
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    raise Exception(f"Endpoint deployment timed out after {timeout_minutes} minutes")
                
                time.sleep(30)  # Check every 30 seconds
                
        except Exception as e:
            logger.error(f"Failed to wait for endpoint: {e}")
            raise
    
    def update_endpoint(
        self,
        endpoint_name: str,
        new_config_name: str,
        retain_all_variant_properties: bool = False
    ) -> str:
        """
        Update existing endpoint with new configuration.
        
        Args:
            endpoint_name: Endpoint name
            new_config_name: New endpoint configuration name
            retain_all_variant_properties: Whether to retain variant properties
            
        Returns:
            Endpoint name
        """
        try:
            update_config = {
                "EndpointName": endpoint_name,
                "EndpointConfigName": new_config_name,
                "RetainAllVariantProperties": retain_all_variant_properties
            }
            
            response = self.sagemaker.update_endpoint(**update_config)
            logger.info(f"Updated endpoint: {endpoint_name}")
            
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Failed to update endpoint: {e}")
            raise
    
    def delete_endpoint(self, endpoint_name: str) -> None:
        """
        Delete SageMaker endpoint.
        
        Args:
            endpoint_name: Endpoint name
        """
        try:
            self.sagemaker.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Deleted endpoint: {endpoint_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete endpoint: {e}")
            raise
    
    def delete_endpoint_config(self, config_name: str) -> None:
        """
        Delete SageMaker endpoint configuration.
        
        Args:
            config_name: Endpoint configuration name
        """
        try:
            self.sagemaker.delete_endpoint_config(EndpointConfigName=config_name)
            logger.info(f"Deleted endpoint config: {config_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete endpoint config: {e}")
            raise
    
    def delete_model(self, model_name: str) -> None:
        """
        Delete SageMaker model.
        
        Args:
            model_name: Model name
        """
        try:
            self.sagemaker.delete_model(ModelName=model_name)
            logger.info(f"Deleted model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            raise
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all SageMaker endpoints.
        
        Returns:
            List of endpoint information
        """
        try:
            response = self.sagemaker.list_endpoints()
            endpoints = []
            
            for endpoint in response["Endpoints"]:
                endpoint_details = self.sagemaker.describe_endpoint(
                    EndpointName=endpoint["EndpointName"]
                )
                endpoints.append({
                    "name": endpoint["EndpointName"],
                    "status": endpoint["EndpointStatus"],
                    "creation_time": endpoint["CreationTime"],
                    "last_modified_time": endpoint["LastModifiedTime"],
                    "config_name": endpoint_details["EndpointConfigName"],
                    "production_variants": endpoint_details["ProductionVariants"]
                })
            
            return endpoints
            
        except Exception as e:
            logger.error(f"Failed to list endpoints: {e}")
            raise


def main():
    """
    Main function for SageMaker model deployment.
    """
    parser = argparse.ArgumentParser(description="Deploy Arcee Agent model to SageMaker")
    
    parser.add_argument("--model-name", required=True, help="Model name")
    parser.add_argument("--endpoint-name", required=True, help="Endpoint name")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--model-artifacts-uri", required=True, help="S3 URI of model artifacts")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--instance-type", default="ml.m5.large", help="Instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Initial instance count")
    parser.add_argument("--timeout-minutes", type=int, default=30, help="Deployment timeout (minutes)")
    parser.add_argument("--inference-image", help="Custom inference image URI")
    parser.add_argument("--update", action="store_true", help="Update existing endpoint")
    parser.add_argument("--delete", action="store_true", help="Delete endpoint")
    parser.add_argument("--list", action="store_true", help="List endpoints")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = SageMakerDeployer(region_name=args.region)
    
    try:
        if args.list:
            # List endpoints
            endpoints = deployer.list_endpoints()
            print("\nSageMaker Endpoints:")
            print(json.dumps(endpoints, indent=2, default=str))
            return 0
        
        if args.delete:
            # Delete endpoint and related resources
            logger.info(f"Deleting endpoint: {args.endpoint_name}")
            deployer.delete_endpoint(args.endpoint_name)
            
            config_name = f"{args.endpoint_name}-config"
            deployer.delete_endpoint_config(config_name)
            deployer.delete_model(args.model_name)
            
            logger.info("Deletion completed")
            return 0
        
        # Create or update endpoint
        config_name = f"{args.endpoint_name}-config"
        
        if not args.update:
            # Create new model
            logger.info("Creating model...")
            deployer.create_model(
                model_name=args.model_name,
                role_arn=args.role_arn,
                model_artifacts_s3_uri=args.model_artifacts_uri,
                inference_image_uri=args.inference_image
            )
            
            # Create endpoint configuration
            logger.info("Creating endpoint configuration...")
            deployer.create_endpoint_config(
                config_name=config_name,
                model_name=args.model_name,
                instance_type=args.instance_type,
                initial_instance_count=args.instance_count
            )
            
            # Create endpoint
            logger.info("Creating endpoint...")
            deployer.create_endpoint(
                endpoint_name=args.endpoint_name,
                config_name=config_name,
                tags=[
                    {"Key": "Project", "Value": "ArceeAgent"},
                    {"Key": "Environment", "Value": "Production"},
                    {"Key": "CreatedBy", "Value": "SageMakerDeployment"}
                ]
            )
        else:
            # Update existing endpoint
            logger.info("Updating endpoint...")
            deployer.update_endpoint(
                endpoint_name=args.endpoint_name,
                new_config_name=config_name
            )
        
        # Wait for endpoint to be in service
        endpoint_details = deployer.wait_for_endpoint(
            args.endpoint_name,
            args.timeout_minutes
        )
        
        print(f"\nEndpoint deployed successfully!")
        print(f"Endpoint name: {args.endpoint_name}")
        print(f"Status: {endpoint_details['EndpointStatus']}")
        print(f"Creation time: {endpoint_details['CreationTime']}")
        
        # Test endpoint
        print(f"\nTest the endpoint with:")
        print(f"python sagemaker_inference.py --endpoint-name {args.endpoint_name} --test-query 'What is the weather like?'")
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
