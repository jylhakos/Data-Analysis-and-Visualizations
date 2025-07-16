#!/usr/bin/env python3
"""
SageMaker Inference Client for Arcee Agent

Handles communication with deployed Arcee Agent model on AWS SageMaker endpoints.
Provides both real-time and batch inference capabilities.
"""

import boto3
import json
import logging
import os
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import uuid
import base64
import time

logger = logging.getLogger(__name__)

class SageMakerInference:
    """
    SageMaker inference client for Arcee Agent model.
    """
    
    def __init__(
        self,
        endpoint_name: Optional[str] = None,
        region_name: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None
    ):
        """
        Initialize SageMaker inference client.
        
        Args:
            endpoint_name: SageMaker endpoint name (can be set via environment)
            region_name: AWS region
            aws_access_key_id: AWS access key (optional, uses default credentials)
            aws_secret_access_key: AWS secret key (optional)
            aws_session_token: AWS session token (optional)
        """
        self.endpoint_name = endpoint_name or os.getenv("SAGEMAKER_ENDPOINT_NAME")
        self.region_name = region_name
        
        # Initialize boto3 client
        session_kwargs = {"region_name": region_name}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                "aws_access_key_id": aws_access_key_id,
                "aws_secret_access_key": aws_secret_access_key
            })
            if aws_session_token:
                session_kwargs["aws_session_token"] = aws_session_token
        
        try:
            self.session = boto3.Session(**session_kwargs)
            self.sagemaker_runtime = self.session.client("sagemaker-runtime")
            self.sagemaker = self.session.client("sagemaker")
            logger.info(f"Initialized SageMaker client for region {region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SageMaker client: {e}")
            raise
        
        # Validate endpoint if provided
        if self.endpoint_name:
            self.validate_endpoint()
    
    def validate_endpoint(self) -> bool:
        """
        Validate that the SageMaker endpoint exists and is in service.
        
        Returns:
            True if endpoint is valid and in service
        """
        try:
            response = self.sagemaker.describe_endpoint(EndpointName=self.endpoint_name)
            status = response["EndpointStatus"]
            
            if status == "InService":
                logger.info(f"Endpoint {self.endpoint_name} is in service")
                return True
            else:
                logger.warning(f"Endpoint {self.endpoint_name} status: {status}")
                return False
        except Exception as e:
            logger.error(f"Failed to validate endpoint {self.endpoint_name}: {e}")
            return False
    
    def invoke_endpoint(
        self,
        payload: Dict[str, Any],
        content_type: str = "application/json",
        accept: str = "application/json",
        custom_attributes: Optional[str] = None,
        target_model: Optional[str] = None,
        target_variant: Optional[str] = None,
        inference_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Invoke SageMaker endpoint for real-time inference.
        
        Args:
            payload: Input data for the model
            content_type: Content type of the payload
            accept: Accept header for response
            custom_attributes: Custom attributes for the request
            target_model: Target model name (for multi-model endpoints)
            target_variant: Target variant name (for A/B testing)
            inference_id: Unique inference identifier
            
        Returns:
            Model response as dictionary
        """
        if not self.endpoint_name:
            raise ValueError("Endpoint name not provided")
        
        try:
            # Prepare request parameters
            invoke_args = {
                "EndpointName": self.endpoint_name,
                "ContentType": content_type,
                "Accept": accept,
                "Body": json.dumps(payload)
            }
            
            if custom_attributes:
                invoke_args["CustomAttributes"] = custom_attributes
            if target_model:
                invoke_args["TargetModel"] = target_model
            if target_variant:
                invoke_args["TargetVariant"] = target_variant
            if inference_id:
                invoke_args["InferenceId"] = inference_id
            
            # Make the request
            start_time = time.time()
            response = self.sagemaker_runtime.invoke_endpoint(**invoke_args)
            end_time = time.time()
            
            # Parse response
            result = json.loads(response["Body"].read().decode())
            
            # Add metadata
            result["_metadata"] = {
                "inference_time": end_time - start_time,
                "endpoint_name": self.endpoint_name,
                "timestamp": datetime.utcnow().isoformat(),
                "inference_id": inference_id or str(uuid.uuid4())
            }
            
            logger.info(f"Inference completed in {end_time - start_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Failed to invoke endpoint: {e}")
            raise
    
    def function_call(
        self,
        query: str,
        tools: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_p: float = 0.9,
        stop_sequences: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform function calling inference with the Arcee Agent model.
        
        Args:
            query: User query
            tools: Available tools/functions
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop_sequences: Stop sequences for generation
            
        Returns:
            Function calling response
        """
        # Import here to avoid circular imports
        from main import create_function_calling_prompt
        
        # Create the prompt
        prompt = create_function_calling_prompt(query, tools)
        
        # Prepare payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                "return_full_text": False
            }
        }
        
        if stop_sequences:
            payload["parameters"]["stop"] = stop_sequences
        
        # Invoke endpoint
        response = self.invoke_endpoint(payload)
        
        # Process response for function calling
        if "generated_text" in response:
            response["function_calls"] = self._parse_function_calls(response["generated_text"])
        
        return response
    
    def _parse_function_calls(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse function calls from model output.
        
        Args:
            text: Generated text from model
            
        Returns:
            List of parsed function calls
        """
        try:
            # Import here to avoid circular imports
            from main import parse_tool_calls
            return parse_tool_calls(text)
        except Exception as e:
            logger.warning(f"Failed to parse function calls: {e}")
            return []
    
    def batch_inference(
        self,
        batch_input_s3_uri: str,
        batch_output_s3_uri: str,
        transform_job_name: Optional[str] = None,
        model_name: Optional[str] = None,
        instance_type: str = "ml.m5.xlarge",
        instance_count: int = 1,
        max_concurrent_transforms: Optional[int] = None,
        max_payload_in_mb: int = 6,
        batch_strategy: str = "MultiRecord"
    ) -> str:
        """
        Create a batch transform job for batch inference.
        
        Args:
            batch_input_s3_uri: S3 URI for input data
            batch_output_s3_uri: S3 URI for output data
            transform_job_name: Name for the transform job
            model_name: Name of the model to use
            instance_type: EC2 instance type
            instance_count: Number of instances
            max_concurrent_transforms: Max concurrent transforms
            max_payload_in_mb: Max payload size in MB
            batch_strategy: Batch strategy
            
        Returns:
            Transform job name
        """
        if not transform_job_name:
            transform_job_name = f"arcee-agent-batch-{int(time.time())}"
        
        if not model_name:
            model_name = f"{self.endpoint_name}-model" if self.endpoint_name else "arcee-agent-model"
        
        try:
            transform_args = {
                "TransformJobName": transform_job_name,
                "ModelName": model_name,
                "TransformInput": {
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": batch_input_s3_uri
                        }
                    },
                    "ContentType": "application/json",
                    "SplitType": "Line"
                },
                "TransformOutput": {
                    "S3OutputPath": batch_output_s3_uri,
                    "Accept": "application/json"
                },
                "TransformResources": {
                    "InstanceType": instance_type,
                    "InstanceCount": instance_count
                },
                "MaxPayloadInMB": max_payload_in_mb,
                "BatchStrategy": batch_strategy
            }
            
            if max_concurrent_transforms:
                transform_args["MaxConcurrentTransforms"] = max_concurrent_transforms
            
            response = self.sagemaker.create_transform_job(**transform_args)
            
            logger.info(f"Created batch transform job: {transform_job_name}")
            return transform_job_name
            
        except Exception as e:
            logger.error(f"Failed to create batch transform job: {e}")
            raise
    
    def get_transform_job_status(self, transform_job_name: str) -> Dict[str, Any]:
        """
        Get the status of a batch transform job.
        
        Args:
            transform_job_name: Name of the transform job
            
        Returns:
            Transform job details
        """
        try:
            response = self.sagemaker.describe_transform_job(
                TransformJobName=transform_job_name
            )
            return {
                "job_name": transform_job_name,
                "status": response["TransformJobStatus"],
                "creation_time": response["CreationTime"],
                "start_time": response.get("TransformStartTime"),
                "end_time": response.get("TransformEndTime"),
                "failure_reason": response.get("FailureReason"),
                "model_name": response["ModelName"],
                "input_s3_uri": response["TransformInput"]["DataSource"]["S3DataSource"]["S3Uri"],
                "output_s3_uri": response["TransformOutput"]["S3OutputPath"]
            }
        except Exception as e:
            logger.error(f"Failed to get transform job status: {e}")
            raise
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """
        List available SageMaker endpoints.
        
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
    
    def get_endpoint_metrics(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Get CloudWatch metrics for the endpoint.
        
        Args:
            start_time: Start time for metrics
            end_time: End time for metrics
            
        Returns:
            Endpoint metrics
        """
        if not self.endpoint_name:
            raise ValueError("Endpoint name not provided")
        
        try:
            cloudwatch = self.session.client("cloudwatch")
            
            metrics = {}
            metric_names = [
                "Invocations",
                "InvocationsPerInstance", 
                "ModelLatency",
                "OverheadLatency",
                "Invocation4XXErrors",
                "Invocation5XXErrors"
            ]
            
            for metric_name in metric_names:
                response = cloudwatch.get_metric_statistics(
                    Namespace="AWS/SageMaker",
                    MetricName=metric_name,
                    Dimensions=[
                        {
                            "Name": "EndpointName",
                            "Value": self.endpoint_name
                        }
                    ],
                    StartTime=start_time,
                    EndTime=end_time,
                    Period=300,  # 5 minutes
                    Statistics=["Average", "Sum", "Maximum"]
                )
                metrics[metric_name] = response["Datapoints"]
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to get endpoint metrics: {e}")
            raise


class SageMakerTrainingJob:
    """
    Helper class for managing SageMaker training jobs.
    """
    
    def __init__(self, session: Optional[boto3.Session] = None, region_name: str = "us-east-1"):
        """
        Initialize SageMaker training job manager.
        
        Args:
            session: Boto3 session (optional)
            region_name: AWS region
        """
        self.session = session or boto3.Session(region_name=region_name)
        self.sagemaker = self.session.client("sagemaker")
        self.region_name = region_name
    
    def create_training_job(
        self,
        job_name: str,
        role_arn: str,
        training_image: str,
        input_data_s3_uri: str,
        output_s3_uri: str,
        instance_type: str = "ml.g4dn.xlarge",
        instance_count: int = 1,
        volume_size_gb: int = 30,
        max_runtime_seconds: int = 86400,  # 24 hours
        hyperparameters: Optional[Dict[str, str]] = None,
        environment: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Create a SageMaker training job for fine-tuning.
        
        Args:
            job_name: Training job name
            role_arn: IAM role ARN for SageMaker
            training_image: Docker image URI for training
            input_data_s3_uri: S3 URI for training data
            output_s3_uri: S3 URI for model artifacts
            instance_type: EC2 instance type
            instance_count: Number of training instances
            volume_size_gb: EBS volume size
            max_runtime_seconds: Maximum training time
            hyperparameters: Training hyperparameters
            environment: Environment variables
            
        Returns:
            Training job name
        """
        try:
            training_args = {
                "TrainingJobName": job_name,
                "RoleArn": role_arn,
                "AlgorithmSpecification": {
                    "TrainingImage": training_image,
                    "TrainingInputMode": "File"
                },
                "InputDataConfig": [
                    {
                        "ChannelName": "training",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": input_data_s3_uri,
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
                    "MaxRuntimeInSeconds": max_runtime_seconds
                }
            }
            
            if hyperparameters:
                training_args["HyperParameters"] = hyperparameters
            
            if environment:
                training_args["AlgorithmSpecification"]["Environment"] = environment
            
            response = self.sagemaker.create_training_job(**training_args)
            logger.info(f"Created training job: {job_name}")
            return job_name
            
        except Exception as e:
            logger.error(f"Failed to create training job: {e}")
            raise
    
    def get_training_job_status(self, job_name: str) -> Dict[str, Any]:
        """
        Get training job status and details.
        
        Args:
            job_name: Training job name
            
        Returns:
            Training job details
        """
        try:
            response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
            return {
                "job_name": job_name,
                "status": response["TrainingJobStatus"],
                "creation_time": response["CreationTime"],
                "start_time": response.get("TrainingStartTime"),
                "end_time": response.get("TrainingEndTime"),
                "failure_reason": response.get("FailureReason"),
                "model_artifacts": response.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
                "training_time_seconds": response.get("TrainingTimeInSeconds"),
                "billable_time_seconds": response.get("BillableTimeInSeconds")
            }
        except Exception as e:
            logger.error(f"Failed to get training job status: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SageMaker inference")
    parser.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--test-query", default="What is the weather like?", help="Test query")
    
    args = parser.parse_args()
    
    # Initialize client
    client = SageMakerInference(
        endpoint_name=args.endpoint_name,
        region_name=args.region
    )
    
    # Test function calling
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get weather for"
                    }
                },
                "required": ["location"]
            }
        }
    ]
    
    try:
        response = client.function_call(args.test_query, tools)
        print(f"Response: {json.dumps(response, indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
