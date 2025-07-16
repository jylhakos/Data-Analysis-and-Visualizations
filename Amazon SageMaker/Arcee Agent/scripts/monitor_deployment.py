#!/usr/bin/env python3
"""
AWS Cost and Performance Monitoring for Arcee Agent SageMaker Deployment

This script provides comprehensive monitoring, cost tracking, and alerting
for the Arcee Agent deployment on AWS SageMaker.
"""

import argparse
import boto3
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArceeAgentMonitor:
    """
    Monitoring and cost management for Arcee Agent SageMaker deployment.
    """
    
    def __init__(self, region_name: str = "us-east-1"):
        """
        Initialize monitoring client.
        
        Args:
            region_name: AWS region
        """
        self.region_name = region_name
        self.session = boto3.Session(region_name=region_name)
        
        # Initialize AWS clients
        self.sagemaker = self.session.client("sagemaker")
        self.cloudwatch = self.session.client("cloudwatch")
        self.cost_explorer = self.session.client("ce", region_name="us-east-1")  # Cost Explorer is global
        self.sns = self.session.client("sns")
    
    def get_endpoint_metrics(
        self,
        endpoint_name: str,
        start_time: datetime,
        end_time: datetime,
        period: int = 300
    ) -> Dict[str, Any]:
        """
        Get CloudWatch metrics for SageMaker endpoint.
        
        Args:
            endpoint_name: SageMaker endpoint name
            start_time: Start time for metrics
            end_time: End time for metrics
            period: Metric period in seconds
            
        Returns:
            Endpoint metrics
        """
        try:
            metrics = {}
            
            # Define metrics to collect
            metric_definitions = [
                ("Invocations", ["Sum", "Average"]),
                ("InvocationsPerInstance", ["Sum", "Average"]),
                ("ModelLatency", ["Average", "Maximum"]),
                ("OverheadLatency", ["Average", "Maximum"]),
                ("Invocation4XXErrors", ["Sum"]),
                ("Invocation5XXErrors", ["Sum"]),
                ("CPUUtilization", ["Average", "Maximum"]),
                ("MemoryUtilization", ["Average", "Maximum"]),
                ("DiskUtilization", ["Average", "Maximum"])
            ]
            
            for metric_name, statistics in metric_definitions:
                try:
                    response = self.cloudwatch.get_metric_statistics(
                        Namespace="AWS/SageMaker",
                        MetricName=metric_name,
                        Dimensions=[
                            {
                                "Name": "EndpointName",
                                "Value": endpoint_name
                            }
                        ],
                        StartTime=start_time,
                        EndTime=end_time,
                        Period=period,
                        Statistics=statistics
                    )
                    
                    metrics[metric_name] = {
                        "datapoints": response["Datapoints"],
                        "label": response["Label"]
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to get metric {metric_name}: {e}")
                    metrics[metric_name] = {"datapoints": [], "error": str(e)}
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get endpoint metrics: {e}")
            raise
    
    def get_training_job_metrics(
        self,
        training_job_name: str
    ) -> Dict[str, Any]:
        """
        Get metrics for a training job.
        
        Args:
            training_job_name: SageMaker training job name
            
        Returns:
            Training job metrics
        """
        try:
            # Get training job details
            response = self.sagemaker.describe_training_job(
                TrainingJobName=training_job_name
            )
            
            # Calculate costs
            start_time = response.get("TrainingStartTime")
            end_time = response.get("TrainingEndTime")
            billable_seconds = response.get("BillableTimeInSeconds", 0)
            
            # Get instance pricing (approximate)
            instance_type = response["ResourceConfig"]["InstanceType"]
            instance_count = response["ResourceConfig"]["InstanceCount"]
            
            # Approximate hourly costs (these should be updated with current pricing)
            instance_costs = {
                "ml.m5.large": 0.115,
                "ml.m5.xlarge": 0.230,
                "ml.m5.2xlarge": 0.461,
                "ml.m5.4xlarge": 0.922,
                "ml.g4dn.xlarge": 0.526,
                "ml.g4dn.2xlarge": 0.752,
                "ml.g4dn.4xlarge": 1.204,
                "ml.p3.2xlarge": 3.06,
                "ml.p3.8xlarge": 12.24
            }
            
            hourly_cost = instance_costs.get(instance_type, 0.5)  # Default fallback
            total_cost = (billable_seconds / 3600) * hourly_cost * instance_count
            
            return {
                "training_job_name": training_job_name,
                "status": response["TrainingJobStatus"],
                "instance_type": instance_type,
                "instance_count": instance_count,
                "start_time": start_time,
                "end_time": end_time,
                "training_time_seconds": response.get("TrainingTimeInSeconds"),
                "billable_time_seconds": billable_seconds,
                "estimated_cost_usd": round(total_cost, 2),
                "hourly_cost_usd": hourly_cost,
                "model_artifacts": response.get("ModelArtifacts", {}).get("S3ModelArtifacts")
            }
            
        except Exception as e:
            logger.error(f"Failed to get training job metrics: {e}")
            raise
    
    def get_endpoint_costs(
        self,
        endpoint_name: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """
        Get cost breakdown for SageMaker endpoint.
        
        Args:
            endpoint_name: SageMaker endpoint name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Cost breakdown
        """
        try:
            # Get cost data from Cost Explorer
            response = self.cost_explorer.get_cost_and_usage(
                TimePeriod={
                    "Start": start_date,
                    "End": end_date
                },
                Granularity="DAILY",
                Metrics=["BlendedCost", "UsageQuantity"],
                GroupBy=[
                    {
                        "Type": "DIMENSION",
                        "Key": "SERVICE"
                    }
                ],
                Filter={
                    "Dimensions": {
                        "Key": "SERVICE",
                        "Values": ["Amazon SageMaker"]
                    }
                }
            )
            
            # Calculate total costs
            total_cost = 0
            daily_costs = []
            
            for result in response["ResultsByTime"]:
                date = result["TimePeriod"]["Start"]
                cost = 0
                
                for group in result["Groups"]:
                    if group["Keys"][0] == "Amazon SageMaker":
                        cost = float(group["Metrics"]["BlendedCost"]["Amount"])
                        break
                
                daily_costs.append({
                    "date": date,
                    "cost": cost
                })
                total_cost += cost
            
            return {
                "endpoint_name": endpoint_name,
                "period": f"{start_date} to {end_date}",
                "total_cost_usd": round(total_cost, 2),
                "daily_costs": daily_costs,
                "average_daily_cost": round(total_cost / len(daily_costs), 2) if daily_costs else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get endpoint costs: {e}")
            raise
    
    def create_cost_alert(
        self,
        alert_name: str,
        budget_amount: float,
        email_address: str,
        threshold_percent: int = 80
    ) -> str:
        """
        Create cost alert for SageMaker usage.
        
        Args:
            alert_name: Alert name
            budget_amount: Monthly budget amount in USD
            email_address: Email for notifications
            threshold_percent: Alert threshold percentage
            
        Returns:
            SNS topic ARN
        """
        try:
            # Create SNS topic for alerts
            topic_response = self.sns.create_topic(Name=f"{alert_name}-alerts")
            topic_arn = topic_response["TopicArn"]
            
            # Subscribe email to topic
            self.sns.subscribe(
                TopicArn=topic_arn,
                Protocol="email",
                Endpoint=email_address
            )
            
            # Create CloudWatch alarm for cost monitoring
            alarm_name = f"{alert_name}-cost-alarm"
            
            self.cloudwatch.put_metric_alarm(
                AlarmName=alarm_name,
                ComparisonOperator="GreaterThanThreshold",
                EvaluationPeriods=1,
                MetricName="EstimatedCharges",
                Namespace="AWS/Billing",
                Period=86400,  # 24 hours
                Statistic="Maximum",
                Threshold=budget_amount * (threshold_percent / 100),
                ActionsEnabled=True,
                AlarmActions=[topic_arn],
                AlarmDescription=f"Alert when SageMaker costs exceed {threshold_percent}% of budget",
                Dimensions=[
                    {
                        "Name": "Currency",
                        "Value": "USD"
                    },
                    {
                        "Name": "ServiceName",
                        "Value": "AmazonSageMaker"
                    }
                ]
            )
            
            logger.info(f"Created cost alert: {alarm_name}")
            return topic_arn
            
        except Exception as e:
            logger.error(f"Failed to create cost alert: {e}")
            raise
    
    def generate_daily_report(
        self,
        endpoint_name: Optional[str] = None,
        include_costs: bool = True
    ) -> Dict[str, Any]:
        """
        Generate daily monitoring report.
        
        Args:
            endpoint_name: SageMaker endpoint name (optional)
            include_costs: Whether to include cost information
            
        Returns:
            Daily report
        """
        try:
            report = {
                "report_date": datetime.utcnow().isoformat(),
                "region": self.region_name
            }
            
            # Get all endpoints if none specified
            if endpoint_name:
                endpoints = [endpoint_name]
            else:
                endpoint_list = self.sagemaker.list_endpoints()
                endpoints = [ep["EndpointName"] for ep in endpoint_list["Endpoints"]]
            
            report["endpoints"] = {}
            
            # Get metrics for each endpoint
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)
            
            for ep_name in endpoints:
                try:
                    # Get endpoint details
                    endpoint_details = self.sagemaker.describe_endpoint(EndpointName=ep_name)
                    
                    # Get metrics
                    metrics = self.get_endpoint_metrics(ep_name, start_time, end_time)
                    
                    # Calculate summary statistics
                    invocations = 0
                    avg_latency = 0
                    errors = 0
                    
                    if metrics.get("Invocations", {}).get("datapoints"):
                        invocations = sum(dp["Sum"] for dp in metrics["Invocations"]["datapoints"])
                    
                    if metrics.get("ModelLatency", {}).get("datapoints"):
                        latencies = [dp["Average"] for dp in metrics["ModelLatency"]["datapoints"]]
                        avg_latency = sum(latencies) / len(latencies) if latencies else 0
                    
                    if metrics.get("Invocation4XXErrors", {}).get("datapoints"):
                        errors += sum(dp["Sum"] for dp in metrics["Invocation4XXErrors"]["datapoints"])
                    
                    if metrics.get("Invocation5XXErrors", {}).get("datapoints"):
                        errors += sum(dp["Sum"] for dp in metrics["Invocation5XXErrors"]["datapoints"])
                    
                    endpoint_report = {
                        "status": endpoint_details["EndpointStatus"],
                        "creation_time": endpoint_details["CreationTime"],
                        "last_modified": endpoint_details["LastModifiedTime"],
                        "instance_type": endpoint_details["ProductionVariants"][0]["InstanceType"],
                        "instance_count": endpoint_details["ProductionVariants"][0]["CurrentInstanceCount"],
                        "metrics": {
                            "total_invocations": int(invocations),
                            "average_latency_ms": round(avg_latency, 2),
                            "total_errors": int(errors),
                            "error_rate": round((errors / invocations * 100) if invocations > 0 else 0, 2)
                        }
                    }
                    
                    # Add cost information if requested
                    if include_costs:
                        try:
                            yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
                            today = datetime.utcnow().strftime("%Y-%m-%d")
                            cost_info = self.get_endpoint_costs(ep_name, yesterday, today)
                            endpoint_report["costs"] = cost_info
                        except Exception as e:
                            logger.warning(f"Failed to get cost info for {ep_name}: {e}")
                    
                    report["endpoints"][ep_name] = endpoint_report
                    
                except Exception as e:
                    logger.error(f"Failed to get report for endpoint {ep_name}: {e}")
                    report["endpoints"][ep_name] = {"error": str(e)}
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate daily report: {e}")
            raise
    
    def cleanup_old_resources(
        self,
        days_threshold: int = 7,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Clean up old SageMaker resources to reduce costs.
        
        Args:
            days_threshold: Delete resources older than this many days
            dry_run: If True, only list resources without deleting
            
        Returns:
            Cleanup report
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)
            cleanup_report = {
                "cutoff_date": cutoff_date.isoformat(),
                "dry_run": dry_run,
                "resources_to_delete": [],
                "resources_deleted": []
            }
            
            # Find old training jobs
            training_jobs = self.sagemaker.list_training_jobs(
                StatusEquals="Completed",
                MaxResults=100
            )
            
            for job in training_jobs["TrainingJobSummaries"]:
                if job["CreationTime"].replace(tzinfo=None) < cutoff_date:
                    resource = {
                        "type": "training_job",
                        "name": job["TrainingJobName"],
                        "creation_time": job["CreationTime"],
                        "status": job["TrainingJobStatus"]
                    }
                    
                    if dry_run:
                        cleanup_report["resources_to_delete"].append(resource)
                    else:
                        # Note: Training jobs are automatically cleaned up by AWS
                        # We just track them for cost analysis
                        cleanup_report["resources_deleted"].append(resource)
            
            # Find old models
            models = self.sagemaker.list_models(MaxResults=100)
            
            for model in models["Models"]:
                if model["CreationTime"].replace(tzinfo=None) < cutoff_date:
                    resource = {
                        "type": "model",
                        "name": model["ModelName"],
                        "creation_time": model["CreationTime"]
                    }
                    
                    if dry_run:
                        cleanup_report["resources_to_delete"].append(resource)
                    else:
                        try:
                            self.sagemaker.delete_model(ModelName=model["ModelName"])
                            cleanup_report["resources_deleted"].append(resource)
                            logger.info(f"Deleted model: {model['ModelName']}")
                        except Exception as e:
                            logger.error(f"Failed to delete model {model['ModelName']}: {e}")
            
            # Find old endpoint configurations
            configs = self.sagemaker.list_endpoint_configs(MaxResults=100)
            
            for config in configs["EndpointConfigs"]:
                if config["CreationTime"].replace(tzinfo=None) < cutoff_date:
                    # Check if config is in use by any endpoint
                    in_use = False
                    try:
                        endpoints = self.sagemaker.list_endpoints()
                        for endpoint in endpoints["Endpoints"]:
                            endpoint_details = self.sagemaker.describe_endpoint(
                                EndpointName=endpoint["EndpointName"]
                            )
                            if endpoint_details["EndpointConfigName"] == config["EndpointConfigName"]:
                                in_use = True
                                break
                    except:
                        pass
                    
                    if not in_use:
                        resource = {
                            "type": "endpoint_config",
                            "name": config["EndpointConfigName"],
                            "creation_time": config["CreationTime"]
                        }
                        
                        if dry_run:
                            cleanup_report["resources_to_delete"].append(resource)
                        else:
                            try:
                                self.sagemaker.delete_endpoint_config(
                                    EndpointConfigName=config["EndpointConfigName"]
                                )
                                cleanup_report["resources_deleted"].append(resource)
                                logger.info(f"Deleted endpoint config: {config['EndpointConfigName']}")
                            except Exception as e:
                                logger.error(f"Failed to delete config {config['EndpointConfigName']}: {e}")
            
            return cleanup_report
            
        except Exception as e:
            logger.error(f"Failed to cleanup resources: {e}")
            raise

def main():
    """
    Main function for monitoring and cost management.
    """
    parser = argparse.ArgumentParser(description="Arcee Agent SageMaker monitoring and cost management")
    
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--endpoint-name", help="SageMaker endpoint name")
    parser.add_argument("--training-job-name", help="SageMaker training job name")
    
    # Actions
    parser.add_argument("--daily-report", action="store_true", help="Generate daily report")
    parser.add_argument("--endpoint-metrics", action="store_true", help="Get endpoint metrics")
    parser.add_argument("--training-metrics", action="store_true", help="Get training job metrics")
    parser.add_argument("--cost-analysis", action="store_true", help="Get cost analysis")
    parser.add_argument("--create-alert", action="store_true", help="Create cost alert")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old resources")
    
    # Parameters
    parser.add_argument("--hours", type=int, default=24, help="Hours of metrics to retrieve")
    parser.add_argument("--budget", type=float, default=100.0, help="Monthly budget for alerts")
    parser.add_argument("--email", help="Email address for alerts")
    parser.add_argument("--dry-run", action="store_true", help="Dry run for cleanup")
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = ArceeAgentMonitor(region_name=args.region)
    
    try:
        if args.daily_report:
            # Generate daily report
            report = monitor.generate_daily_report(
                endpoint_name=args.endpoint_name,
                include_costs=True
            )
            print("Daily Report:")
            print(json.dumps(report, indent=2, default=str))
        
        elif args.endpoint_metrics:
            # Get endpoint metrics
            if not args.endpoint_name:
                logger.error("--endpoint-name is required for endpoint metrics")
                return 1
            
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=args.hours)
            
            metrics = monitor.get_endpoint_metrics(
                args.endpoint_name,
                start_time,
                end_time
            )
            print("Endpoint Metrics:")
            print(json.dumps(metrics, indent=2, default=str))
        
        elif args.training_metrics:
            # Get training job metrics
            if not args.training_job_name:
                logger.error("--training-job-name is required for training metrics")
                return 1
            
            metrics = monitor.get_training_job_metrics(args.training_job_name)
            print("Training Job Metrics:")
            print(json.dumps(metrics, indent=2, default=str))
        
        elif args.cost_analysis:
            # Get cost analysis
            if not args.endpoint_name:
                logger.error("--endpoint-name is required for cost analysis")
                return 1
            
            end_date = datetime.utcnow().strftime("%Y-%m-%d")
            start_date = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            costs = monitor.get_endpoint_costs(args.endpoint_name, start_date, end_date)
            print("Cost Analysis:")
            print(json.dumps(costs, indent=2, default=str))
        
        elif args.create_alert:
            # Create cost alert
            if not args.email:
                logger.error("--email is required for creating alerts")
                return 1
            
            topic_arn = monitor.create_cost_alert(
                f"arcee-agent-{args.endpoint_name or 'deployment'}",
                args.budget,
                args.email
            )
            print(f"Cost alert created. SNS Topic ARN: {topic_arn}")
        
        elif args.cleanup:
            # Cleanup old resources
            report = monitor.cleanup_old_resources(dry_run=args.dry_run)
            print("Cleanup Report:")
            print(json.dumps(report, indent=2, default=str))
        
        else:
            logger.error("No action specified. Use --help for options.")
            return 1
        
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
