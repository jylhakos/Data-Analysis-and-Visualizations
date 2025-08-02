#!/usr/bin/env python3
"""
AWS Resource Allocation Calculator for BERT Fine-tuning
======================================================

This script helps DevOps teams estimate and allocate the optimal AWS resources
for BERT fine-tuning workloads based on dataset size, model complexity,
budget constraints, and performance requirements.

Features:
- Resource estimation based on dataset characteristics
- Cost calculation for different instance types
- Performance optimization recommendations
- Infrastructure as Code generation
- Real-time AWS pricing integration

Usage:
    python aws_resource_calculator.py --dataset-size 5GB --budget 1000 --duration 24h
    python aws_resource_calculator.py --interactive
    python aws_resource_calculator.py --generate-terraform
"""

import argparse
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import boto3
from datetime import datetime, timedelta

@dataclass
class DatasetProfile:
    """Dataset characteristics for resource estimation"""
    size_gb: float
    num_samples: int
    avg_sequence_length: int
    complexity: str  # 'simple', 'medium', 'complex'
    data_type: str   # 'text', 'multimodal'

@dataclass
class TrainingRequirements:
    """Training requirements and constraints"""
    max_budget: float
    max_duration_hours: int
    priority: str  # 'cost', 'performance', 'balanced'
    fault_tolerance: str  # 'low', 'medium', 'high'
    scalability: str  # 'single', 'distributed'

@dataclass
class InstanceOption:
    """AWS Instance option with characteristics"""
    name: str
    instance_type: str
    vcpus: int
    memory_gb: int
    gpus: int
    gpu_memory_gb: int
    hourly_cost: float
    performance_score: float
    use_case: str

class AWSResourceCalculator:
    """Calculate optimal AWS resources for BERT fine-tuning"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.region = region
        self.pricing_cache = {}
        
        # Pre-defined instance configurations (approximate pricing)
        self.instance_options = [
            # GPU Instances - P5 Series (Latest Generation)
            InstanceOption("P5 XLarge", "p5.xlarge", 12, 128, 1, 80, 6.13, 95, "Development, small models"),
            InstanceOption("P5 2XLarge", "p5.2xlarge", 12, 128, 1, 80, 8.19, 95, "Small datasets, POC"),
            InstanceOption("P5 12XLarge", "p5.12xlarge", 48, 512, 2, 160, 24.58, 100, "Medium datasets, production"),
            InstanceOption("P5 24XLarge", "p5.24xlarge", 96, 1024, 4, 320, 49.16, 100, "Large datasets, multi-GPU"),
            InstanceOption("P5 48XLarge", "p5.48xlarge", 192, 2048, 8, 640, 98.32, 100, "Ultra-large scale, research"),
            
            # GPU Instances - P4d Series (Cost-Effective)
            InstanceOption("P4d 24XLarge", "p4d.24xlarge", 96, 1152, 8, 320, 32.77, 85, "Cost-effective multi-GPU"),
            InstanceOption("P4de 24XLarge", "p4de.24xlarge", 96, 1152, 8, 640, 40.96, 85, "Large memory requirements"),
            
            # GPU Instances - G5 Series (Budget-Friendly)
            InstanceOption("G5 XLarge", "g5.xlarge", 4, 16, 1, 24, 1.19, 60, "Budget development"),
            InstanceOption("G5 2XLarge", "g5.2xlarge", 8, 32, 1, 24, 1.38, 60, "Small training jobs"),
            InstanceOption("G5 12XLarge", "g5.12xlarge", 48, 192, 4, 96, 5.52, 70, "Medium budget training"),
            
            # Trainium Instances (Cost-Optimized)
            InstanceOption("Trn1 2XLarge", "trn1.2xlarge", 8, 32, 1, 32, 1.34, 75, "Cost-optimized development"),
            InstanceOption("Trn1 32XLarge", "trn1.32xlarge", 128, 512, 16, 512, 21.50, 80, "Large-scale cost optimization"),
            InstanceOption("Trn1n 32XLarge", "trn1n.32xlarge", 128, 512, 16, 512, 24.78, 85, "Network-intensive training"),
        ]
    
    def estimate_resource_requirements(self, dataset: DatasetProfile) -> Dict:
        """Estimate resource requirements based on dataset characteristics"""
        
        # Base memory requirements (GB)
        base_memory = max(8, dataset.size_gb * 0.5)
        
        # GPU memory requirements based on model size and sequence length
        model_params = self._estimate_model_parameters(dataset)
        gpu_memory_needed = self._calculate_gpu_memory(model_params, dataset.avg_sequence_length)
        
        # Estimated training time (hours)
        training_time = self._estimate_training_time(dataset, model_params)
        
        # Storage requirements
        storage_gb = max(100, dataset.size_gb * 3)  # 3x for processed data, checkpoints
        
        return {
            "memory_gb": base_memory,
            "gpu_memory_gb": gpu_memory_needed,
            "estimated_training_hours": training_time,
            "storage_gb": storage_gb,
            "model_parameters": model_params,
            "recommended_batch_size": self._calculate_batch_size(gpu_memory_needed)
        }
    
    def _estimate_model_parameters(self, dataset: DatasetProfile) -> int:
        """Estimate BERT model parameters based on complexity"""
        base_params = {
            'simple': 110_000_000,    # BERT-Base
            'medium': 340_000_000,    # BERT-Large
            'complex': 1_000_000_000  # Custom large model
        }
        return base_params.get(dataset.complexity, 110_000_000)
    
    def _calculate_gpu_memory(self, model_params: int, seq_length: int) -> int:
        """Calculate required GPU memory in GB"""
        # Rough estimation: 4 bytes per parameter + activation memory
        model_memory = (model_params * 4) / (1024**3)  # Convert to GB
        activation_memory = (seq_length * seq_length * 4) / (1024**3)
        optimizer_memory = model_memory * 2  # Adam optimizer overhead
        
        total_memory = model_memory + activation_memory + optimizer_memory
        return math.ceil(total_memory * 1.5)  # 50% buffer
    
    def _estimate_training_time(self, dataset: DatasetProfile, model_params: int) -> float:
        """Estimate training time in hours"""
        # Base calculation: samples per hour based on model size
        if model_params < 200_000_000:
            samples_per_hour = 50_000
        elif model_params < 500_000_000:
            samples_per_hour = 25_000
        else:
            samples_per_hour = 10_000
        
        # Adjust for sequence length
        if dataset.avg_sequence_length > 256:
            samples_per_hour *= 0.7
        elif dataset.avg_sequence_length > 512:
            samples_per_hour *= 0.5
        
        # Calculate epochs (typically 3-5 for fine-tuning)
        epochs = 3
        total_samples = dataset.num_samples * epochs
        
        return max(1, total_samples / samples_per_hour)
    
    def _calculate_batch_size(self, gpu_memory_gb: int) -> int:
        """Calculate optimal batch size based on GPU memory"""
        if gpu_memory_gb < 16:
            return 4
        elif gpu_memory_gb < 32:
            return 8
        elif gpu_memory_gb < 80:
            return 16
        else:
            return 32
    
    def find_optimal_instances(self, 
                             requirements: Dict, 
                             training_reqs: TrainingRequirements) -> List[Dict]:
        """Find optimal instance configurations"""
        
        suitable_instances = []
        
        for instance in self.instance_options:
            # Check if instance meets requirements
            if (instance.gpu_memory_gb >= requirements["gpu_memory_gb"] and
                instance.memory_gb >= requirements["memory_gb"]):
                
                # Calculate total cost
                total_cost = instance.hourly_cost * requirements["estimated_training_hours"]
                
                if total_cost <= training_reqs.max_budget:
                    efficiency_score = self._calculate_efficiency_score(
                        instance, requirements, training_reqs, total_cost
                    )
                    
                    suitable_instances.append({
                        "instance": instance,
                        "total_cost": total_cost,
                        "efficiency_score": efficiency_score,
                        "cost_per_hour": instance.hourly_cost,
                        "training_hours": requirements["estimated_training_hours"]
                    })
        
        # Sort by efficiency score (descending)
        suitable_instances.sort(key=lambda x: x["efficiency_score"], reverse=True)
        return suitable_instances[:5]  # Return top 5
    
    def _calculate_efficiency_score(self, 
                                  instance: InstanceOption, 
                                  requirements: Dict,
                                  training_reqs: TrainingRequirements,
                                  total_cost: float) -> float:
        """Calculate efficiency score for instance selection"""
        
        # Base performance score
        score = instance.performance_score
        
        # Adjust based on priority
        if training_reqs.priority == 'cost':
            # Favor lower cost
            cost_factor = max(0.1, 1 - (total_cost / training_reqs.max_budget))
            score *= (0.3 + 0.7 * cost_factor)
        elif training_reqs.priority == 'performance':
            # Favor higher performance
            score *= 1.2
        else:  # balanced
            cost_factor = max(0.1, 1 - (total_cost / training_reqs.max_budget))
            score *= (0.6 + 0.4 * cost_factor)
        
        # Adjust for overprovisioning penalty
        gpu_utilization = requirements["gpu_memory_gb"] / instance.gpu_memory_gb
        if gpu_utilization < 0.5:
            score *= 0.8  # Penalty for underutilization
        
        return score
    
    def generate_storage_recommendations(self, requirements: Dict, dataset: DatasetProfile) -> Dict:
        """Generate storage configuration recommendations"""
        
        storage_config = {
            "primary_storage": {},
            "backup_storage": {},
            "high_performance_storage": {}
        }
        
        # Primary storage (EBS)
        if requirements["storage_gb"] < 1000:
            storage_config["primary_storage"] = {
                "type": "EBS gp3",
                "size_gb": requirements["storage_gb"],
                "iops": 3000,
                "throughput_mbps": 250,
                "monthly_cost": requirements["storage_gb"] * 0.08
            }
        else:
            storage_config["primary_storage"] = {
                "type": "EBS io2",
                "size_gb": requirements["storage_gb"],
                "iops": 10000,
                "throughput_mbps": 1000,
                "monthly_cost": requirements["storage_gb"] * 0.125
            }
        
        # High-performance storage for large datasets
        if dataset.size_gb > 100:
            storage_config["high_performance_storage"] = {
                "type": "FSx Lustre",
                "size_tb": math.ceil(dataset.size_gb / 1000),
                "throughput_per_tb": 250,
                "monthly_cost": math.ceil(dataset.size_gb / 1000) * 240
            }
        
        # S3 for training data
        storage_config["backup_storage"] = {
            "type": "S3 Express One Zone",
            "size_gb": dataset.size_gb,
            "monthly_cost": dataset.size_gb * 0.16,
            "access_pattern": "High-frequency training data access"
        }
        
        return storage_config
    
    def generate_terraform_config(self, 
                                instance_config: Dict, 
                                storage_config: Dict,
                                requirements: Dict) -> str:
        """Generate Terraform configuration for the recommended setup"""
        
        instance = instance_config["instance"]
        
        terraform_config = f"""
# Terraform configuration for BERT fine-tuning infrastructure
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = "{self.region}"
}}

# Variables
variable "project_name" {{
  description = "Project name for resource naming"
  type        = string
  default     = "bert-fine-tuning"
}}

variable "environment" {{
  description = "Environment name"
  type        = string
  default     = "dev"
}}

# Data sources
data "aws_ami" "deep_learning" {{
  most_recent = true
  owners      = ["amazon"]
  
  filter {{
    name   = "name"
    values = ["Deep Learning AMI (Ubuntu 18.04) Version *"]
  }}
}}

# Security group for ML training
resource "aws_security_group" "ml_training" {{
  name_prefix = "${{var.project_name}}-ml-training"
  
  ingress {{
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }}
  
  ingress {{
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }}
  
  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  tags = {{
    Name = "${{var.project_name}}-ml-training-sg"
  }}
}}

# EC2 instance for training
resource "aws_instance" "training_instance" {{
  ami           = data.aws_ami.deep_learning.id
  instance_type = "{instance.instance_type}"
  
  vpc_security_group_ids = [aws_security_group.ml_training.id]
  
  # EBS root volume
  root_block_device {{
    volume_type = "{storage_config['primary_storage']['type'].split()[1]}"
    volume_size = {storage_config['primary_storage']['size_gb']}
    iops        = {storage_config['primary_storage'].get('iops', 3000)}
    throughput  = {storage_config['primary_storage'].get('throughput_mbps', 250)}
    encrypted   = true
    
    tags = {{
      Name = "${{var.project_name}}-training-storage"
    }}
  }}
  
  # User data for setup
  user_data = base64encode(<<-EOF
    #!/bin/bash
    
    # Update system
    apt-get update -y
    
    # Install required packages
    pip3 install --upgrade pip
    pip3 install torch transformers datasets boto3 accelerate
    
    # Create directories
    mkdir -p /opt/bert-training/data
    mkdir -p /opt/bert-training/models
    mkdir -p /opt/bert-training/logs
    
    # Set permissions
    chown -R ubuntu:ubuntu /opt/bert-training
    
    # Install NVIDIA drivers and CUDA (if not already installed)
    nvidia-smi || {{
      wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
      mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
      apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
      add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
      apt-get update
      apt-get -y install cuda
    }}
    
    # Configure environment
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /home/ubuntu/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /home/ubuntu/.bashrc
  EOF
  )
  
  tags = {{
    Name        = "${{var.project_name}}-training-instance"
    Environment = var.environment
    Purpose     = "BERT Fine-tuning"
    InstanceType = "{instance.instance_type}"
    EstimatedCost = "${instance_config['total_cost']:.2f}"
  }}
}}

# S3 bucket for training data
resource "aws_s3_bucket" "training_data" {{
  bucket = "${{var.project_name}}-training-data-${{random_id.bucket_suffix.hex}}"
  
  tags = {{
    Name        = "${{var.project_name}}-training-data"
    Environment = var.environment
  }}
}}

resource "random_id" "bucket_suffix" {{
  byte_length = 4
}}

# S3 bucket encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "training_data_encryption" {{
  bucket = aws_s3_bucket.training_data.id
  
  rule {{
    apply_server_side_encryption_by_default {{
      sse_algorithm = "AES256"
    }}
  }}
}}

# CloudWatch log group
resource "aws_cloudwatch_log_group" "training_logs" {{
  name              = "/aws/ec2/${{var.project_name}}-training"
  retention_in_days = 14
  
  tags = {{
    Name        = "${{var.project_name}}-training-logs"
    Environment = var.environment
  }}
}}

# Outputs
output "instance_id" {{
  description = "ID of the training instance"
  value       = aws_instance.training_instance.id
}}

output "instance_public_ip" {{
  description = "Public IP of the training instance"
  value       = aws_instance.training_instance.public_ip
}}

output "s3_bucket_name" {{
  description = "Name of the S3 bucket for training data"
  value       = aws_s3_bucket.training_data.bucket
}}

output "estimated_monthly_cost" {{
  description = "Estimated monthly cost for this configuration"
  value       = "${{{
    ({instance.hourly_cost} * 24 * 30) + 
    {storage_config['primary_storage']['monthly_cost']} + 
    {storage_config['backup_storage']['monthly_cost']}
  }:.2f}} USD"
}}

output "training_configuration" {{
  description = "Recommended training configuration"
  value = {{
    instance_type = "{instance.instance_type}"
    gpu_memory_gb = {instance.gpu_memory_gb}
    recommended_batch_size = {requirements['recommended_batch_size']}
    estimated_training_hours = {requirements['estimated_training_hours']:.1f}
  }}
}}
"""
        
        return terraform_config
    
    def generate_cost_report(self, 
                           instance_configs: List[Dict], 
                           storage_config: Dict,
                           requirements: Dict) -> str:
        """Generate detailed cost analysis report"""
        
        report = f"""
# AWS Resource Cost Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Analysis
- Size: {requirements.get('dataset_size_gb', 'N/A')} GB
- Estimated Samples: {requirements.get('num_samples', 'N/A')}
- Training Hours: {requirements['estimated_training_hours']:.1f}
- GPU Memory Required: {requirements['gpu_memory_gb']} GB

## Instance Recommendations (Top 5)
"""
        
        for i, config in enumerate(instance_configs[:5], 1):
            instance = config["instance"]
            report += f"""
### Option {i}: {instance.name}
- **Instance Type**: {instance.instance_type}
- **Specifications**: {instance.gpus} GPU(s), {instance.gpu_memory_gb}GB GPU RAM, {instance.vcpus} vCPUs
- **Hourly Cost**: ${instance.hourly_cost:.2f}
- **Total Training Cost**: ${config['total_cost']:.2f}
- **Efficiency Score**: {config['efficiency_score']:.1f}/100
- **Use Case**: {instance.use_case}
"""
        
        # Storage costs
        primary_cost = storage_config['primary_storage']['monthly_cost']
        backup_cost = storage_config['backup_storage']['monthly_cost']
        
        report += f"""
## Storage Costs (Monthly)
- **Primary Storage**: {storage_config['primary_storage']['type']} - ${primary_cost:.2f}
- **Backup Storage**: {storage_config['backup_storage']['type']} - ${backup_cost:.2f}
- **Total Storage**: ${primary_cost + backup_cost:.2f}

## Total Cost Breakdown (Best Option)
- **Compute**: ${instance_configs[0]['total_cost']:.2f} (one-time training)
- **Storage**: ${primary_cost + backup_cost:.2f} (monthly)
- **Data Transfer**: ~$20-50 (estimated)
- **Total Project Cost**: ${instance_configs[0]['total_cost'] + primary_cost + backup_cost + 35:.2f}

## Cost Optimization Tips
1. Use Spot Instances for development (up to 90% savings)
2. Consider Reserved Instances for long-term projects (40-60% savings)
3. Use S3 Intelligent Tiering for variable access patterns
4. Set up billing alerts and auto-shutdown policies
5. Monitor GPU utilization and right-size instances

## Performance Optimization
1. Use placement groups for multi-instance training
2. Enable Enhanced Networking (ENA/SR-IOV)
3. Use FSx Lustre for large datasets (>100GB)
4. Implement gradient accumulation for memory efficiency
5. Use mixed precision training to reduce memory usage
"""
        
        return report

def parse_size(size_str: str) -> float:
    """Parse size string (e.g., '5GB', '1.5TB') to GB"""
    size_str = size_str.upper().replace(' ', '')
    
    if size_str.endswith('TB'):
        return float(size_str[:-2]) * 1024
    elif size_str.endswith('GB'):
        return float(size_str[:-2])
    elif size_str.endswith('MB'):
        return float(size_str[:-2]) / 1024
    else:
        return float(size_str)

def parse_duration(duration_str: str) -> int:
    """Parse duration string (e.g., '24h', '3d') to hours"""
    duration_str = duration_str.lower().replace(' ', '')
    
    if duration_str.endswith('d'):
        return int(duration_str[:-1]) * 24
    elif duration_str.endswith('h'):
        return int(duration_str[:-1])
    else:
        return int(duration_str)

def interactive_mode():
    """Interactive mode for resource calculation"""
    print("🤖 AWS Resource Calculator for BERT Fine-tuning")
    print("=" * 50)
    
    # Dataset information
    print("\n📊 Dataset Information:")
    size_str = input("Dataset size (e.g., 5GB, 1.5TB): ")
    dataset_size = parse_size(size_str)
    
    num_samples = int(input("Number of training samples: "))
    seq_length = int(input("Average sequence length (default 256): ") or "256")
    
    complexity = input("Model complexity (simple/medium/complex) [medium]: ") or "medium"
    
    # Training requirements
    print("\n⚙️ Training Requirements:")
    budget = float(input("Maximum budget ($): "))
    duration_str = input("Maximum training duration (e.g., 24h, 3d): ")
    max_duration = parse_duration(duration_str)
    
    priority = input("Priority (cost/performance/balanced) [balanced]: ") or "balanced"
    
    # Create objects
    dataset = DatasetProfile(
        size_gb=dataset_size,
        num_samples=num_samples,
        avg_sequence_length=seq_length,
        complexity=complexity,
        data_type='text'
    )
    
    training_reqs = TrainingRequirements(
        max_budget=budget,
        max_duration_hours=max_duration,
        priority=priority,
        fault_tolerance='medium',
        scalability='single'
    )
    
    # Calculate recommendations
    calculator = AWSResourceCalculator()
    requirements = calculator.estimate_resource_requirements(dataset)
    requirements['dataset_size_gb'] = dataset_size
    requirements['num_samples'] = num_samples
    
    instances = calculator.find_optimal_instances(requirements, training_reqs)
    storage = calculator.generate_storage_recommendations(requirements, dataset)
    
    # Display results
    print("\n" + "=" * 60)
    print("🎯 RECOMMENDATIONS")
    print("=" * 60)
    
    if not instances:
        print("❌ No suitable instances found within budget and requirements.")
        return
    
    best_instance = instances[0]
    print(f"\n🏆 Best Option: {best_instance['instance'].name}")
    print(f"   Instance Type: {best_instance['instance'].instance_type}")
    print(f"   Total Cost: ${best_instance['total_cost']:.2f}")
    print(f"   Training Time: {requirements['estimated_training_hours']:.1f} hours")
    print(f"   Efficiency Score: {best_instance['efficiency_score']:.1f}/100")
    
    print(f"\n📈 Resource Requirements:")
    print(f"   GPU Memory: {requirements['gpu_memory_gb']} GB")
    print(f"   System Memory: {requirements['memory_gb']} GB")
    print(f"   Storage: {requirements['storage_gb']} GB")
    print(f"   Recommended Batch Size: {requirements['recommended_batch_size']}")
    
    # Ask for additional outputs
    print("\n🔧 Additional Options:")
    if input("Generate Terraform configuration? (y/N): ").lower() == 'y':
        terraform_config = calculator.generate_terraform_config(
            best_instance, storage, requirements
        )
        with open('terraform_config.tf', 'w') as f:
            f.write(terraform_config)
        print("✅ Terraform configuration saved to 'terraform_config.tf'")
    
    if input("Generate detailed cost report? (y/N): ").lower() == 'y':
        cost_report = calculator.generate_cost_report(instances, storage, requirements)
        with open('cost_analysis_report.md', 'w') as f:
            f.write(cost_report)
        print("✅ Cost analysis report saved to 'cost_analysis_report.md'")

def main():
    parser = argparse.ArgumentParser(description='AWS Resource Calculator for BERT Fine-tuning')
    parser.add_argument('--dataset-size', help='Dataset size (e.g., 5GB, 1.5TB)')
    parser.add_argument('--num-samples', type=int, help='Number of training samples')
    parser.add_argument('--sequence-length', type=int, default=256, help='Average sequence length')
    parser.add_argument('--complexity', choices=['simple', 'medium', 'complex'], 
                       default='medium', help='Model complexity')
    parser.add_argument('--budget', type=float, help='Maximum budget in USD')
    parser.add_argument('--duration', help='Maximum training duration (e.g., 24h, 3d)')
    parser.add_argument('--priority', choices=['cost', 'performance', 'balanced'], 
                       default='balanced', help='Optimization priority')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--generate-terraform', action='store_true', 
                       help='Generate Terraform configuration')
    parser.add_argument('--output-format', choices=['json', 'yaml', 'text'], 
                       default='text', help='Output format')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return
    
    # Validate required arguments for non-interactive mode
    if not all([args.dataset_size, args.num_samples, args.budget, args.duration]):
        print("Error: --dataset-size, --num-samples, --budget, and --duration are required for non-interactive mode")
        print("Use --interactive for guided setup")
        return
    
    # Parse arguments
    dataset_size = parse_size(args.dataset_size)
    max_duration = parse_duration(args.duration)
    
    # Create objects
    dataset = DatasetProfile(
        size_gb=dataset_size,
        num_samples=args.num_samples,
        avg_sequence_length=args.sequence_length,
        complexity=args.complexity,
        data_type='text'
    )
    
    training_reqs = TrainingRequirements(
        max_budget=args.budget,
        max_duration_hours=max_duration,
        priority=args.priority,
        fault_tolerance='medium',
        scalability='single'
    )
    
    # Calculate recommendations
    calculator = AWSResourceCalculator(region=args.region)
    requirements = calculator.estimate_resource_requirements(dataset)
    requirements['dataset_size_gb'] = dataset_size
    requirements['num_samples'] = args.num_samples
    
    instances = calculator.find_optimal_instances(requirements, training_reqs)
    storage = calculator.generate_storage_recommendations(requirements, dataset)
    
    # Output results
    if args.output_format == 'json':
        result = {
            'requirements': requirements,
            'instances': [
                {
                    'name': inst['instance'].name,
                    'type': inst['instance'].instance_type,
                    'cost': inst['total_cost'],
                    'efficiency': inst['efficiency_score']
                } for inst in instances[:3]
            ],
            'storage': storage
        }
        print(json.dumps(result, indent=2))
    else:
        # Text output
        print("🎯 AWS Resource Recommendations for BERT Fine-tuning")
        print("=" * 60)
        
        if instances:
            best = instances[0]
            print(f"\n🏆 Recommended: {best['instance'].name}")
            print(f"   Cost: ${best['total_cost']:.2f}")
            print(f"   Time: {requirements['estimated_training_hours']:.1f}h")
            print(f"   Efficiency: {best['efficiency_score']:.1f}/100")
        else:
            print("❌ No suitable instances found within constraints")
    
    if args.generate_terraform and instances:
        terraform_config = calculator.generate_terraform_config(
            instances[0], storage, requirements
        )
        with open('generated_terraform.tf', 'w') as f:
            f.write(terraform_config)
        print("\n✅ Terraform configuration saved to 'generated_terraform.tf'")

if __name__ == "__main__":
    main()
