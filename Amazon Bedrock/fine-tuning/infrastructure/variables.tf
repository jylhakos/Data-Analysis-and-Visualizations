# Terraform variables definition file
# This file defines all configurable parameters for the Bedrock infrastructure

variable "aws_region" {
  description = "AWS region where resources will be created"
  type        = string
  default     = "us-east-1"
  
  validation {
    condition = contains([
      "us-east-1", "us-west-2", "eu-west-1", "eu-central-1", 
      "ap-southeast-1", "ap-northeast-1"
    ], var.aws_region)
    error_message = "AWS region must be one that supports Amazon Bedrock."
  }
}

variable "environment" {
  description = "Environment name for resource tagging and naming"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "project_name" {
  description = "Project name used for resource naming and tagging"
  type        = string
  default     = "bert-fine-tuning"
  
  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "Project name must contain only lowercase letters, numbers, and hyphens."
  }
}

variable "max_budget_amount" {
  description = "Maximum budget amount in USD for cost control"
  type        = number
  default     = 100
  
  validation {
    condition     = var.max_budget_amount > 0 && var.max_budget_amount <= 10000
    error_message = "Budget amount must be between 1 and 10000 USD."
  }
}

variable "notification_email" {
  description = "Email address for budget alerts and notifications"
  type        = string
  default     = "your-email@example.com"
  
  validation {
    condition     = can(regex("^[\\w\\.-]+@[\\w\\.-]+\\.[a-z]{2,}$", var.notification_email))
    error_message = "Must be a valid email address."
  }
}

variable "log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 14
  
  validation {
    condition = contains([
      1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
    ], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch retention period."
  }
}

variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "s3_versioning_enabled" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

variable "training_data_encryption" {
  description = "S3 encryption method for training data"
  type        = string
  default     = "AES256"
  
  validation {
    condition     = contains(["AES256", "aws:kms"], var.training_data_encryption)
    error_message = "Encryption must be AES256 or aws:kms."
  }
}

variable "bedrock_model_access" {
  description = "List of Bedrock models to request access for"
  type        = list(string)
  default = [
    "amazon.titan-text-express-v1",
    "anthropic.claude-v2",
    "anthropic.claude-instant-v1"
  ]
}

variable "cost_alert_thresholds" {
  description = "Cost alert thresholds as percentages of budget"
  type        = list(number)
  default     = [50, 80, 90, 100]
  
  validation {
    condition     = alltrue([for threshold in var.cost_alert_thresholds : threshold > 0 && threshold <= 200])
    error_message = "All thresholds must be between 1 and 200."
  }
}

variable "allowed_ips" {
  description = "List of IP addresses allowed to access resources (CIDR notation)"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Allow all IPs - restrict in production
}

variable "auto_scaling_config" {
  description = "Auto-scaling configuration for compute resources"
  type = object({
    min_capacity = number
    max_capacity = number
    target_cpu   = number
  })
  default = {
    min_capacity = 1
    max_capacity = 10
    target_cpu   = 70
  }
}

variable "backup_retention" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

variable "multi_az_enabled" {
  description = "Enable Multi-AZ deployment for high availability"
  type        = bool
  default     = false  # Set to true for production
}

variable "enable_cost_optimization" {
  description = "Enable cost optimization features"
  type        = bool
  default     = true
}

variable "development_mode" {
  description = "Enable development mode with relaxed security and cost controls"
  type        = bool
  default     = true
}
