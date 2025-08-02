# Amazon Bedrock Infrastructure as Code (Terraform)
# This configuration sets up the required AWS resources for BERT fine-tuning with Amazon Bedrock

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Configure the AWS Provider
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "BERT-Fine-Tuning"
      Environment = var.environment
      ManagedBy   = "Terraform"
      CostCenter  = "ML-Research"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "bert-fine-tuning"
}

variable "max_budget_amount" {
  description = "Maximum budget amount in USD"
  type        = number
  default     = 100
}

variable "notification_email" {
  description = "Email for budget and cost alerts"
  type        = string
  default     = "your-email@example.com"
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# S3 bucket for training data and model artifacts
resource "aws_s3_bucket" "bedrock_training_bucket" {
  bucket = "${var.project_name}-${var.environment}-${random_id.bucket_suffix.hex}"
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 bucket versioning
resource "aws_s3_bucket_versioning" "bedrock_training_bucket_versioning" {
  bucket = aws_s3_bucket.bedrock_training_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 bucket encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "bedrock_training_bucket_encryption" {
  bucket = aws_s3_bucket.bedrock_training_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 bucket public access block
resource "aws_s3_bucket_public_access_block" "bedrock_training_bucket_pab" {
  bucket = aws_s3_bucket.bedrock_training_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM role for Amazon Bedrock
resource "aws_iam_role" "bedrock_execution_role" {
  name = "${var.project_name}-bedrock-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "bedrock.amazonaws.com"
        }
      }
    ]
  })
}

# IAM policy for Bedrock execution role
resource "aws_iam_role_policy" "bedrock_execution_policy" {
  name = "${var.project_name}-bedrock-execution-policy"
  role = aws_iam_role.bedrock_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.bedrock_training_bucket.arn,
          "${aws_s3_bucket.bedrock_training_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      }
    ]
  })
}

# IAM role for local development/CI-CD
resource "aws_iam_role" "bedrock_developer_role" {
  name = "${var.project_name}-developer-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Condition = {
          StringEquals = {
            "sts:ExternalId" = "bedrock-fine-tuning"
          }
        }
      }
    ]
  })
}

# IAM policy for developer role
resource "aws_iam_role_policy" "bedrock_developer_policy" {
  name = "${var.project_name}-developer-policy"
  role = aws_iam_role.bedrock_developer_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:*",
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
          "iam:PassRole"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      }
    ]
  })
}

# CloudWatch Log Group for Bedrock fine-tuning
resource "aws_cloudwatch_log_group" "bedrock_logs" {
  name              = "/aws/bedrock/${var.project_name}"
  retention_in_days = 14

  tags = {
    Purpose = "Bedrock fine-tuning logs"
  }
}

# SNS topic for alerts
resource "aws_sns_topic" "bedrock_alerts" {
  name = "${var.project_name}-alerts"
}

# SNS subscription for email alerts
resource "aws_sns_topic_subscription" "email_alerts" {
  topic_arn = aws_sns_topic.bedrock_alerts.arn
  protocol  = "email"
  endpoint  = var.notification_email
}

# Budget for cost control
resource "aws_budgets_budget" "bedrock_budget" {
  name       = "${var.project_name}-budget"
  budget_type = "COST"
  limit_amount = var.max_budget_amount
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  time_period_start = "2024-01-01_00:00"

  cost_filters {
    service = ["Amazon Bedrock"]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 80
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = [var.notification_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.notification_email]
  }
}

# CloudWatch alarm for high costs
resource "aws_cloudwatch_metric_alarm" "high_cost_alarm" {
  alarm_name          = "${var.project_name}-high-cost-alarm"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"  # 24 hours
  statistic           = "Maximum"
  threshold           = var.max_budget_amount * 0.8
  alarm_description   = "This metric monitors estimated charges for Bedrock fine-tuning"
  alarm_actions       = [aws_sns_topic.bedrock_alerts.arn]

  dimensions = {
    Currency = "USD"
  }
}

# Outputs
output "s3_bucket_name" {
  description = "Name of the S3 bucket for training data"
  value       = aws_s3_bucket.bedrock_training_bucket.bucket
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.bedrock_training_bucket.arn
}

output "bedrock_execution_role_arn" {
  description = "ARN of the Bedrock execution role"
  value       = aws_iam_role.bedrock_execution_role.arn
}

output "developer_role_arn" {
  description = "ARN of the developer role for local access"
  value       = aws_iam_role.bedrock_developer_role.arn
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = aws_cloudwatch_log_group.bedrock_logs.name
}

output "sns_topic_arn" {
  description = "ARN of the SNS topic for alerts"
  value       = aws_sns_topic.bedrock_alerts.arn
}

# Local file outputs for easy reference
resource "local_file" "aws_config" {
  content = templatefile("${path.module}/templates/aws_config.tpl", {
    bucket_name = aws_s3_bucket.bedrock_training_bucket.bucket
    role_arn    = aws_iam_role.bedrock_execution_role.arn
    region      = var.aws_region
  })
  filename = "${path.module}/../config/aws_config.json"
}
