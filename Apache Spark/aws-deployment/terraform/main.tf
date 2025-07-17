# Provider configuration
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "airtraffic-analysis"
}

# S3 Bucket for data and notebooks
resource "aws_s3_bucket" "airtraffic_data" {
  bucket = "${var.project_name}-data-${random_string.suffix.result}"
}

resource "aws_s3_bucket_public_access_block" "airtraffic_data" {
  bucket = aws_s3_bucket.airtraffic_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "airtraffic_data" {
  bucket = aws_s3_bucket.airtraffic_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_versioning" "airtraffic_data" {
  bucket = aws_s3_bucket.airtraffic_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# VPC for EMR cluster
resource "aws_vpc" "emr_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

resource "aws_internet_gateway" "emr_igw" {
  vpc_id = aws_vpc.emr_vpc.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

resource "aws_subnet" "emr_subnet" {
  vpc_id                  = aws_vpc.emr_vpc.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-subnet"
  }
}

resource "aws_route_table" "emr_route_table" {
  vpc_id = aws_vpc.emr_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.emr_igw.id
  }

  tags = {
    Name = "${var.project_name}-route-table"
  }
}

resource "aws_route_table_association" "emr_route_table_association" {
  subnet_id      = aws_subnet.emr_subnet.id
  route_table_id = aws_route_table.emr_route_table.id
}

data "aws_availability_zones" "available" {
  state = "available"
}

# Security Groups
resource "aws_security_group" "emr_master" {
  name_prefix = "${var.project_name}-emr-master"
  vpc_id      = aws_vpc.emr_vpc.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 9443
    to_port     = 9443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-emr-master-sg"
  }
}

resource "aws_security_group" "emr_slave" {
  name_prefix = "${var.project_name}-emr-slave"
  vpc_id      = aws_vpc.emr_vpc.id

  ingress {
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [aws_security_group.emr_master.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-emr-slave-sg"
  }
}

# IAM Role for EMR Service
resource "aws_iam_role" "emr_service_role" {
  name = "${var.project_name}-emr-service-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "elasticmapreduce.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "emr_service_role_policy" {
  role       = aws_iam_role.emr_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceRole"
}

# IAM Role for EMR EC2 instances
resource "aws_iam_role" "emr_instance_profile_role" {
  name = "${var.project_name}-emr-instance-profile-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "emr_instance_profile_policy" {
  role       = aws_iam_role.emr_instance_profile_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonElasticMapReduceforEC2Role"
}

# Additional policy for S3 access
resource "aws_iam_policy" "emr_s3_policy" {
  name        = "${var.project_name}-emr-s3-policy"
  description = "S3 access policy for EMR cluster"

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
          aws_s3_bucket.airtraffic_data.arn,
          "${aws_s3_bucket.airtraffic_data.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "emr_s3_policy_attachment" {
  role       = aws_iam_role.emr_instance_profile_role.name
  policy_arn = aws_iam_policy.emr_s3_policy.arn
}

resource "aws_iam_instance_profile" "emr_instance_profile" {
  name = "${var.project_name}-emr-instance-profile"
  role = aws_iam_role.emr_instance_profile_role.name
}

# Key Pair for SSH access
resource "aws_key_pair" "emr_key_pair" {
  key_name   = "${var.project_name}-key"
  public_key = file("~/.ssh/id_rsa.pub")
}

# EMR Cluster
resource "aws_emr_cluster" "airtraffic_cluster" {
  name          = "${var.project_name}-emr-cluster"
  release_label = "emr-6.15.0"
  applications  = ["Spark", "Hadoop", "JupyterHub", "Zeppelin"]

  ec2_attributes {
    subnet_id                         = aws_subnet.emr_subnet.id
    emr_managed_master_security_group = aws_security_group.emr_master.id
    emr_managed_slave_security_group  = aws_security_group.emr_slave.id
    instance_profile                  = aws_iam_instance_profile.emr_instance_profile.arn
    key_name                         = aws_key_pair.emr_key_pair.key_name
  }

  master_instance_group {
    instance_type = "m5.xlarge"
  }

  core_instance_group {
    instance_type  = "m5.large"
    instance_count = 2

    ebs_config {
      size                 = "40"
      type                 = "gp3"
      volumes_per_instance = 1
    }
  }

  service_role = aws_iam_role.emr_service_role.arn

  configurations_json = jsonencode([
    {
      Classification = "jupyter-s3-conf"
      Properties = {
        "s3.persistence.enabled" = "true"
        "s3.persistence.bucket"  = aws_s3_bucket.airtraffic_data.bucket
      }
    },
    {
      Classification = "spark-defaults"
      Properties = {
        "spark.driver.memory"           = "4g"
        "spark.driver.maxResultSize"    = "2g"
        "spark.sql.adaptive.enabled"    = "true"
        "spark.sql.adaptive.coalescePartitions.enabled" = "true"
      }
    }
  ])

  auto_termination_policy {
    idle_timeout = 3600  # Auto-terminate after 1 hour of inactivity
  }

  tags = {
    Name        = "${var.project_name}-emr-cluster"
    Environment = "development"
  }
}

# Outputs
output "emr_master_public_dns" {
  description = "EMR Master public DNS"
  value       = aws_emr_cluster.airtraffic_cluster.master_public_dns
}

output "emr_cluster_id" {
  description = "EMR Cluster ID"
  value       = aws_emr_cluster.airtraffic_cluster.id
}

output "jupyter_url" {
  description = "JupyterHub URL"
  value       = "https://${aws_emr_cluster.airtraffic_cluster.master_public_dns}:9443"
}

output "zeppelin_url" {
  description = "Zeppelin URL"
  value       = "http://${aws_emr_cluster.airtraffic_cluster.master_public_dns}:8890"
}

output "s3_bucket_name" {
  description = "S3 bucket for data storage"
  value       = aws_s3_bucket.airtraffic_data.bucket
}

output "ssh_command" {
  description = "SSH command to connect to master node"
  value       = "ssh -i ~/.ssh/id_rsa hadoop@${aws_emr_cluster.airtraffic_cluster.master_public_dns}"
}
