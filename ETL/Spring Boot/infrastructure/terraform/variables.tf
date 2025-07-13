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
  description = "Name of the project"
  type        = string
  default     = "weather-etl"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

variable "enable_vpn_gateway" {
  description = "Enable VPN Gateway"
  type        = bool
  default     = false
}

variable "msk_instance_type" {
  description = "Instance type for MSK brokers"
  type        = string
  default     = "kafka.t3.small"
}

variable "msk_kafka_version" {
  description = "Kafka version for MSK cluster"
  type        = string
  default     = "3.4.0"
}

variable "ecs_cluster_capacity_providers" {
  description = "List of capacity providers for ECS cluster"
  type        = list(string)
  default     = ["FARGATE", "FARGATE_SPOT"]
}

variable "enable_container_insights" {
  description = "Enable CloudWatch Container Insights for ECS cluster"
  type        = bool
  default     = true
}

variable "glue_version" {
  description = "AWS Glue version"
  type        = string
  default     = "4.0"
}

variable "athena_query_result_retention_days" {
  description = "Number of days to retain Athena query results"
  type        = number
  default     = 30
}

variable "cloudwatch_log_retention_days" {
  description = "Number of days to retain CloudWatch logs"
  type        = number
  default     = 7
}

variable "enable_s3_encryption" {
  description = "Enable S3 bucket encryption"
  type        = bool
  default     = true
}

variable "enable_s3_versioning" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

variable "weather_stations" {
  description = "Configuration for weather stations"
  type = map(object({
    name      = string
    latitude  = number
    longitude = number
    altitude  = number
    city      = string
    country   = string
  }))
  default = {
    "station-001" = {
      name      = "Central Station"
      latitude  = 40.7128
      longitude = -74.0060
      altitude  = 10.0
      city      = "New York"
      country   = "USA"
    }
    "station-002" = {
      name      = "North Station"
      latitude  = 40.8176
      longitude = -73.9782
      altitude  = 15.0
      city      = "Bronx"
      country   = "USA"
    }
    "station-003" = {
      name      = "South Station"
      latitude  = 40.6892
      longitude = -74.0445
      altitude  = 5.0
      city      = "Jersey City"
      country   = "USA"
    }
    "station-004" = {
      name      = "East Station"
      latitude  = 40.7282
      longitude = -73.7949
      altitude  = 20.0
      city      = "Queens"
      country   = "USA"
    }
    "station-005" = {
      name      = "West Station"
      latitude  = 40.7505
      longitude = -74.1348
      altitude  = 8.0
      city      = "Newark"
      country   = "USA"
    }
  }
}

variable "mqtt_broker_config" {
  description = "MQTT broker configuration"
  type = object({
    port     = number
    protocol = string
    topics   = list(string)
  })
  default = {
    port     = 1883
    protocol = "tcp"
    topics   = ["weather/measurements", "weather/alerts"]
  }
}

variable "kafka_topics" {
  description = "Kafka topics configuration"
  type = map(object({
    partitions         = number
    replication_factor = number
    retention_ms       = number
  }))
  default = {
    "weather-measurements" = {
      partitions         = 6
      replication_factor = 3
      retention_ms       = 604800000 # 7 days
    }
    "weather-alerts" = {
      partitions         = 3
      replication_factor = 3
      retention_ms       = 259200000 # 3 days
    }
    "processed-measurements" = {
      partitions         = 6
      replication_factor = 3
      retention_ms       = 2592000000 # 30 days
    }
  }
}

variable "flink_config" {
  description = "Apache Flink configuration"
  type = object({
    parallelism       = number
    checkpointing_interval = number
    restart_strategy  = string
  })
  default = {
    parallelism       = 3
    checkpointing_interval = 60000 # 1 minute
    restart_strategy  = "fixed-delay"
  }
}

variable "monitoring_config" {
  description = "Monitoring and alerting configuration"
  type = object({
    enable_detailed_monitoring = bool
    alert_email                = string
    cloudwatch_namespace       = string
  })
  default = {
    enable_detailed_monitoring = true
    alert_email                = "admin@example.com"
    cloudwatch_namespace       = "WeatherETL"
  }
}
