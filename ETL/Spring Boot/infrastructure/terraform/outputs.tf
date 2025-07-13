output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "public_subnet_ids" {
  description = "List of public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "List of private subnet IDs"
  value       = aws_subnet.private[*].id
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.main.id
}

output "nat_gateway_ids" {
  description = "List of NAT Gateway IDs"
  value       = aws_nat_gateway.main[*].id
}

output "data_lake_bucket_name" {
  description = "Name of the S3 data lake bucket"
  value       = aws_s3_bucket.data_lake.bucket
}

output "data_lake_bucket_arn" {
  description = "ARN of the S3 data lake bucket"
  value       = aws_s3_bucket.data_lake.arn
}

output "athena_results_bucket_name" {
  description = "Name of the S3 bucket for Athena query results"
  value       = aws_s3_bucket.athena_results.bucket
}

output "athena_workgroup_name" {
  description = "Name of the Athena workgroup"
  value       = aws_athena_workgroup.weather_workgroup.name
}

output "glue_database_name" {
  description = "Name of the Glue catalog database"
  value       = aws_glue_catalog_database.weather_database.name
}

output "glue_table_name" {
  description = "Name of the Glue catalog table for weather measurements"
  value       = aws_glue_catalog_table.weather_measurements.name
}

output "msk_cluster_arn" {
  description = "ARN of the MSK cluster"
  value       = aws_msk_cluster.weather_kafka.arn
}

output "msk_cluster_name" {
  description = "Name of the MSK cluster"
  value       = aws_msk_cluster.weather_kafka.cluster_name
}

output "msk_bootstrap_brokers" {
  description = "Bootstrap brokers for MSK cluster"
  value       = aws_msk_cluster.weather_kafka.bootstrap_brokers
}

output "msk_bootstrap_brokers_tls" {
  description = "TLS bootstrap brokers for MSK cluster"
  value       = aws_msk_cluster.weather_kafka.bootstrap_brokers_tls
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.weather_cluster.name
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster"
  value       = aws_ecs_cluster.weather_cluster.arn
}

output "ecs_execution_role_arn" {
  description = "ARN of the ECS execution role"
  value       = aws_iam_role.ecs_execution_role.arn
}

output "ecs_task_role_arn" {
  description = "ARN of the ECS task role"
  value       = aws_iam_role.ecs_task_role.arn
}

output "glue_etl_role_arn" {
  description = "ARN of the Glue ETL role"
  value       = aws_iam_role.glue_etl_role.arn
}

output "alb_security_group_id" {
  description = "ID of the Application Load Balancer security group"
  value       = aws_security_group.alb.id
}

output "ecs_security_group_id" {
  description = "ID of the ECS security group"
  value       = aws_security_group.ecs.id
}

output "msk_security_group_id" {
  description = "ID of the MSK security group"
  value       = aws_security_group.msk.id
}

output "weather_stations_config" {
  description = "Configuration of weather stations"
  value       = var.weather_stations
}

output "kafka_topics_config" {
  description = "Configuration of Kafka topics"
  value       = var.kafka_topics
}

output "mqtt_broker_config" {
  description = "MQTT broker configuration"
  value       = var.mqtt_broker_config
}

output "region" {
  description = "AWS region"
  value       = var.aws_region
}

output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "project_name" {
  description = "Project name"
  value       = var.project_name
}

# CloudWatch Log Groups
output "msk_log_group_name" {
  description = "Name of the MSK CloudWatch log group"
  value       = aws_cloudwatch_log_group.msk.name
}

# VPC Endpoints
output "s3_vpc_endpoint_id" {
  description = "ID of the S3 VPC endpoint"
  value       = aws_vpc_endpoint.s3.id
}

# Deployment Information
output "deployment_commands" {
  description = "Commands to deploy the application"
  value = {
    docker_build = "docker build -t weather-etl-microservice ."
    docker_tag   = "docker tag weather-etl-microservice:latest ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/weather-etl-microservice:latest"
    docker_push  = "docker push ${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/weather-etl-microservice:latest"
    
    terraform_init    = "terraform init"
    terraform_plan    = "terraform plan"
    terraform_apply   = "terraform apply"
    terraform_destroy = "terraform destroy"
  }
}

output "useful_aws_cli_commands" {
  description = "Useful AWS CLI commands for this deployment"
  value = {
    describe_msk_cluster = "aws kafka describe-cluster --cluster-arn ${aws_msk_cluster.weather_kafka.arn}"
    list_glue_tables     = "aws glue get-tables --database-name ${aws_glue_catalog_database.weather_database.name}"
    query_athena         = "aws athena start-query-execution --query-string 'SELECT * FROM ${aws_glue_catalog_table.weather_measurements.name} LIMIT 10;' --work-group ${aws_athena_workgroup.weather_workgroup.name}"
    list_s3_objects      = "aws s3 ls s3://${aws_s3_bucket.data_lake.bucket}/ --recursive"
    ecs_services         = "aws ecs list-services --cluster ${aws_ecs_cluster.weather_cluster.name}"
  }
}
