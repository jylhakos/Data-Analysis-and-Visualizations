#!/bin/bash

# AWS Glue Infrastructure Deployment Script
# =========================================
# This script deploys the complete AWS infrastructure for the weather ETL pipeline
# including S3 buckets, IAM roles, Glue jobs, and other required resources.

set -e  # Exit on any error

# Configuration
STACK_NAME="weather-etl-glue-stack"
REGION="us-east-1"
S3_BUCKET="weather-data-lake"
GLUE_DATABASE="weather_analytics_db"
PROJECT_NAME="weather-etl-pipeline"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if AWS CLI is installed and configured
check_aws_cli() {
    log_info "Checking AWS CLI configuration..."
    
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS CLI is not configured. Please run 'aws configure' first."
        exit 1
    fi
    
    log_info "AWS CLI is properly configured."
}

# Create S3 bucket for data lake
create_s3_bucket() {
    log_info "Creating S3 bucket: $S3_BUCKET"
    
    if aws s3 ls "s3://$S3_BUCKET" 2>&1 | grep -q 'NoSuchBucket'; then
        aws s3 mb "s3://$S3_BUCKET" --region "$REGION"
        log_info "S3 bucket created successfully."
    else
        log_warn "S3 bucket already exists."
    fi
    
    # Create folder structure
    log_info "Creating S3 folder structure..."
    aws s3api put-object --bucket "$S3_BUCKET" --key "raw/weather_data/" --region "$REGION"
    aws s3api put-object --bucket "$S3_BUCKET" --key "processed/weather_data/" --region "$REGION"
    aws s3api put-object --bucket "$S3_BUCKET" --key "aggregated/weather_data/" --region "$REGION"
    aws s3api put-object --bucket "$S3_BUCKET" --key "scripts/glue_jobs/" --region "$REGION"
    aws s3api put-object --bucket "$S3_BUCKET" --key "checkpoints/streaming/" --region "$REGION"
    aws s3api put-object --bucket "$S3_BUCKET" --key "temp/redshift/" --region "$REGION"
}

# Create IAM role for AWS Glue
create_glue_iam_role() {
    log_info "Creating IAM role for AWS Glue..."
    
    ROLE_NAME="AWSGlueWeatherETLRole"
    POLICY_NAME="WeatherETLGluePolicy"
    
    # Trust policy for Glue service
    cat > /tmp/glue-trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "glue.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

    # Create IAM role
    if ! aws iam get-role --role-name "$ROLE_NAME" &> /dev/null; then
        aws iam create-role \
            --role-name "$ROLE_NAME" \
            --assume-role-policy-document file:///tmp/glue-trust-policy.json \
            --description "IAM role for Weather ETL Glue jobs"
        log_info "IAM role created: $ROLE_NAME"
    else
        log_warn "IAM role already exists: $ROLE_NAME"
    fi
    
    # Attach AWS managed policies
    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
    
    # Create custom policy for S3 and CloudWatch access
    cat > /tmp/weather-etl-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::$S3_BUCKET",
                "arn:aws:s3:::$S3_BUCKET/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "kinesis:DescribeStream",
                "kinesis:GetShardIterator",
                "kinesis:GetRecords",
                "kinesis:ListStreams"
            ],
            "Resource": "*"
        }
    ]
}
EOF

    # Create and attach custom policy
    if ! aws iam get-policy --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/$POLICY_NAME" &> /dev/null; then
        aws iam create-policy \
            --policy-name "$POLICY_NAME" \
            --policy-document file:///tmp/weather-etl-policy.json \
            --description "Custom policy for Weather ETL pipeline"
        log_info "Custom policy created: $POLICY_NAME"
    else
        log_warn "Custom policy already exists: $POLICY_NAME"
    fi
    
    aws iam attach-role-policy \
        --role-name "$ROLE_NAME" \
        --policy-arn "arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/$POLICY_NAME"
    
    # Clean up temporary files
    rm -f /tmp/glue-trust-policy.json /tmp/weather-etl-policy.json
}

# Upload Glue scripts to S3
upload_glue_scripts() {
    log_info "Uploading Glue scripts to S3..."
    
    # Upload ETL job script
    aws s3 cp "aws_glue/weather_etl_glue_job.py" \
        "s3://$S3_BUCKET/scripts/glue_jobs/weather_etl_glue_job.py"
    
    # Upload streaming job script
    aws s3 cp "aws_glue/weather_streaming_etl_job.py" \
        "s3://$S3_BUCKET/scripts/glue_jobs/weather_streaming_etl_job.py"
    
    log_info "Glue scripts uploaded successfully."
}

# Setup AWS Glue Data Catalog
setup_data_catalog() {
    log_info "Setting up AWS Glue Data Catalog..."
    
    # Run the Data Catalog setup script
    python3 aws_glue/setup_data_catalog.py
    
    log_info "Data Catalog setup completed."
}

# Create AWS Glue jobs
create_glue_jobs() {
    log_info "Creating AWS Glue jobs..."
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/AWSGlueWeatherETLRole"
    
    # Create batch ETL job
    aws glue create-job \
        --name "weather-batch-etl-job" \
        --role "$ROLE_ARN" \
        --description "Batch ETL job for weather station data processing" \
        --command '{
            "Name": "glueetl",
            "ScriptLocation": "s3://'$S3_BUCKET'/scripts/glue_jobs/weather_etl_glue_job.py",
            "PythonVersion": "3"
        }' \
        --default-arguments '{
            "--job-language": "python",
            "--job-bookmark-option": "job-bookmark-enable",
            "--enable-metrics": "",
            "--enable-continuous-cloudwatch-log": "true",
            "--enable-spark-ui": "true",
            "--spark-event-logs-path": "s3://'$S3_BUCKET'/sparkHistoryLogs/"
        }' \
        --max-retries 1 \
        --timeout 2880 \
        --glue-version "4.0" \
        --number-of-workers 5 \
        --worker-type "G.1X" \
        --region "$REGION" || log_warn "Batch ETL job might already exist"
    
    # Create streaming ETL job
    aws glue create-job \
        --name "weather-streaming-etl-job" \
        --role "$ROLE_ARN" \
        --description "Streaming ETL job for real-time weather data processing" \
        --command '{
            "Name": "gluestreaming",
            "ScriptLocation": "s3://'$S3_BUCKET'/scripts/glue_jobs/weather_streaming_etl_job.py",
            "PythonVersion": "3"
        }' \
        --default-arguments '{
            "--job-language": "python",
            "--enable-metrics": "",
            "--enable-continuous-cloudwatch-log": "true"
        }' \
        --max-retries 0 \
        --timeout 2880 \
        --glue-version "4.0" \
        --number-of-workers 5 \
        --worker-type "G.1X" \
        --region "$REGION" || log_warn "Streaming ETL job might already exist"
    
    log_info "Glue jobs created successfully."
}

# Create Kinesis Data Stream for real-time processing
create_kinesis_stream() {
    log_info "Creating Kinesis Data Stream for real-time weather data..."
    
    STREAM_NAME="weather-data-stream"
    
    if ! aws kinesis describe-stream --stream-name "$STREAM_NAME" &> /dev/null; then
        aws kinesis create-stream \
            --stream-name "$STREAM_NAME" \
            --shard-count 2 \
            --region "$REGION"
        log_info "Kinesis stream created: $STREAM_NAME"
        
        # Wait for stream to become active
        log_info "Waiting for Kinesis stream to become active..."
        aws kinesis wait stream-exists --stream-name "$STREAM_NAME"
    else
        log_warn "Kinesis stream already exists: $STREAM_NAME"
    fi
}

# Create CloudWatch dashboard
create_cloudwatch_dashboard() {
    log_info "Creating CloudWatch dashboard for monitoring..."
    
    DASHBOARD_NAME="WeatherETLPipeline"
    
    cat > /tmp/dashboard-body.json << EOF
{
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "AWS/Glue", "glue.driver.aggregate.numCompletedTasks", "JobName", "weather-batch-etl-job" ],
                    [ ".", "glue.driver.aggregate.numFailedTasks", ".", "." ]
                ],
                "period": 300,
                "stat": "Sum",
                "region": "$REGION",
                "title": "Glue Job Task Metrics"
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    [ "WeatherETL/GlueJob", "JobExecutionCount" ],
                    [ ".", "ProcessingLatency" ]
                ],
                "period": 300,
                "stat": "Average",
                "region": "$REGION",
                "title": "Custom ETL Metrics"
            }
        }
    ]
}
EOF

    aws cloudwatch put-dashboard \
        --dashboard-name "$DASHBOARD_NAME" \
        --dashboard-body file:///tmp/dashboard-body.json \
        --region "$REGION"
    
    rm -f /tmp/dashboard-body.json
    log_info "CloudWatch dashboard created: $DASHBOARD_NAME"
}

# Main deployment function
deploy_infrastructure() {
    log_info "🚀 Starting AWS Glue infrastructure deployment..."
    
    check_aws_cli
    create_s3_bucket
    create_glue_iam_role
    upload_glue_scripts
    setup_data_catalog
    create_glue_jobs
    create_kinesis_stream
    create_cloudwatch_dashboard
    
    log_info "✅ AWS Glue infrastructure deployment completed successfully!"
    log_info ""
    log_info "📋 Summary of created resources:"
    log_info "  - S3 Bucket: $S3_BUCKET"
    log_info "  - Glue Database: $GLUE_DATABASE"
    log_info "  - Glue Jobs: weather-batch-etl-job, weather-streaming-etl-job"
    log_info "  - IAM Role: AWSGlueWeatherETLRole"
    log_info "  - Kinesis Stream: weather-data-stream"
    log_info "  - CloudWatch Dashboard: WeatherETLPipeline"
    log_info ""
    log_info "🎯 Next steps:"
    log_info "  1. Run crawlers to discover data schema"
    log_info "  2. Start the streaming ETL job for real-time processing"
    log_info "  3. Schedule the batch ETL job for regular execution"
    log_info "  4. Configure the weather data simulation to send data to Kinesis"
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    deploy_infrastructure
fi
