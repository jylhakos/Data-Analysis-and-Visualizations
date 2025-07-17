#!/bin/bash

BUCKET_NAME=${1:-"your-bucket-name"}
AWS_REGION=${2:-"us-east-1"}

echo "📤 Uploading AirTraffic data to S3 bucket: $BUCKET_NAME"

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ AWS credentials not configured. Run 'aws configure'"
    exit 1
fi

# Create bucket if it doesn't exist
echo "🪣 Creating S3 bucket if it doesn't exist..."
aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION 2>/dev/null || echo "Bucket may already exist"

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket $BUCKET_NAME \
    --versioning-configuration Status=Enabled

# Upload data files
echo "📊 Uploading CSV data files..."
aws s3 cp 2008.csv s3://$BUCKET_NAME/data/2008.csv 2>/dev/null || echo "⚠️  2008.csv not found"
aws s3 cp 2008_sample.csv s3://$BUCKET_NAME/data/2008_sample.csv 2>/dev/null || echo "⚠️  Sample file not found"
aws s3 cp 2008_testsample.csv s3://$BUCKET_NAME/data/2008_testsample.csv 2>/dev/null || echo "⚠️  Test sample file not found"
aws s3 cp 2008_testsample2.csv s3://$BUCKET_NAME/data/2008_testsample2.csv 2>/dev/null || echo "⚠️  Test sample 2 file not found"
aws s3 cp carriers.csv s3://$BUCKET_NAME/data/carriers.csv
aws s3 cp airports.csv s3://$BUCKET_NAME/data/airports.csv

# Upload notebooks
echo "📓 Uploading Jupyter notebooks..."
aws s3 cp AirTrafficProcessor.ipynb s3://$BUCKET_NAME/notebooks/AirTrafficProcessor.ipynb

# Upload scripts
echo "🔧 Uploading utility scripts..."
aws s3 cp verify_pyspark_mllib.py s3://$BUCKET_NAME/scripts/verify_pyspark_mllib.py 2>/dev/null || echo "⚠️  verify_pyspark_mllib.py not found"
aws s3 cp quick_verify.py s3://$BUCKET_NAME/scripts/quick_verify.py 2>/dev/null || echo "⚠️  quick_verify.py not found"

# Upload documentation
echo "📚 Uploading documentation..."
aws s3 cp README.md s3://$BUCKET_NAME/docs/README.md 2>/dev/null || echo "⚠️  README.md not found"
aws s3 cp aws-deployment/AMAZON-AWS-DEPLOYMENT.md s3://$BUCKET_NAME/docs/AMAZON-AWS-DEPLOYMENT.md 2>/dev/null || echo "⚠️  AWS deployment docs not found"

# Create a manifest file with all uploaded files
echo "📋 Creating file manifest..."
cat > file-manifest.json << EOF
{
  "upload_timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "bucket": "$BUCKET_NAME",
  "files": {
    "data": [
      "2008.csv",
      "2008_sample.csv", 
      "2008_testsample.csv",
      "2008_testsample2.csv",
      "carriers.csv",
      "airports.csv"
    ],
    "notebooks": [
      "AirTrafficProcessor.ipynb"
    ],
    "scripts": [
      "verify_pyspark_mllib.py",
      "quick_verify.py"
    ],
    "documentation": [
      "README.md",
      "AWS-DEPLOYMENT.md"
    ]
  }
}
EOF

aws s3 cp file-manifest.json s3://$BUCKET_NAME/manifest.json

# Set bucket policies for security
echo "🔐 Setting bucket security policies..."
cat > bucket-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "DenyUnSecureCommunications",
            "Effect": "Deny",
            "Principal": "*",
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::$BUCKET_NAME",
                "arn:aws:s3:::$BUCKET_NAME/*"
            ],
            "Condition": {
                "Bool": {
                    "aws:SecureTransport": "false"
                }
            }
        }
    ]
}
EOF

aws s3api put-bucket-policy --bucket $BUCKET_NAME --policy file://bucket-policy.json

# Enable encryption
aws s3api put-bucket-encryption \
    --bucket $BUCKET_NAME \
    --server-side-encryption-configuration '{
        "Rules": [
            {
                "ApplyServerSideEncryptionByDefault": {
                    "SSEAlgorithm": "AES256"
                }
            }
        ]
    }'

# List uploaded files
echo "📁 Files uploaded to S3:"
aws s3 ls s3://$BUCKET_NAME --recursive --human-readable --summarize

echo "✅ Data upload complete!"
echo "🌐 S3 Browser URL: https://s3.console.aws.amazon.com/s3/buckets/$BUCKET_NAME"
echo "📊 Bucket: $BUCKET_NAME"

# Clean up temporary files
rm -f file-manifest.json bucket-policy.json

# Show access instructions
echo ""
echo "📋 Access Instructions:"
echo "1. EMR: Data will be automatically available in /mnt/s3/$BUCKET_NAME/"
echo "2. SageMaker: Use boto3 to access s3://$BUCKET_NAME/"
echo "3. EC2: Install AWS CLI and use 'aws s3 sync s3://$BUCKET_NAME/data/ ./data/'"
