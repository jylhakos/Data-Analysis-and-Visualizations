#!/bin/bash
"""
AWS IAM Role Setup Script for Arcee Agent SageMaker Deployment

This script creates the necessary IAM roles and policies for deploying
Arcee Agent to AWS SageMaker.
"""

set -e

# Configuration
ROLE_NAME="ArceeAgentSageMakerRole"
POLICY_NAME="ArceeAgentSageMakerPolicy"
TRUST_POLICY_FILE="/tmp/sagemaker-trust-policy.json"
POLICY_FILE="/tmp/sagemaker-policy.json"

echo "Setting up IAM roles for Arcee Agent SageMaker deployment..."

# Create trust policy for SageMaker
cat > $TRUST_POLICY_FILE << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF

# Create policy document
cat > $POLICY_FILE << EOF
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
                "arn:aws:s3:::arcee-agent-*",
                "arn:aws:s3:::arcee-agent-*/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams"
            ],
            "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData",
                "cloudwatch:GetMetricStatistics",
                "cloudwatch:ListMetrics"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:*"
            ],
            "Resource": "*"
        }
    ]
}
EOF

# Check if role already exists
if aws iam get-role --role-name $ROLE_NAME >/dev/null 2>&1; then
    echo "Role $ROLE_NAME already exists. Updating policy..."
    aws iam put-role-policy --role-name $ROLE_NAME --policy-name $POLICY_NAME --policy-document file://$POLICY_FILE
else
    echo "Creating IAM role: $ROLE_NAME"
    aws iam create-role --role-name $ROLE_NAME --assume-role-policy-document file://$TRUST_POLICY_FILE
    
    echo "Attaching policy to role..."
    aws iam put-role-policy --role-name $ROLE_NAME --policy-name $POLICY_NAME --policy-document file://$POLICY_FILE
fi

# Get role ARN
ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)
echo "Role ARN: $ROLE_ARN"

# Clean up temporary files
rm -f $TRUST_POLICY_FILE $POLICY_FILE

echo "IAM setup completed successfully!"
echo "Role ARN: $ROLE_ARN"
echo ""
echo "Save this role ARN for use in deployment scripts."
