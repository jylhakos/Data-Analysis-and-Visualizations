#!/bin/bash
# validate_aws_setup.sh - Validation script for AWS setup

echo "🔍 Validating AWS Setup for Arcee Agent SageMaker Deployment..."
echo "=============================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

ERRORS=0

echo ""
log_info "Starting AWS configuration validation..."
echo ""

# Test 1: AWS CLI Installation
echo "1. Testing AWS CLI installation..."
if command -v aws &> /dev/null; then
    AWS_VERSION=$(aws --version 2>&1 | head -n1)
    log_success "AWS CLI installed: $AWS_VERSION"
else
    log_error "AWS CLI not found - install with: sudo apt install awscli"
    ERRORS=$((ERRORS + 1))
fi

# Test 2: AWS Credentials
echo ""
echo "2. Testing AWS credentials..."
if aws sts get-caller-identity >/dev/null 2>&1; then
    log_success "AWS credentials valid"
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    USER_ARN=$(aws sts get-caller-identity --query Arn --output text)
    REGION=$(aws configure get region)
    
    log_info "Account ID: $ACCOUNT_ID"
    log_info "User ARN: $USER_ARN"
    log_info "Default Region: $REGION"
else
    log_error "AWS credentials invalid"
    log_warning "Run: aws configure"
    log_warning "You need: Access Key ID, Secret Access Key, Region (us-east-1), Output (json)"
    ERRORS=$((ERRORS + 1))
fi

# Test 3: SageMaker Access
echo ""
echo "3. Testing SageMaker service access..."
if aws sagemaker list-endpoints >/dev/null 2>&1; then
    log_success "SageMaker access working"
    ENDPOINT_COUNT=$(aws sagemaker list-endpoints --query 'length(Endpoints)' --output text)
    log_info "Current endpoints: $ENDPOINT_COUNT"
else
    log_error "SageMaker access failed"
    log_warning "Check IAM permissions for SageMaker"
    ERRORS=$((ERRORS + 1))
fi

# Test 4: SageMaker Execution Role
echo ""
echo "4. Testing SageMaker execution role..."
ROLE_NAME="ArceeAgentSageMakerExecutionRole"
if aws iam get-role --role-name $ROLE_NAME >/dev/null 2>&1; then
    log_success "SageMaker execution role exists"
    ROLE_ARN=$(aws iam get-role --role-name $ROLE_NAME --query 'Role.Arn' --output text)
    log_info "Role ARN: $ROLE_ARN"
    
    # Check attached policies
    POLICIES=$(aws iam list-attached-role-policies --role-name $ROLE_NAME --query 'AttachedPolicies[].PolicyName' --output text)
    log_info "Attached policies: $POLICIES"
else
    log_error "SageMaker execution role missing"
    log_warning "Run: ./scripts/setup_aws.sh to create the role"
    ERRORS=$((ERRORS + 1))
fi

# Test 5: S3 Access
echo ""
echo "5. Testing S3 access..."
if aws s3 ls >/dev/null 2>&1; then
    log_success "S3 access working"
    
    # Check for project-specific bucket
    S3_BUCKET_NAME="arcee-agent-${ACCOUNT_ID}"
    if aws s3 ls "s3://${S3_BUCKET_NAME}" >/dev/null 2>&1; then
        log_success "Project S3 bucket exists: $S3_BUCKET_NAME"
    else
        log_warning "Project S3 bucket not found: $S3_BUCKET_NAME"
        log_info "Will be created during setup if needed"
    fi
else
    log_error "S3 access failed"
    log_warning "Check IAM permissions for S3"
    ERRORS=$((ERRORS + 1))
fi

# Test 6: ECR Access
echo ""
echo "6. Testing ECR access..."
if aws ecr describe-repositories >/dev/null 2>&1; then
    log_success "ECR access working"
    
    # Check for project-specific repository
    if aws ecr describe-repositories --repository-names arcee-agent >/dev/null 2>&1; then
        ECR_URI=$(aws ecr describe-repositories --repository-names arcee-agent --query 'repositories[0].repositoryUri' --output text)
        log_success "Project ECR repository exists: $ECR_URI"
    else
        log_warning "Project ECR repository not found: arcee-agent"
        log_info "Will be created during setup if needed"
    fi
else
    log_error "ECR access failed"
    log_warning "Check IAM permissions for ECR"
    ERRORS=$((ERRORS + 1))
fi

# Test 7: CloudWatch Logs Access
echo ""
echo "7. Testing CloudWatch Logs access..."
if aws logs describe-log-groups --limit 1 >/dev/null 2>&1; then
    log_success "CloudWatch Logs access working"
else
    log_error "CloudWatch Logs access failed"
    log_warning "Check IAM permissions for CloudWatch Logs"
    ERRORS=$((ERRORS + 1))
fi

# Test 8: Environment Variables
echo ""
echo "8. Checking environment variables..."
if [[ -n "$SAGEMAKER_ROLE_ARN" ]]; then
    log_success "SAGEMAKER_ROLE_ARN is set"
    log_info "Value: $SAGEMAKER_ROLE_ARN"
else
    log_warning "SAGEMAKER_ROLE_ARN not set"
    log_info "Run: source ~/.bashrc after setup"
fi

if [[ -n "$S3_BUCKET_NAME" ]]; then
    log_success "S3_BUCKET_NAME is set"
    log_info "Value: $S3_BUCKET_NAME"
else
    log_warning "S3_BUCKET_NAME not set"
fi

if [[ -n "$ECR_REPOSITORY_URI" ]]; then
    log_success "ECR_REPOSITORY_URI is set"
    log_info "Value: $ECR_REPOSITORY_URI"
else
    log_warning "ECR_REPOSITORY_URI not set"
fi

# Test 9: Check User Permissions
echo ""
echo "9. Checking user IAM permissions..."
USER_NAME=$(aws sts get-caller-identity --query Arn --output text | cut -d'/' -f2)
if aws iam list-attached-user-policies --user-name "$USER_NAME" >/dev/null 2>&1; then
    USER_POLICIES=$(aws iam list-attached-user-policies --user-name "$USER_NAME" --query 'AttachedPolicies[].PolicyName' --output text)
    log_success "User IAM policies found"
    log_info "Attached policies: $USER_POLICIES"
    
    # Check for essential policies
    if echo "$USER_POLICIES" | grep -q "AmazonSageMakerFullAccess"; then
        log_success "SageMaker permissions available"
    else
        log_warning "SageMaker permissions may be missing"
    fi
else
    log_warning "Could not check user permissions"
fi

# Summary
echo ""
echo "=========================================="
echo "🎯 VALIDATION SUMMARY"
echo "=========================================="

if [ $ERRORS -eq 0 ]; then
    log_success "All validations passed! AWS setup is complete."
    echo ""
    echo "🚀 READY FOR SAGEMAKER DEPLOYMENT!"
    echo ""
    echo "Next steps:"
    echo "1. Build Docker image: docker build -t arcee-agent-api ."
    echo "2. Deploy to SageMaker: python scripts/sagemaker_deployment.py"
    echo "3. Test API endpoints: curl http://localhost:8000/health"
    echo ""
    echo "📊 Estimated deployment costs:"
    echo "- Training (one-time): ~$2-10 depending on instance type"
    echo "- Inference endpoint: ~$50-150/month depending on usage"
    echo "- Storage (S3/ECR): ~$1-5/month"
else
    log_error "Found $ERRORS error(s) in AWS configuration"
    echo ""
    echo "🔧 FIXES REQUIRED:"
    echo ""
    if [ $ERRORS -ge 1 ]; then
        echo "1. Fix AWS credentials: aws configure"
        echo "2. Create IAM roles: ./scripts/setup_aws.sh"
        echo "3. Verify permissions: Check IAM policies"
        echo "4. Re-run validation: ./validate_aws_setup.sh"
    fi
fi

echo ""
echo "📖 For troubleshooting help:"
echo "- Check README.md AWS Setup section"
echo "- Run: ./scripts/setup_aws.sh for automated setup"
echo "- Review AWS CloudTrail logs for permission issues"
echo ""

# Clean exit
if [ $ERRORS -eq 0 ]; then
    exit 0
else
    exit 1
fi
