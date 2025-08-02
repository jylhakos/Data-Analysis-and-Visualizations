# AWS Resource Calculator Usage Guide

The AWS Resource Calculator (`aws_resource_calculator.py`) is a comprehensive tool designed to help DevOps teams estimate, allocate, and optimize AWS resources for BERT fine-tuning workloads.

## Start

### Interactive mode (Recommended)
```bash
python aws_resource_calculator.py --interactive
```

This will guide you through:
- Dataset characteristics input
- Budget and time constraints
- Performance requirements
- Generate recommendations and configurations

### Command Line mode
```bash
# Basic usage
python aws_resource_calculator.py \
    --dataset-size 5GB \
    --num-samples 100000 \
    --budget 1000 \
    --duration 24h

# Advanced usage with all options
python aws_resource_calculator.py \
    --dataset-size 10GB \
    --num-samples 250000 \
    --sequence-length 512 \
    --complexity medium \
    --budget 2000 \
    --duration 48h \
    --priority performance \
    --region us-west-2 \
    --generate-terraform \
    --output-format json
```

## Input Parameters

### Dataset Parameters
- **--dataset-size**: Size of your training dataset
  - Examples: `5GB`, `1.5TB`, `500MB`
  - Used to estimate storage and processing requirements

- **--num-samples**: Number of training samples
  - Example: `100000`
  - Used to estimate training time

- **--sequence-length**: Average sequence length (default: 256)
  - Common values: `128`, `256`, `512`, `1024`
  - Affects memory requirements significantly

- **--complexity**: Model complexity level
  - `simple`: BERT-Base (~110M parameters)
  - `medium`: BERT-Large (~340M parameters)
  - `complex`: Custom large models (~1B+ parameters)

### Budget and time constraints
- **--budget**: Maximum budget in USD
  - Example: `1000`
  - Used to filter instance options

- **--duration**: Maximum training duration
  - Examples: `24h`, `3d`, `48h`
  - Used to calculate total costs

### Optimization Parameters
- **--priority**: Optimization priority
  - `cost`: Minimize costs, may sacrifice performance
  - `performance`: Maximize performance, higher costs
  - `balanced`: Balance between cost and performance

- **--region**: AWS region (default: us-east-1)
  - Example: `us-west-2`, `eu-west-1`
  - Affects pricing and instance availability

## Output Options

### Text Output (Default)
Simple, human-readable recommendations:
```
Amazon AWS Resource Recommendations for BERT Fine-tuning
============================================================

  Recommended: P5 12XLarge
   Cost: $590.00
   Time: 24.0h
   Efficiency: 92.5/100
```

### JSON Output
Structured data for automation:
```bash
python aws_resource_calculator.py \
    --dataset-size 5GB \
    --num-samples 100000 \
    --budget 1000 \
    --duration 24h \
    --output-format json
```

### Generate Terraform Configuration
```bash
python aws_resource_calculator.py \
    --dataset-size 5GB \
    --num-samples 100000 \
    --budget 1000 \
    --duration 24h \
    --generate-terraform
```

This creates `generated_terraform.tf` with complete infrastructure setup.

## 🔧 Advanced Features

### 1. Cost Analysis Report
In interactive mode, you can generate detailed cost analysis:
```
Generate detailed cost report? (y/N): y
Cost analysis report saved to 'cost_analysis_report.md'
```

### 2. Storage Optimization
The tool automatically recommends:
- **EBS volumes**: For general storage needs
- **FSx Lustre**: For large datasets requiring high throughput
- **S3 configurations**: For data archival and backup

### 3. Instance Selection Matrix
The tool considers:
- **P5 instances**: Latest generation, best performance
- **P4d instances**: Cost-effective for multi-GPU workloads
- **G5 instances**: Budget-friendly option
- **Trainium instances**: Cost-optimized for specific workloads

## Cost optimization (Examples)

### Small Dataset (< 1GB)
```bash
python aws_resource_calculator.py \
    --dataset-size 500MB \
    --num-samples 10000 \
    --budget 100 \
    --duration 6h \
    --priority cost
```
**Expected Output**: G5 instances, ~$10-20 total cost

### Medium Dataset (1-50GB)
```bash
python aws_resource_calculator.py \
    --dataset-size 10GB \
    --num-samples 100000 \
    --budget 500 \
    --duration 24h \
    --priority balanced
```
**Expected Output**: P4d or P5 instances, ~$200-400 total cost

### Large Dataset (> 50GB)
```bash
python aws_resource_calculator.py \
    --dataset-size 100GB \
    --num-samples 1000000 \
    --budget 2000 \
    --duration 72h \
    --priority performance
```
**Expected Output**: P5 multi-GPU instances, ~$1000-1500 total cost

## Real scenarios

### 1. Development/testing
```bash
# Quick prototype testing
python aws_resource_calculator.py \
    --dataset-size 1GB \
    --num-samples 5000 \
    --budget 50 \
    --duration 4h \
    --priority cost
```

### 2. Production training
```bash
# Production model training
python aws_resource_calculator.py \
    --dataset-size 25GB \
    --num-samples 500000 \
    --sequence-length 512 \
    --complexity medium \
    --budget 1500 \
    --duration 48h \
    --priority balanced \
    --generate-terraform
```

### 3. Experimentation
```bash
# Large-scale research
python aws_resource_calculator.py \
    --dataset-size 200GB \
    --num-samples 2000000 \
    --sequence-length 1024 \
    --complexity complex \
    --budget 5000 \
    --duration 168h \
    --priority performance
```

## Generated files

### Terraform configuration (`generated_terraform.tf`)
- Complete infrastructure setup
- Security groups and networking
- Storage configuration
- Instance provisioning
- Cost tracking tags

### Cost analysis report (`cost_analysis_report.md`)
- Detailed cost breakdown
- Performance optimization tips
- Alternative configurations
- Cost optimization strategies

## 🛠️ Integration with existing workflow

### 1. CI/CD integration
```yaml
# .github/workflows/estimate-costs.yml
name: Estimate Training Costs
on:
  push:
    paths: ['data/**', 'config/**']

jobs:
  estimate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Calculate Resources
        run: |
          python aws_resource_calculator.py \
            --dataset-size ${{ env.DATASET_SIZE }} \
            --num-samples ${{ env.NUM_SAMPLES }} \
            --budget ${{ env.MAX_BUDGET }} \
            --duration ${{ env.MAX_DURATION }} \
            --output-format json > cost-estimate.json
      - name: Upload Cost Estimate
        uses: actions/upload-artifact@v3
        with:
          name: cost-estimate
          path: cost-estimate.json
```

### 2. Terraform integration
```bash
# Generate Terraform configuration
python aws_resource_calculator.py \
    --dataset-size 10GB \
    --num-samples 100000 \
    --budget 1000 \
    --duration 24h \
    --generate-terraform

# Deploy infrastructure
terraform init
terraform plan -var="project_name=bert-training-prod"
terraform apply
```

### 3. Cost monitoring
```bash
# Set up automated cost monitoring
python cost_monitor.py --budget 1000 --alert-threshold 80
```

## Troubleshooting

### Issues

1. **"No suitable instances found"**
   - Increase budget or extend duration
   - Reduce sequence length or model complexity
   - Consider cost-optimized instances

2. **High estimated costs**
   - Use spot instances (up to 90% savings)
   - Reduce sequence length
   - Use gradient accumulation instead of larger batch sizes

3. **Long training times**
   - Use multi-GPU instances (P5.48xlarge)
   - Implement distributed training
   - Optimize data loading with FSx Lustre


For current pricing, consider using AWS Pricing API integration.

```python
# Tool supports real-time pricing (requires AWS credentials)
calculator = AWSResourceCalculator(region='us-east-1')
# Will fetch real-time pricing if AWS credentials available
```
### Performance optimization

1. **Memory optimization**
   ```bash
   # Use gradient accumulation for memory efficiency
   --sequence-length 256  # Instead of 512
   # Tool will recommend appropriate batch sizes
   ```

2. **Storage optimization**
   ```bash
   # For large datasets, tool recommends FSx Lustre
   # Automatically configured in Terraform output
   ```

3. **Cost optimization**
   ```bash
   # Use cost priority for development
   --priority cost
   
   # Use balanced for production
   --priority balanced
   ```

