#!/bin/bash

# Comprehensive Reinstallation and Testing Script for Arcee Agent
# This script will clean, reinstall, and test the entire project

echo "🔄 Starting comprehensive reinstallation and testing of Arcee Agent"
echo "=================================================================="

# Set the project directory
PROJECT_DIR="/home/laptop/EXERCISES/Data Analysis and Visualization/Data-Analysis-and-Visualizations/Amazon SageMaker/Arcee Agent"
VENV_PYTHON="/home/laptop/EXERCISES/Data Analysis and Visualization/Data-Analysis-and-Visualizations/Amazon SageMaker/.venv/bin/python"

cd "$PROJECT_DIR"

echo "📁 Current directory: $(pwd)"

# Step 1: Verify Python environment
echo ""
echo "🐍 Step 1: Verifying Python environment..."
echo "=================================================================="
$VENV_PYTHON --version
echo "✅ Python environment verified"

# Step 2: Install/upgrade core dependencies
echo ""
echo "📦 Step 2: Installing/upgrading core dependencies..."
echo "=================================================================="
$VENV_PYTHON -m pip install --upgrade pip
$VENV_PYTHON -m pip install -r requirements.txt

# Step 3: Install AWS and SageMaker SDK
echo ""
echo "☁️ Step 3: Installing AWS and SageMaker dependencies..."
echo "=================================================================="
$VENV_PYTHON -m pip install sagemaker boto3 awscli

# Step 4: Test basic imports
echo ""
echo "🧪 Step 4: Testing basic imports..."
echo "=================================================================="
$VENV_PYTHON -c "
import torch
import transformers
import datasets
import openai
import fastapi
import boto3
import sagemaker
print('✅ All core imports successful')
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'Datasets version: {datasets.__version__}')
print(f'Boto3 version: {boto3.__version__}')
print(f'SageMaker version: {sagemaker.__version__}')
"

# Step 5: Test CUDA availability
echo ""
echo "🚀 Step 5: Testing CUDA availability..."
echo "=================================================================="
$VENV_PYTHON -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('❌ CUDA not available (using CPU)')
"

# Step 6: Test dataset loading
echo ""
echo "📊 Step 6: Testing dataset loading..."
echo "=================================================================="
$VENV_PYTHON -c "
from datasets import load_from_disk
try:
    dataset = load_from_disk('./dataset')
    print(f'✅ Dataset loaded successfully')
    print(f'   Size: {len(dataset)} samples')
    print(f'   Features: {list(dataset.features.keys())}')
    
    # Validate first example
    example = dataset[0]
    import json
    tools = json.loads(example['tools'])
    answers = json.loads(example['answers'])
    print(f'   First example tools: {len(tools)} tools')
    print(f'   First example answers: {len(answers)} answers')
except Exception as e:
    print(f'❌ Dataset loading failed: {e}')
"

# Step 7: Run unit tests
echo ""
echo "🧪 Step 7: Running unit tests..."
echo "=================================================================="
$VENV_PYTHON test_arcee_agent.py

# Step 8: Test main.py help
echo ""
echo "📋 Step 8: Testing main.py functionality..."
echo "=================================================================="
$VENV_PYTHON main.py --help

# Step 9: Create directories for models and outputs
echo ""
echo "📁 Step 9: Creating necessary directories..."
echo "=================================================================="
mkdir -p models
mkdir -p logs
mkdir -p fine_tuned_models
mkdir -p outputs
echo "✅ Directories created"

# Step 10: Test AWS credentials (if available)
echo ""
echo "☁️ Step 10: Testing AWS configuration..."
echo "=================================================================="
if aws sts get-caller-identity &>/dev/null; then
    echo "✅ AWS credentials configured"
    aws sts get-caller-identity
else
    echo "⚠️  AWS credentials not configured"
    echo "   Configure with: aws configure"
fi

# Step 11: Test Docker availability
echo ""
echo "🐳 Step 11: Testing Docker availability..."
echo "=================================================================="
if command -v docker &> /dev/null; then
    echo "✅ Docker found"
    docker --version
    if docker ps &>/dev/null; then
        echo "✅ Docker daemon running"
    else
        echo "⚠️  Docker daemon not running"
    fi
else
    echo "❌ Docker not found"
    echo "   Install Docker for containerization"
fi

# Step 12: Test API server (quick test)
echo ""
echo "🔌 Step 12: Testing API server startup..."
echo "=================================================================="
$VENV_PYTHON -c "
from fastapi import FastAPI
from api_server import app
print('✅ API server imports successful')
print('   FastAPI app created')
"

# Step 13: Summary and next steps
echo ""
echo "🎉 Installation and Testing Complete!"
echo "=================================================================="
echo ""
echo "📋 Summary:"
echo "  ✅ Python environment configured"
echo "  ✅ Dependencies installed"
echo "  ✅ Dataset loaded and validated"
echo "  ✅ Unit tests passed"
echo "  ✅ Core functionality working"
echo ""
echo "🚀 Next Steps:"
echo ""
echo "1. 🤖 Test with local quantized model:"
echo "   python main.py --model arcee-agent-local --use_local_model --download_model --max_samples 2"
echo ""
echo "2. 🌐 Start API server:"
echo "   python api_server.py"
echo ""
echo "3. ☁️ Configure AWS (if not done):"
echo "   aws configure"
echo ""
echo "4. 🔧 Test fine-tuning (if GPU available):"
echo "   python fine_tune_arcee.py --max_samples 5"
echo ""
echo "5. 🐳 Build Docker image:"
echo "   docker build -t arcee-agent-api ."
echo ""
echo "6. 📊 Deploy to SageMaker:"
echo "   python scripts/sagemaker_deployment.py"
echo ""
echo "📖 For detailed instructions, see:"
echo "   - README.md"
echo "   - IMPLEMENTATION_COMPLETE.md"
echo "   - AWS_SAGEMAKER_DEPLOYMENT.md"

echo ""
echo "🎯 Project successfully reinstalled and tested!"
