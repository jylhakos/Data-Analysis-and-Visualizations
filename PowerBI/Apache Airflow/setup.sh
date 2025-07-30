#!/bin/bash

# Apache Airflow ML pipeline setup script
# =======================================
# This script sets up the Apache Airflow environment for machine learning pipeline orchestration.

echo "� Apache Airflow ML Pipeline Setup"
echo "===================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

# Create virtual environment
echo ""
echo "📦 Creating Python virtual environment..."
python3 -m venv venv

if [ $? -eq 0 ]; then
    echo "✅ Virtual environment created successfully"
else
    echo "❌ Failed to create virtual environment"
    echo "Please ensure Python 3 is installed"
    exit 1
fi

# Activate virtual environment
echo ""
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo ""
echo "📋 Installing required packages..."
echo "This may take a few minutes..."

pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully"
else
    echo "❌ Failed to install some packages"
    echo "You may need to install them manually"
fi

# Set Airflow home directory
export AIRFLOW_HOME=$(pwd)/airflow_home
echo ""
echo "🏠 Setting AIRFLOW_HOME to: $AIRFLOW_HOME"
mkdir -p $AIRFLOW_HOME

# Initialize Airflow database
echo ""
echo "🗄️  Initializing Airflow database..."
airflow db init

# Create admin user
echo ""
echo "👤 Creating Airflow admin user..."
airflow users create \
    --username admin \
    --firstname ML \
    --lastname Admin \
    --role Admin \
    --email admin@mlproject.com \
    --password admin

# Copy DAGs to Airflow home
echo ""
echo "📂 Setting up DAGs directory..."
mkdir -p $AIRFLOW_HOME/dags
cp -r dags/* $AIRFLOW_HOME/dags/

# Create necessary directories
mkdir -p $AIRFLOW_HOME/logs
mkdir -p $AIRFLOW_HOME/plugins
mkdir -p $AIRFLOW_HOME/ml_data
mkdir -p $AIRFLOW_HOME/ml_models

# Make scripts executable
chmod +x quickstart.sh

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python3 -c "
import pandas as pd
import numpy as np
import sklearn
import airflow
print('✅ pandas version:', pd.__version__)
print('✅ numpy version:', np.__version__)
print('✅ scikit-learn version:', sklearn.__version__)
print('✅ airflow version:', airflow.__version__)
print('✅ All libraries imported successfully!')
"

echo ""
echo "🎯 Setup complete! Here's what you can do next:"
echo ""
echo "1. Run the supervised learning pipeline:"
echo "   python supervised_regression_pipeline.py"
echo ""
echo "2. Start Airflow webserver (in background):"
echo "   ./quickstart.sh"
echo ""
echo "3. Access Airflow Web UI:"
echo "   http://localhost:8080"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "4. Run with Docker:"
echo "   docker-compose up -d"
echo ""
echo "📝 Don't forget to activate your virtual environment and set AIRFLOW_HOME:"
echo "   source venv/bin/activate"
echo "   export AIRFLOW_HOME=$(pwd)/airflow_home"
echo ""
echo "🎉 Happy fish machine learning!"
