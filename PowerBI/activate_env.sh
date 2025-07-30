#!/bin/bash
# Fish weight prediction environment Activation script

VENV_NAME="fish_analysis_env"

if [ -d "$VENV_NAME" ]; then
    echo "Activating Fish Weight Prediction environment..."
    source "$VENV_NAME/bin/activate"
    echo "Environment activated!"
    echo ""
    echo "Available commands:"
    echo "  python scikit-learn/fish_predictive_model.py  - Run main prediction model"
    echo "  python powerbi_integration.py                 - Prepare Power BI data"
    echo "  jupyter notebook                              - Start Jupyter for development"
    echo "  deactivate                                    - Exit virtual environment"
    echo ""
else
    echo "Virtual environment not found. Please run setup.sh first."
fi
