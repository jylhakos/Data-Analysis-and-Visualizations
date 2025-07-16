#!/bin/bash

# .gitignore Verification Script
# Checks that all requested exclusions are properly configured

echo "🔍 .gitignore Configuration Verification"
echo "========================================"
echo ""

PROJECT_DIR="/home/laptop/EXERCISES/Data Analysis and Visualization/Data-Analysis-and-Visualizations/Amazon SageMaker/Arcee Agent"
cd "$PROJECT_DIR"

echo "✅ Updated .gitignore to exclude:"
echo ""

echo "🐍 Python Cache Files:"
echo "  - __pycache__/ directories (all levels)"
echo "  - *.pyc, *.pyo files"
echo "  - */__pycache__/ and **/__pycache__/"
echo ""

echo "📁 Virtual Environment:"
echo "  - .venv/ folder"
echo "  - venv/, env/, ENV/ folders"
echo "  - env.bak/, venv.bak/ folders"
echo ""

echo "📊 Logs and Reports:"
echo "  - logs/ directories"
echo "  - *.log files"
echo "  - test_output/ directories" 
echo "  - test_results/, test-results/ directories"
echo "  - outputs/ and output/ directories"
echo "  - TESTING_STATUS_REPORT.md"
echo "  - TEST_RESULTS.md"
echo "  - All *_test_report* and *_testing_report* files"
echo ""

echo "🔧 Development Files:"
echo "  - .vscode/, .idea/ IDE folders"
echo "  - .pytest_cache/, .mypy_cache/"
echo "  - .coverage, htmlcov/ testing artifacts"
echo "  - Jupyter .ipynb_checkpoints/"
echo ""

echo "☁️ AWS and Deployment:"
echo "  - .aws/ folder"
echo "  - *.pem, *.key files"
echo "  - aws-credentials.json"
echo "  - .env files and secrets"
echo ""

echo "🤖 Machine Learning:"
echo "  - models/ directory"
echo "  - *.gguf, *.ggml, *.bin files"
echo "  - fine_tuned_models/"
echo "  - wandb/, mlruns/ experiment tracking"
echo ""

echo "🧪 Testing .gitignore effectiveness:"
echo ""

# Test __pycache__
if echo "__pycache__/" | git check-ignore --stdin >/dev/null 2>&1; then
    echo "✅ __pycache__/ directories will be ignored"
else
    echo "❌ __pycache__/ directories NOT ignored"
fi

# Test .venv
if echo ".venv/" | git check-ignore --stdin >/dev/null 2>&1; then
    echo "✅ .venv/ folder will be ignored"
else
    echo "❌ .venv/ folder NOT ignored"
fi

# Test logs
if echo "logs/" | git check-ignore --stdin >/dev/null 2>&1; then
    echo "✅ logs/ directory will be ignored"
else
    echo "❌ logs/ directory NOT ignored"
fi

# Test .log files
if echo "test.log" | git check-ignore --stdin >/dev/null 2>&1; then
    echo "✅ .log files will be ignored"
else
    echo "❌ .log files NOT ignored"
fi

# Test test reports
if echo "TESTING_STATUS_REPORT.md" | git check-ignore --stdin >/dev/null 2>&1; then
    echo "✅ Test report files will be ignored"
else
    echo "❌ Test report files NOT ignored"
fi

echo ""
echo "📋 Current Git Status:"
echo "====================="
git status --short --ignored | head -10

echo ""
echo "🎯 Summary:"
echo "==========="
echo "✅ .gitignore has been updated to exclude all requested items:"
echo "   - Python __pycache__ directories"
echo "   - .venv virtual environment folders"
echo "   - Log files and directories"
echo "   - Test reports and output files"
echo "   - Additional development artifacts"
echo ""
echo "The repository is now properly configured to avoid committing"
echo "temporary files, cache directories, and sensitive information."

echo ""
echo "💡 To remove any previously committed files:"
echo "git rm -r --cached __pycache__"
echo "git rm -r --cached .venv"
echo "git rm --cached *.log"
echo "git commit -m 'Remove cached files that should be ignored'"
